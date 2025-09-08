import logging
import multiprocessing
import random
import time
from typing import Union

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from slime.backends.sglang_utils.sglang_engine import SGLangEngine
from slime.ray.rollout_data_source import RolloutDataSourceWithBuffer
from slime.utils.http_utils import find_available_port, get_host_info, post, run_router
from slime.utils.ray_utils import Box
from slime.utils.types import Sample

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def launch_engines(args, pg):

    num_gpu_per_engine = min(args.engine_num_gpus_per_engine, args.engine_num_gpus_per_node)
    num_engines = args.engine_num_gpus // num_gpu_per_engine

    pg, reordered_bundle_indices = pg

    EngineRayActor = ray.remote(SGLangEngine)

    engines = []
    for i in range(num_engines):
        num_gpus = 0.2
        num_cpus = num_gpus

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=reordered_bundle_indices[i * num_gpu_per_engine],
        )

        engines.append(
            EngineRayActor.options(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                runtime_env={"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}},
            ).remote(args, rank=i)
        )

    # get ports
    # there are 4 ports we need to allocate
    # 1. server port
    # 2. nccl port
    # 3. dist_init_addr port
    # 4. other ports for dp_attention, which is of size 4 + dp_size
    num_engines_per_node = max(
        1, min(args.engine_num_gpus_per_node, args.engine_num_gpus) // args.engine_num_gpus_per_engine
    )
    addr_and_ports = [{} for _ in range(num_engines)]
    for rank, engine in enumerate(engines):
        if rank % num_engines_per_node != 0:
            continue

        def get_addr_and_ports():
            # use small ports to prevent ephemeral port between 32768 and 65536.
            start_port = 10000

            def port(consecutive=1):
                nonlocal start_port
                _, port = ray.get(
                    engine._get_current_node_ip_and_free_port.remote(
                        start_port=start_port,
                        consecutive=consecutive,
                    )
                )
                start_port = port + consecutive
                return port

            def addr():
                addr, _ = ray.get(engine._get_current_node_ip_and_free_port.remote())
                return addr

            return addr, port

        get_addr, get_port = get_addr_and_ports()

        for i in range(num_engines_per_node):
            addr_and_ports[rank + i]["port"] = get_port()
            addr_and_ports[rank + i]["nccl_port"] = get_port()

        if args.engine_num_gpus_per_engine > args.engine_num_gpus_per_node:
            num_node_per_engine = args.engine_num_gpus_per_engine // args.engine_num_gpus_per_node
            if rank % num_node_per_engine == 0:
                # this is the first node in the engine, we need to allocate the dist_init_addr port
                dist_init_addr = f"{get_addr()}:{get_port(6 + args.sglang_dp_size)}"
                for i in range(num_node_per_engine):
                    addr_and_ports[rank + i]["dist_init_addr"] = dist_init_addr
        else:
            for i in range(num_engines_per_node):
                addr_and_ports[rank + i]["dist_init_addr"] = f"{get_addr()}:{get_port(6 + args.sglang_dp_size)}"

    for i in range(num_engines):
        for key in ["port", "nccl_port", "dist_init_addr"]:
            assert key in addr_and_ports[i], f"Engine {i} {key} is not set."
        print(f"Ports for engine {i}: {addr_and_ports[i]}")

    # TODO: don't ray.get here to overlap train actor init with rollout engine init.
    # somehow if we don't sync here, the --debug-rollout-only mode will crash.
    init_handles = [engine.init.remote(**ports) for engine, ports in zip(engines, addr_and_ports)]
    ray.get(init_handles)

    if args.offload:
        ray.get([engine.release_memory_occupation.remote() for engine in engines])

    return engines


def launch_router(args):
    if args.sglang_router_ip is not None:
        return

    from sglang_router.launch_router import RouterArgs

    args.sglang_router_ip = get_host_info()[1]
    args.sglang_router_port = find_available_port(random.randint(3000, 4000))

    router_args = RouterArgs(
        host=args.sglang_router_ip,
        port=args.sglang_router_port,
        balance_abs_threshold=0,
        prometheus_port=find_available_port(random.randint(4000, 5000)),
    )

    if hasattr(router_args, "log_level"):
        router_args.log_level = "warn"

    if hasattr(router_args, "request_timeout_secs"):
        router_args.request_timeout_secs = args.sglang_router_request_timeout_secs

    process = multiprocessing.Process(
        target=run_router,
        args=(router_args,),
    )
    process.daemon = True  # Set the process as a daemon
    process.start()
    # Wait 3 seconds
    time.sleep(3)
    assert process.is_alive()
    # If router ip is specified, use the specified launched router
    print(f"SGLang router launched at {args.sglang_router_ip}:{args.sglang_router_port}")


@ray.remote
class EngineController:
    """The class to run rollout and convert rollout data to training data."""

    def __init__(self, args):
        self.args = args

        self.data_source = RolloutDataSourceWithBuffer(args)

    def generate(self):
        """
        For now we only support supervised distillation, so we don't need to generate data,
        rather we take a batch directly from the buffer.
        """
        data = self.data_source.get_samples(self.args.train_batch_size)
        data = self._convert_samples_to_train_data(data)
        return Box(ray.put(data))

    def _convert_samples_to_train_data(self, samples: Union[list[Sample], list[list[Sample]]]):
        """
        Apply any remaining post-processing to the samples.
        """

        return samples

    def compute_log_probs(self, data_ref):
        # TODO (MrAta): fix and complete this implementation
        data = ray.get(data_ref.inner)
        url = f"http://{self.args.sglang_router_ip}:{self.args.sglang_router_port}/generate"
        sampling_params = {
            "max_new_tokens": 1,
            "return_logprob": True,
            "num_logprobs": 1024,
            # TODO: add other sampling params
        }
        payload = {
            "input_ids": data["tokens"],
            "sampling_params": sampling_params,
            "return_logprob": True,
        }
        import asyncio

        output = asyncio.run(post(url, payload, use_http2=self.args.use_http2))
        return ray.put(Box(output))


class EngineManager:
    def __init__(self, args, pg):
        self.args = args
        launch_router(args)
        self.controller = EngineController.options(
            num_cpus=1,
            num_gpus=0,
        ).remote(args)

        self.all_engines = launch_engines(args, pg)
        nodes_per_engine = max(1, args.engine_num_gpus_per_engine // args.engine_num_gpus_per_node)
        # when doing multi-node serving, we will only send request to node-0 for each engine.
        self.engines = self.all_engines[::nodes_per_engine]

    def async_generate(self):
        return self.controller.generate.remote()

    def async_compute_log_probs(self, data_ref):
        return self.controller.compute_log_probs.remote(data_ref)
