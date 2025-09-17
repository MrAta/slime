import os
from datetime import timedelta

import ray
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from slime.backends.utils.data import process_rollout_data
from slime.utils.distributed_utils import get_gloo_group, init_gloo_group
from slime.utils.http_utils import is_port_available


def get_local_gpu_id():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cvd is None:
        return ray.get_gpu_ids()[0]
    else:
        return cvd.split(",").index(str(ray.get_gpu_ids()[0]))


class FSDPDistillationRayActor:
    @staticmethod
    def _get_current_node_ip_and_free_port(start_port=10000, consecutive=1):
        address = ray._private.services.get_node_ip_address()
        # strip ipv6 address
        address = address.strip("[]")

        # find the port where port, port + 1, port + 2, ... port + consecutive - 1 are all available
        port = start_port
        while not all(is_port_available(port + i) for i in range(consecutive)):
            port += 1

        return address, port

    def get_master_addr_and_port(self):
        return self.master_addr, self.master_port

    def __init__(self, world_size, rank, master_addr, master_port):
        self._world_size = world_size
        self._rank = rank
        if master_addr:
            self.master_addr, self.master_port = master_addr, master_port
        else:
            self.master_addr, self.master_port = self._get_current_node_ip_and_free_port(start_port=20000)

        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        # TODO: currently this doesn't work as ray has already set torch.cuda.device_count().
        # os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        # os.environ["LOCAL_RANK"] = str(ray.get_gpu_ids()[0])
        os.environ["LOCAL_RANK"] = str(get_local_gpu_id())

    def init(self, args):
        self.args = args
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(f"cuda:{local_rank}")

        dist.init_process_group(
            backend=args.distributed_backend,
            timeout=timedelta(minutes=args.distributed_timeout_minutes),
        )
        init_gloo_group()

        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()

        # set current device
        args.local_rank = args.rank % torch.cuda.device_count()
        torch.cuda.set_device(f"cuda:{args.local_rank}")

        torch.manual_seed(args.seed)

        # Serialize tokenizer/config loading across ranks to avoid HF cache race
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                self.hf_config = AutoConfig.from_pretrained(self.args.model_path, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, trust_remote_code=True)
            dist.barrier(group=get_gloo_group())

        # Load model
        with torch.device(f"cuda:{torch.cuda.current_device()}"):
            model = AutoModelForCausalLM.from_pretrained(
                self.args.model_path,
                trust_remote_code=True,
            )
        model.train()

        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # TODO: set correct auto_wrap_policy
        auto_wrap_policy = None

        self.model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
            sharding_strategy=ShardingStrategy[self.args.fsdp_sharding_strategy],
            cpu_offload=self.args.fsdp_cpu_offload,
            forward_prefetch=self.args.fsdp_forward_prefetch,
            backward_prefetch=self.args.fsdp_backward_prefetch,
            limit_all_gathers=self.args.fsdp_limit_all_gathers,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )

        return 0

    def save_model(self, iteration, with_optimizer=True):
        if self.args.debug_rollout_only:
            return

        raise NotImplementedError()

    def distill(self, rollout_data_ref, teacher_log_probs_ref):  # type: ignore[override]

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        train_data = process_rollout_data(self.args, rollout_data_ref, rank, world_size)
        padded_batches = self.pad_and_move_to_device(train_data)
        student_logits = self.model(padded_batches)

        teacher_log_probs = ray.get(teacher_log_probs_ref)

        padded_teacher_log_probs = self.pad_and_move_to_device(teacher_log_probs)

        # TODO: compute loss and update the model.
        print(padded_teacher_log_probs)

        return

    def eval(self):
        raise NotImplementedError
