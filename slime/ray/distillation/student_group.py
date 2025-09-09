from typing import Dict, Optional
import ray

from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from slime.backends.fsdp_utils import FSDPDistillationRayActor
from slime.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST


class RayDistillationGroup:
    """
    A group of ray actors for distillation

    Args:
        args (Namespace): Arguments for the actor group.
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): Number of gpus for this actor group.
        pg (PlacementGroup, optional): Placement group to schedule actor on.
            If none, create new placement group automatically. Defaults to None.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each actor.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
        resources (Dict[str, float], optional): Custom resources to allocate for each actor.
            See https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        num_resources_per_node (int, optional): Number of custom resources to allocate for each node.
            See https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
    """

    def __init__(
        self,
        args,
        num_nodes,
        num_gpus_per_node,
        pg: tuple[PlacementGroup, list[int]],
        num_gpus_per_actor: float = 1,
        resources: Optional[Dict[str, float] | None] = None,
        num_resources_per_node: Optional[int | None] = None,
    ) -> None:
        self.args = args
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node

        # custom resources, see https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        self._resources = resources
        self._num_resources_per_node = num_resources_per_node

        # Allocate the GPUs for actors w/o instantiating them
        self._allocate_gpus_for_actor(pg, num_gpus_per_actor)

    def _allocate_gpus_for_actor(self, pg, num_gpus_per_actor):
        world_size = self._num_nodes * self._num_gpus_per_node

        # Use placement group to lock resources for models of same type
        assert pg is not None
        pg, reordered_bundle_indices = pg

        env_vars = {
            # because sglang will always set NCCL_CUMEM_ENABLE to 0
            # we need also set it to 0 to prevent nccl error.
            "NCCL_CUMEM_ENABLE": "0",
            **{name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST},
        }

        TrainRayActor = ray.remote(
            num_gpus=1,
            runtime_env={"env_vars": env_vars},
        )(FSDPDistillationRayActor)

        # Create worker actors
        self._actor_handlers = []
        master_addr, master_port = None, None
        for rank in range(world_size):
            actor = TrainRayActor.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=reordered_bundle_indices[rank],
                ),
            ).remote(world_size, rank, master_addr, master_port)
            if rank == 0:
                master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
            self._actor_handlers.append(actor)

    def async_init(self, args):
        """
        Allocate GPU resourced and initialize model, optimzier, local ckpt, etc.
        """
        self.args = args
        return [actor.init.remote(args) for actor in self._actor_handlers]

    def async_distill(self, rollout_data_ref, teacher_log_probs_ref):
        """Do one rollout training"""
        return [actor.distill.remote(rollout_data_ref, teacher_log_probs_ref) for actor in self._actor_handlers]

    def async_save_model(self, step_id):
        """Save actor model on rank 0."""
        return [actor.save_model.remote(step_id) for actor in self._actor_handlers]

    def async_eval(self, step_id):
        """Evaluate actor model on rank 0."""
        return [actor.eval.remote(step_id) for actor in self._actor_handlers]
