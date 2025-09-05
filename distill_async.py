import ray
from slime.ray.placement_group import create_actor_group, create_placement_groups, create_rollout_manager
from slime.utils.arguments import parse_args
from slime.utils.wandb_utils import init_wandb_primary

def distill_async(args):
    pgs = create_placement_groups(args)
    wandb_run_id = init_wandb_primary(args)

    student_model = create_actor_group(args, pgs["student"], wandb_run_id=wandb_run_id)

    # create the rollout manager, with sglang engines inside.
    rollout_manager = create_rollout_manager(args, pgs["rollout"], wandb_run_id=wandb_run_id)

    assert not args.offload and not args.colocate, "Offload and colocate are not supported for full async RL training."
