import ray

from slime.ray.distillation.engine import EngineManager
from slime.ray.distillation.placement_group import create_placement_groups
from slime.ray.distillation.student_group import RayDistillationGroup
from slime.utils.arguments import parse_args


def distill_async(args):
    pgs = create_placement_groups(args)

    student_model = RayDistillationGroup(
        args=args,
        num_nodes=args.student_num_nodes,
        num_gpus_per_node=args.student_num_gpus_per_node,
        pg=pgs["student"],
        num_gpus_per_actor=0.8,
    )

    teacher_engine = EngineManager(args, pgs["teacher"])

    ids = student_model.async_init(args)
    assert len(set(ids)) == 1
    train_step = 0
    train_data_ref = None
    train_data_next_future = teacher_engine.async_generate()
    for train_step in range(args.num_train_steps):
        if train_data_next_future is not None:
            train_data_ref = ray.get(train_data_next_future)

        if train_step + 1 < args.num_train_steps:
            train_data_next_future = teacher_engine.async_generate()

        teacher_log_probs_ref = ray.get(teacher_engine.async_compute_log_probs(train_data_ref))

        ray.get(student_model.async_distill(train_data_ref, teacher_log_probs_ref))
        if train_step % args.save_interval == 0:
            ray.get(student_model.async_save_model(train_step))

        if args.eval_interval is not None and train_step % args.eval_interval == 0:
            ray.get(student_model.async_eval(train_step))


if __name__ == "__main__":
    args = parse_args()
    distill_async(args)
