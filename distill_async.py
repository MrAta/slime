import ray
from slime.ray.distillation.placement_group import create_placement_groups, create_student_group, create_teacher_engine
from slime.utils.arguments import parse_args

def distill_async(args):
    pgs = create_placement_groups(args)

    student_model = create_student_group(args, pgs["student"])

    # create the rollout manager, with sglang engines inside.
    teacher_engine = create_teacher_engine(args, pgs["teacher"])

    ids = student_model.async_init(args)
    assert len(set(ids)) == 1
    train_step = 0
    for train_step in range(args.num_train_steps):
        rollout_data_ref = ray.get(teacher_engine.async_generate(train_step))
        teacher_log_probs_ref = ray.get(teacher_engine.async_compute_log_probs(rollout_data_ref))

        ray.get(student_model.async_distill(rollout_data_ref, teacher_log_probs_ref))
        if train_step % args.save_interval == 0:
            ray.get(student_model.async_save_model(train_step))

        if args.eval_interval is not None and train_step % args.eval_interval == 0:
            ray.get(student_model.async_eval(train_step))


if __name__ == "__main__":
    args = parse_args()
    distill_async(args)