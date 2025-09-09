import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from slime.backends.utils.data import process_rollout_data
from slime.ray.train_actor import TrainRayActor
from slime.utils.distributed_utils import get_gloo_group
from slime.utils.timer import Timer

from .update_weight_utils import UpdateWeightFromTensor


class FSDPDistillationRayActor(TrainRayActor):
    """Simplified TrainRayActor for pure HF+FSDP training.

    Responsibilities:
      * Initialize model/tokenizer on rank0 sequentially to avoid race on cache
      * Wrap model with FSDP
      * Provide minimal train / save / update_weights hooks compatible with existing RayTrainGroup

    Weight update strategy:
      * Rank0 gathers state_dict (full) and broadcasts tensor-by-tensor.
      * For small models this is fine; for larger models consider sharded state_dict type.
    """

    def init(self, args, role, wandb_run_id, with_ref: bool = False):  # type: ignore[override]
        super().init(args, role, wandb_run_id, with_ref)
        self.args = args
        torch.manual_seed(args.seed)

        # Serialize tokenizer/config loading across ranks to avoid HF cache race
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                self.hf_config = AutoConfig.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
            dist.barrier(group=get_gloo_group())

        # Load model
        with torch.device(f"cuda:{torch.cuda.current_device()}"):
            model = AutoModelForCausalLM.from_pretrained(
                self.args.hf_checkpoint,
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

        # TODO: load

        self.ref_model = None
        # TODO: support ref model
        if with_ref:
            raise NotImplementedError()

        self.weight_updator = UpdateWeightFromTensor(self.args, self.model)

        if self.args.offload:
            self.sleep(("model"))

        Timer().start("train_wait")
        self.global_step = 0
        self.micro_step = 0
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

        teacher_log_probs = process_rollout_data(self.args, teacher_log_probs_ref, rank, world_size)
        padded_teacher_log_probs = self.pad_and_move_to_device(teacher_log_probs)

        for batch in padded_batches:
            batch["teacher_log_probs"] = padded_teacher_log_probs["log_probs"]

        return


def gather_log_probs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    # log_probs: [B, T-1, V]; input_ids: [B, T]
    pred_logits = logits[:, :-1]
    log_probs_all = torch.log_softmax(pred_logits, dim=-1)
    tgt = input_ids[:, 1:].contiguous()
    log_probs = log_probs_all.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return log_probs


def per_sample_mean(x, loss_mask):
    # TODO: impl per token loss
    return ((x * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp_min(1)).mean()
