from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from torch import nn

IGNORE_INDEX = -100


class PerTokenDistillationLoss(ABC, nn.Module):
    def __init__(self, use_liger=False, *args, **kwargs):
        super().__init__()
        self.use_liger = use_liger

    @abstractmethod
    def per_token_loss(self, probs, teacher_probs, inf_mask):
        pass

    @staticmethod
    def _shift_and_mask(
        gt_token_ids,
        logits,
        teacher_logprobs,
        ignore_index,
        is_shifted,
    ):
        # shift logits or hidden/_states if needed or just shift label with mask check
        shifted_labels = gt_token_ids[:, 1:].contiguous()  # label always shift
        loss_mask = (shifted_labels != ignore_index).int()
        # If all tokens are ignored, skip the loss computation,
        # Otherwise loss will be NaN
        if torch.all(1 - loss_mask):
            return torch.tensor(0.0, requires_grad=True)
        shifted_logits = logits if is_shifted else logits[:, :-1, :].contiguous()
        shifted_teacher_logprobs = teacher_logprobs if is_shifted else teacher_logprobs[:, :-1, :].contiguous()
        return shifted_labels, shifted_logits, shifted_teacher_logprobs, loss_mask

    def forward(
        self,
        gt_token_ids,
        logits,
        teacher_logprobs,
        logits_shifted=False,
        ignore_index=IGNORE_INDEX,
        temperature=1.0,
        **kwargs,
    ):
        # If the teacher and student token size is different, pad student logits to match the teacher's.
        # This only applies to cases where they share exactly the same vocab and tokenizer just
        # that teacher logit is padded for some training efficiency such as
        # https://huggingface.co/Qwen/Qwen1.5-72B-Chat/discussions/1#662883f568adf59b07b176d2
        if torch.all(gt_token_ids == -100):
            return torch.tensor(0.0, requires_grad=True, device=logits.device)

        if teacher_logprobs.shape[-1] > logits.shape[-1]:
            pad_size = teacher_logprobs.shape[-1] - logits.shape[-1]
            pad_tensor = torch.zeros((*logits.shape[:-1], pad_size), dtype=logits.dtype, device=logits.device)
            logits = torch.cat([logits, pad_tensor], dim=-1)

        if temperature != 1.0:
            logits = logits / temperature
            teacher_logits = teacher_logits / temperature
        _, shifted_logits, shifted_teacher_logprobs, loss_mask = self._shift_and_mask(
            gt_token_ids, logits, teacher_logits, ignore_index, logits_shifted
        )

        logprobs = F.log_softmax(shifted_logits, dim=-1, dtype=torch.float32)
        per_token_loss = self.per_token_loss(logprobs, shifted_teacher_logprobs)  # [B * T,]
        distill_loss = torch.sum(per_token_loss * loss_mask) / torch.sum(loss_mask)
        # Perform temperature scaling on the loss based on Hinton's 2015 paper
        # Mathematically we should perform temperature T^2 scaling on the
        # loss to compensate for the scaling of the logits during the
        # gradient computation.

        # TODO: Check if this is necessary since GKDTrainer
        # https://github.com/huggingface/trl/blob/main/trl/trainer/gkd_trainer.py#L167
        # does not perform such temperature scaling
        return distill_loss * (temperature**2)


class ForwardKLDiv(PerTokenDistillationLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loss_fn = nn.KLDivLoss(log_target=True, reduction="none")

    def per_token_loss(self, probs, teacher_logprobs):
        return self._loss_fn(probs, teacher_logprobs)


DISTILL_LOSS_MAP = {
    "forward_kl": ForwardKLDiv,
}
