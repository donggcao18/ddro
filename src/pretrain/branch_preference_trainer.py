"""Sequence DPO plus first-divergence, same-state branch DPO (TRL 0.11.4)."""

from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from trl import DPOConfig, DPOTrainer
from trl.trainer.utils import DPODataCollatorWithPadding

try:
    from .branch_preference_utils import BRANCH_COLUMNS
except ImportError:
    from branch_preference_utils import BRANCH_COLUMNS


@dataclass
class BranchDPOConfig(DPOConfig):
    preference_objective: str = "branch-dpo"
    branch_loss_weight: float = 0.1
    branch_beta: float = 0.4
    branch_schema_version: int = 1

    def __post_init__(self):
        super().__post_init__()
        if not math.isfinite(self.branch_loss_weight) or self.branch_loss_weight < 0:
            raise ValueError("branch_loss_weight must be finite and nonnegative")
        if not math.isfinite(self.branch_beta) or self.branch_beta <= 0:
            raise ValueError("branch_beta must be finite and positive")


class BranchCollator:
    def __init__(self, tokenizer, label_pad_token_id=-100):
        self.base = DPODataCollatorWithPadding(pad_token_id=tokenizer.pad_token_id,
                                              label_pad_token_id=label_pad_token_id,
                                              is_encoder_decoder=True)

    def __call__(self, features):
        batch = self.base([{k: v for k, v in feature.items() if k not in BRANCH_COLUMNS}
                           for feature in features])
        for key, dtype in (("branch_active", torch.bool), ("branch_position", torch.long),
                           ("branch_preferred_token", torch.long), ("branch_rejected_token", torch.long),
                           ("branch_loss_weight", torch.float32)):
            batch[key] = torch.tensor([feature[key] for feature in features], dtype=dtype)
        return batch


def local_branch_statistics(policy_logits, reference_logits, batch, beta):
    """Both actions use chosen-prefix logits. Normalizers cancel in their ratio."""
    rows = torch.arange(policy_logits.shape[0], device=policy_logits.device)
    positions = batch["branch_position"].clamp_min(0)
    positive = batch["branch_preferred_token"].clamp_min(0)
    negative = batch["branch_rejected_token"].clamp_min(0)
    policy_margin = (policy_logits[rows, positions, positive].float()
                     - policy_logits[rows, positions, negative].float())
    reference = reference_logits.detach()
    reference_margin = (reference[rows, positions, positive].float()
                        - reference[rows, positions, negative].float())
    logit = beta * (policy_margin - reference_margin)
    active = batch["branch_active"].to(logit.dtype)
    return F.softplus(-logit) * active, policy_margin, logit


class BranchDPOTrainer(DPOTrainer):
    def __init__(self, *args, **kwargs):
        tokenizer = kwargs["tokenizer"]
        kwargs["data_collator"] = BranchCollator(tokenizer, kwargs["args"].label_pad_token_id)
        super().__init__(*args, **kwargs)
        # Our wrapper produces the exact standard DPO fields, plus auxiliary tensors.
        self.use_dpo_data_collator = True
        if not self.is_encoder_decoder or self.ref_model is None or self.aux_loss_enabled:
            raise ValueError("BranchDPOTrainer requires an encoder-decoder and explicit frozen reference")
        if self.precompute_ref_log_probs:
            raise ValueError("Branch DPO requires reference action logits; scalar reference caches are insufficient")
        for dataset in (self.train_dataset, self.eval_dataset):
            if dataset is None:
                continue
            for row in dataset:
                if row["branch_active"]:
                    k = row["branch_position"]
                    chosen, rejected = row["chosen_labels"], row["rejected_labels"]
                    if (chosen[:k] != row["branch_prefix_tokens"] or chosen[:k] != rejected[:k]
                            or k >= min(len(chosen), len(rejected))
                            or chosen[k] != row["branch_preferred_token"]
                            or rejected[k] != row["branch_rejected_token"]):
                        raise ValueError("TRL tokenization disagrees with branch label alignment")

    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        if self.args.branch_loss_weight == 0:
            return super().get_batch_loss_metrics(model, batch, train_eval)
        policy = self.concatenated_forward(model, batch)
        with torch.no_grad():
            reference = self.concatenated_forward(self.ref_model, batch)
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(policy[0], policy[1], reference[0], reference[1])
        local, margin, branch_logit = local_branch_statistics(policy[2], reference[2], batch, self.args.branch_beta)
        branch_loss = (local * batch["branch_loss_weight"]).mean()
        sequence_loss = losses.mean()
        if self.args.rpo_alpha is not None:
            sequence_loss = sequence_loss + policy[4] * self.args.rpo_alpha
        active = batch["branch_active"].float()
        denominator = active.sum().clamp_min(1)
        prefix = "eval_" if train_eval == "eval" else ""
        values = {"loss/sequence": sequence_loss, "loss/branch": branch_loss,
                  "branch/active_fraction": active.mean(),
                  "branch/action_accuracy": ((margin > 0).float() * active).sum() / denominator,
                  "branch/action_margin": (margin * active).sum() / denominator,
                  "branch/preference_margin": (branch_logit * active).sum() / denominator,
                  "rewards/chosen": chosen_rewards.mean(), "rewards/rejected": rejected_rewards.mean(),
                  "rewards/accuracies": (chosen_rewards > rejected_rewards).float().mean(),
                  "rewards/margins": (chosen_rewards - rejected_rewards).mean(),
                  "logps/chosen": policy[0].mean(), "logps/rejected": policy[1].mean()}
        metrics = {prefix + key: value.detach().cpu() for key, value in values.items()}
        return sequence_loss + self.args.branch_loss_weight * branch_loss, metrics
