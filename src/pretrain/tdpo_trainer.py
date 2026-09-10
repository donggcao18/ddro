"""Encoder-decoder TDPO adapted from Token-level-Direct-Preference-Optimization/trainers.py.

Retains the TDPO1/TDPO2 objectives while using TRL 0.11.4's data and Trainer
integration. Decoder labels are already aligned with T5 logits.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from trl import DPOConfig, DPOTrainer


@dataclass
class PreferenceConfig(DPOConfig):
    preference_objective: str = "dpo"
    tdpo_alpha: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        if self.preference_objective not in {"dpo", "tdpo1", "tdpo2"}:
            raise ValueError("preference_objective must be dpo, tdpo1, or tdpo2")
        if not math.isfinite(self.tdpo_alpha) or self.tdpo_alpha < 0:
            raise ValueError("tdpo_alpha must be finite and non-negative")


def sequence_statistics(policy_logits, reference_logits, labels, label_pad_token_id=-100):
    """Return summed log-ratios, forward KL(ref || policy), and policy logps."""
    if (
        policy_logits.shape != reference_logits.shape
        or policy_logits.shape[:-1] != labels.shape
    ):
        raise ValueError("Policy/reference logits and decoder labels must be aligned")
    mask = labels != label_pad_token_id
    if not mask.any(dim=-1).all():
        raise ValueError("Each response must contain at least one unmasked target token")
    safe_labels = labels.masked_fill(~mask, 0)

    # Use log_softmax directly: softmax().log() can underflow for rare tokens.
    policy_logps = F.log_softmax(policy_logits, dim=-1, dtype=torch.float32)
    with torch.no_grad():
        reference_logps = F.log_softmax(reference_logits, dim=-1, dtype=torch.float32)
        reference_probs = reference_logps.exp()
    token_policy = policy_logps.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_reference = reference_logps.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    position_kl = (reference_probs * (reference_logps - policy_logps)).sum(-1)
    return (
        (token_policy - token_reference).masked_fill(~mask, 0).sum(-1),
        position_kl.masked_fill(~mask, 0).sum(-1),
        token_policy.masked_fill(~mask, 0).sum(-1),
    )


def tdpo_loss(chosen_margin, rejected_margin, chosen_kl, rejected_kl, beta, alpha, objective):
    if objective == "tdpo1":
        correction = rejected_kl - chosen_kl
    elif objective == "tdpo2":
        correction = alpha * (rejected_kl - chosen_kl.detach())
    else:
        raise ValueError("TDPO objective must be tdpo1 or tdpo2")
    logits = chosen_margin - rejected_margin - correction
    losses = -F.logsigmoid(beta * logits)
    # Match the reference implementation's diagnostic rewards for both variants.
    chosen_rewards = beta * (chosen_margin + chosen_kl).detach()
    rejected_rewards = beta * (rejected_margin + rejected_kl).detach()
    return losses, chosen_rewards, rejected_rewards


class TokenDPOTrainer(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.is_encoder_decoder or self.ref_model is None:
            raise ValueError("TokenDPOTrainer requires an encoder-decoder policy and explicit reference")
        if self.args.preference_objective not in {"tdpo1", "tdpo2"}:
            raise ValueError("TokenDPOTrainer requires tdpo1 or tdpo2")
        if (
            self.precompute_ref_log_probs
            or self.reference_free
            or self.args.rpo_alpha is not None
            or self.args.sync_ref_model
            or self.label_smoothing != 0
            or self.loss_type != "sigmoid"
        ):
            raise ValueError("TDPO requires live reference logits and an unmodified pairwise objective")
        self.ref_model.requires_grad_(False)
        self.ref_model.eval()

    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        combined = self.concatenated_inputs(
            batch,
            is_encoder_decoder=True,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        labels = combined["concatenated_labels"]
        model_inputs = {
            "input_ids": combined["concatenated_input_ids"],
            "attention_mask": combined["concatenated_attention_mask"],
            "labels": labels,
            "use_cache": False,
        }
        policy_logits = model(**model_inputs).logits
        with torch.no_grad():
            reference_logits = self.ref_model(**model_inputs).logits
        margins, kls, logps = sequence_statistics(
            policy_logits, reference_logits, labels, self.label_pad_token_id
        )
        n = batch["chosen_labels"].shape[0]
        losses, chosen_rewards, rejected_rewards = tdpo_loss(
            margins[:n], margins[n:], kls[:n], kls[n:],
            self.beta, self.args.tdpo_alpha, self.args.preference_objective,
        )
        prefix = "eval_" if train_eval == "eval" else ""
        metrics = {
            "rewards/chosen": chosen_rewards.mean(),
            "rewards/rejected": rejected_rewards.mean(),
            "rewards/accuracies": (chosen_rewards > rejected_rewards).float().mean(),
            "rewards/margins": (chosen_rewards - rejected_rewards).mean(),
            "logps/chosen": logps[:n].detach().mean(),
            "logps/rejected": logps[n:].detach().mean(),
            "kl/chosen": kls[:n].detach().mean(),
            "kl/rejected": kls[n:].detach().mean(),
            "kl/rejected_minus_chosen": (kls[n:] - kls[:n]).detach().mean(),
            # TRL prediction_step reads these diagnostics during evaluation.
            "logits/chosen": policy_logits[:n].detach().mean(),
            "logits/rejected": policy_logits[n:].detach().mean(),
        }
        return losses.mean(), {prefix + key: value.cpu() for key, value in metrics.items()}
