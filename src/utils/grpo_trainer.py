"""Encoder-decoder GRPO using Transformers Trainer for optimization and DDP.

Each microbatch uses fresh rollouts, with one loss evaluation per rollout.
Sampling is the on-policy GRPO regime; deterministic beam rollouts are an
experimental selection-biased variant. No reward or value model is used.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from math import isfinite
from pathlib import Path
from typing import Any, Sequence
import unicodedata
import weakref

import torch
from torch import nn
from transformers import GenerationConfig, Trainer, TrainerCallback

from utils.trie import Trie
from utils.retrieval_metrics import score_ranking


@dataclass
class GRPOSettings:
    num_generations: int = 8
    do_sample: bool = True
    temperature: float = 1.0
    max_target_length: int = 128
    epsilon: float = 0.2
    beta: float = 0.0
    eval_num_beams: int = 10
    score_normalization: str = "trie"
    ranking_loss_weight: float = 0.0

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in (self.temperature, self.epsilon, self.beta, self.ranking_loss_weight)):
            raise ValueError("temperature, epsilon, beta, and ranking_loss_weight must be finite")
        if self.num_generations < 2:
            raise ValueError("num_generations must be at least 2")
        if self.temperature <= 0 or self.max_target_length <= 0:
            raise ValueError("temperature and max_target_length must be positive")
        if not 0 < self.epsilon < 1 or self.beta < 0 or self.eval_num_beams < 1:
            raise ValueError("Require 0 < epsilon < 1, beta >= 0, and eval_num_beams >= 1")
        if self.score_normalization not in {"trie", "full_vocab"}:
            raise ValueError("score_normalization must be trie or full_vocab")
        if self.do_sample and self.score_normalization != "trie":
            raise ValueError("Sampled GRPO requires score_normalization=trie to match its constrained sampling distribution")
        if self.ranking_loss_weight < 0:
            raise ValueError("ranking_loss_weight must be nonnegative")

    @property
    def scoring_temperature(self) -> float:
        # Transformers does not apply temperature during deterministic decoding.
        return self.temperature if self.do_sample else 1.0

    def rollout_generation_config(self, index: TargetIndex) -> GenerationConfig:
        if not self.do_sample and self.num_generations > len(index.sequence_to_target):
            raise ValueError("Deterministic rollouts require num_generations <= usable corpus targets; "
                             "reduce num_generations or use do_sample=true")
        if not self.do_sample:
            return beam_generation_config(index, self.max_target_length, self.num_generations,
                                          self.num_generations, self.score_normalization)
        return GenerationConfig(
            decoder_start_token_id=index.start_id,
            pad_token_id=index.pad_id, eos_token_id=index.eos_id,
            max_new_tokens=self.max_target_length,
            do_sample=self.do_sample,
            num_beams=1 if self.do_sample else self.num_generations,
            num_return_sequences=self.num_generations,
            temperature=self.scoring_temperature,
            top_k=0 if self.do_sample else 50, top_p=1.0,
            length_penalty=1.0 if self.do_sample else 0.0, use_cache=True,
            renormalize_logits=True,
        )


def group_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    """Population std makes constant groups exactly zero, including small groups."""
    grouped = rewards.float().reshape(-1, group_size)
    centered = grouped - grouped.mean(dim=1, keepdim=True)
    return (centered / (grouped.std(dim=1, unbiased=False, keepdim=True) + 1e-4)).flatten()


def completion_mask(tokens: torch.Tensor, eos_id: int, pad_id: int) -> torch.Tensor:
    """Include the first EOS, exclude decoder start, padding, and all later tokens."""
    eos = tokens.eq(eos_id)
    return (eos.cumsum(dim=1) - eos.long()).eq(0) & tokens.ne(pad_id)


def grpo_loss(
    logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float,
    beta: float = 0.0,
    reference_logps: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    ratio = (logps - old_logps).exp()
    advantage = advantages[:, None]
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - epsilon, 1 + epsilon) * advantage)
    kl = torch.zeros_like(logps)
    if beta:
        if reference_logps is None:
            raise ValueError("A frozen reference is required for beta > 0")
        delta = torch.where(mask.bool(), reference_logps - logps, 0.)
        # exp(delta) - 1 loses precision near zero. Keep the small KL term
        # separate from the O(1), signed advantages until after averaging.
        kl = torch.expm1(delta) - delta
    # Original GRPO: average tokens within each trajectory, then trajectories.
    # Even constant-reward groups retain a connected zero loss for DDP backward.
    lengths = mask.sum(1).clamp_min(1)
    policy_loss = -((surrogate * mask).sum(1) / lengths).mean()
    mean_kl = ((kl * mask).sum(1) / lengths).mean()
    return policy_loss + beta * mean_kl, mean_kl


class TargetIndex:
    """An exact target lookup and trie, shared by sampling and probability scoring."""

    def __init__(self, tokenizer: Any, config: Any, targets: Sequence[str], max_length: int,
                 invalid_target_policy: str = "error"):
        if invalid_target_policy not in {"error", "skip"}:
            raise ValueError("invalid_target_policy must be error or skip")
        self.start_id = config.decoder_start_token_id
        self.pad_id = tokenizer.pad_token_id
        self.eos_id = tokenizer.eos_token_id
        if any(value is None for value in (self.start_id, self.pad_id, self.eos_id)):
            raise ValueError("Checkpoint must define decoder_start, pad, and EOS token IDs")
        if self.pad_id == self.eos_id:
            raise ValueError("This seq2seq trainer requires distinct pad and EOS tokens")
        if config.eos_token_id != self.eos_id or config.pad_token_id != self.pad_id:
            raise ValueError("Model config and tokenizer pad/EOS token IDs must agree")
        self.sequence_to_target: dict[tuple[int, ...], str] = {}
        self.target_to_sequence: dict[str, tuple[int, ...]] = {}
        self.excluded_targets: dict[str, dict[str, Any]] = {}
        candidates: dict[tuple[int, ...], list[str]] = {}
        unique_targets = sorted(set(targets))

        def exclude(target: str, reason: str, message: str, **details: Any) -> None:
            if invalid_target_policy == "error":
                raise ValueError(message)
            self.excluded_targets[target] = {"reason": reason, **details}

        for target in unique_targets:
            if any(unicodedata.category(char) in {"Cc", "Cs"} or char == "\ufffd" for char in target):
                exclude(target, "invalid_characters", f"Target contains a control, surrogate, or replacement character: {target!r}")
                continue
            try:
                sequence = tuple(tokenizer(target, add_special_tokens=True, truncation=False)["input_ids"])
            except (ValueError, UnicodeError) as error:
                exclude(target, "tokenization_error", f"Cannot tokenize target {target!r}: {error}", error_type=type(error).__name__)
                continue
            if not sequence or sequence[-1] != self.eos_id or self.eos_id in sequence[:-1]:
                exclude(target, "invalid_eos", f"Target must tokenize to exactly one terminal EOS: {target!r}")
                continue
            bos_id = getattr(tokenizer, "bos_token_id", None)
            if self.pad_id in sequence or (
                tokenizer.unk_token_id is not None and tokenizer.unk_token_id in sequence
            ) or (bos_id is not None and bos_id in sequence[1:]):
                exclude(target, "reserved_or_unknown_token", f"Target contains a reserved, padding or unknown token: {target!r}")
                continue
            if len(sequence) > max_length:
                exclude(target, "overlong", f"Target {target!r} needs {len(sequence)} tokens; increase max_target_length={max_length}",
                        token_length=len(sequence), max_target_length=max_length)
                continue
            candidates.setdefault(sequence, []).append(target)
        collisions = 0
        for sequence, group in candidates.items():
            if len(group) > 1:
                # Remove every ambiguous spelling, including the first one.
                # Keeping an arbitrary representative would corrupt relevance.
                for target in group:
                    exclude(target, "tokenizer_collision", f"Tokenizer collision between targets {group[0]!r} and {group[1]!r}",
                            collision_group=collisions, token_length=len(sequence))
                collisions += 1
                continue
            target = group[0]
            self.sequence_to_target[sequence] = target
            self.target_to_sequence[target] = sequence
        self.filtering_summary = {
            "policy": invalid_target_policy, "input_targets": len(unique_targets),
            "kept_targets": len(self.target_to_sequence), "excluded_targets": len(self.excluded_targets),
            "excluded_by_reason": dict(Counter(item["reason"] for item in self.excluded_targets.values())),
            "collision_groups": collisions, "max_target_length": max_length,
        }
        if not self.sequence_to_target:
            raise ValueError(f"No usable targets remain for the corpus trie: {self.filtering_summary}")
        self.trie = Trie([[self.start_id, *sequence] for sequence in self.sequence_to_target])

    def write_filter_report(self, output_dir: str | Path) -> None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "target_filtering.json").write_text(
            json.dumps(self.filtering_summary, indent=2) + "\n", encoding="utf-8"
        )
        with (destination / "excluded_targets.jsonl").open("w", encoding="utf-8") as handle:
            for target, details in self.excluded_targets.items():
                # Escaping also makes isolated surrogates safe to audit.
                handle.write(json.dumps({"target": target, **details}) + "\n")

    def canonical(self, sequence: Sequence[int]) -> tuple[int, ...]:
        tokens = list(sequence)
        if tokens and tokens[0] == self.start_id:
            tokens = tokens[1:]
        if self.eos_id in tokens:
            tokens = tokens[:tokens.index(self.eos_id) + 1]
        elif self.pad_id in tokens:
            tokens = tokens[:tokens.index(self.pad_id)]
        return tuple(tokens)

    def allowed(self, batch_id: int, prefix: torch.Tensor) -> list[int]:
        tokens = prefix.tolist()
        allowed = self.trie.get(tokens)
        if allowed:
            return allowed
        if self.eos_id in tokens[1:]:
            return [self.pad_id]
        # Beam search may carry -inf placeholder beams when fewer continuations
        # exist than requested beams. Terminate those; exact lookup rejects them.
        # Independent sampling cannot enter this branch from a valid trie root.
        return [self.eos_id]

    def rewards(self, sequences: torch.Tensor, positives: Sequence[Sequence[str]], group_size: int) -> torch.Tensor:
        values = []
        if len(sequences) != len(positives) * group_size:
            raise ValueError("Rollout count does not match query groups")
        positive_sets = [set(group) for group in positives]
        for i, sequence in enumerate(sequences.detach().cpu().tolist()):
            target = self.sequence_to_target.get(self.canonical(sequence))
            values.append(float(target is not None and target in positive_sets[i // group_size]))
        return torch.tensor(values, dtype=torch.float32, device=sequences.device)

    def score_layout(self, sequences: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sparse allowed-token indices avoid a dense [batch, time, vocab] mask."""
        rows = sequences.detach().cpu().tolist()
        layouts: list[list[list[int]]] = []
        choices: list[list[int]] = []
        maximum = 1
        for row in rows:
            paths, positions = [], []
            ended = False
            for t, token in enumerate(row[1:]):
                if ended:
                    allowed, position = [self.pad_id], 0
                else:
                    allowed = self.trie.get(row[:t + 1])
                    if token not in allowed:
                        raise ValueError(f"Sampled token {token} is outside the trie at {row[:t + 1]}")
                    position = allowed.index(token)
                paths.append(allowed)
                positions.append(position)
                maximum = max(maximum, len(allowed))
                ended = ended or token == self.eos_id
            layouts.append(paths)
            choices.append(positions)
        shape = (len(rows), sequences.shape[1] - 1, maximum)
        indices = torch.full(shape, self.pad_id, dtype=torch.long)
        valid = torch.zeros(shape, dtype=torch.bool)
        for i, paths in enumerate(layouts):
            for t, allowed in enumerate(paths):
                indices[i, t, :len(allowed)] = torch.tensor(allowed)
                valid[i, t, :len(allowed)] = True
        return indices.to(sequences.device), valid.to(sequences.device), torch.tensor(choices, device=sequences.device)


def token_logps(
    logits: torch.Tensor,
    layout: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    temperature: float,
    normalization: str = "trie",
) -> torch.Tensor:
    indices, valid, choices = layout
    if normalization == "full_vocab":
        # Legacy beam search masks invalid tokens after full-vocabulary softmax.
        ids = indices.gather(-1, choices.unsqueeze(-1)).squeeze(-1)
        full_logits = logits.float()
        if temperature != 1.0:
            full_logits = full_logits / temperature
        return full_logits.gather(-1, ids.unsqueeze(-1)).squeeze(-1) - full_logits.logsumexp(-1)
    if normalization != "trie":
        raise ValueError("normalization must be trie or full_vocab")
    # Cast only gathered logits to FP32, keeping the full vocabulary tensor in
    # model precision. Apply exactly the sampler's trie mask and temperature.
    allowed_logits = logits.gather(-1, indices).float() / temperature
    allowed_logits = allowed_logits.masked_fill(~valid, -torch.inf)
    chosen = allowed_logits.gather(-1, choices.unsqueeze(-1)).squeeze(-1)
    return chosen - allowed_logits.logsumexp(-1)


def constrained_logps(logits: torch.Tensor, layout: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                      temperature: float) -> torch.Tensor:
    """Backward-compatible name for trie-normalized probability scoring."""
    return token_logps(logits, layout, temperature, "trie")


def beam_generation_config(index: TargetIndex, max_length: int, beams: int, returned: int,
                           normalization: str = "trie") -> GenerationConfig:
    if normalization not in {"trie", "full_vocab"}:
        raise ValueError("normalization must be trie or full_vocab")
    return GenerationConfig(
        decoder_start_token_id=index.start_id, pad_token_id=index.pad_id, eos_token_id=index.eos_id,
        max_new_tokens=max_length, do_sample=False, num_beams=beams, num_return_sequences=returned,
        length_penalty=0.0 if beams > 1 else 1.0, use_cache=True,
        renormalize_logits=normalization == "trie",
    )


def resolve_score_normalization(model_path: str | Path, requested: str = "auto") -> str:
    """New checkpoints carry their scoring rule; older GRPO decoding was full-vocab."""
    if requested != "auto":
        if requested not in {"trie", "full_vocab"}:
            raise ValueError("score_normalization must be auto, trie, or full_vocab")
        return requested
    model_path = Path(model_path)
    for path in (model_path / "grpo_config.json", model_path.parent / "grpo_run.json"):
        if path.is_file():
            metadata = json.loads(path.read_text(encoding="utf-8"))
            normalization = metadata.get("grpo", {}).get("score_normalization", "full_vocab")
            if normalization not in {"trie", "full_vocab"}:
                raise ValueError(f"Invalid score_normalization in checkpoint metadata: {normalization!r}")
            return normalization
    return "trie"


def listwise_ranking_loss(scores: torch.Tensor, rewards: torch.Tensor, group_size: int,
                          canonical: Sequence[tuple[int, ...]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Optional label-based ranking term; no invented partial credit for wrong IDs.

    Deduplicate candidates so repeated samples do not change the target mass.
    Only groups containing both a positive and a negative provide ranking
    supervision. Average over all queries, including connected zero terms.
    """
    losses, eligible = [], []
    for start in range(0, len(scores), group_size):
        unique = {}
        for position in range(start, start + group_size):
            unique.setdefault(canonical[position], position)
        keep = torch.tensor(list(unique.values()), device=scores.device)
        labels = rewards[keep].gt(0)
        active = labels.any() & (~labels).any()
        target = labels.float() / labels.sum().clamp_min(1)
        value = -(target * scores[keep].log_softmax(0)).sum()
        losses.append(value * active)
        eligible.append(active.float())
    return torch.stack(losses).mean(), torch.stack(eligible).mean()


class _UpdateMonitor(TrainerCallback):
    """Track an actual optimizer update without copying the full policy model."""

    def __init__(self, trainer: Seq2SeqGRPOTrainer):
        self.trainer = weakref.ref(trainer)
        self.parameter_name, self.parameter = next(
            (name, parameter) for name, parameter in trainer.model.named_parameters()
            if parameter.requires_grad and parameter.ndim >= 2 and "embed" not in name and name != "shared.weight"
        )
        self.indices = torch.linspace(0, self.parameter.numel() - 1,
                                      min(256, self.parameter.numel())).long()
        self.before = None

    def snapshot(self) -> torch.Tensor:
        return self.parameter.detach().view(-1)[self.indices.to(self.parameter.device)].float().cpu().clone()

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self.before = self.snapshot()

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        trainer = self.trainer()
        if trainer is not None and self.before is not None:
            difference = self.snapshot() - self.before
            values = trainer.accelerator.reduce(torch.tensor([
                difference.square().mean().sqrt().item(),
                float(trainer.accelerator.optimizer_step_was_skipped),
            ], device=trainer.args.device), reduction="mean").cpu().tolist()
            trainer._rollout_metrics["sampled_update_rms"].append(values[0])
            trainer._rollout_metrics["optimizer_step_skipped"].append(values[1])


@dataclass
class QueryCollator:
    tokenizer: Any
    max_prompt_length: int

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self.tokenizer(
            [row["prompt"] for row in rows], padding=True, truncation=True,
            max_length=self.max_prompt_length, return_tensors="pt",
        )
        return {
            "input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"],
            "positive_targets": [row["positive_targets"] for row in rows],
        }


def retrieval_metrics(prediction: Any) -> dict[str, float]:
    scores = prediction.predictions
    result = {"mrr@10": float(scores[:, 0].mean()), "recall@k": float(scores[:, 1].mean())}
    if scores.shape[1] >= 7:
        result.update({name: float(scores[:, i].mean()) for i, name in enumerate(
            ("hit@1", "hit@5", "hit@10", "mrr@1", "mrr@5"), 2)})
    if scores.shape[1] >= 10:
        result.update({name: float(scores[:, i].mean()) for i, name in enumerate(
            ("recall@10", "ndcg@10", "relevant_count@10"), 7)})
    return result


class Seq2SeqGRPOTrainer(Trainer):
    def __init__(self, *args: Any, target_index: TargetIndex, grpo: GRPOSettings,
                 reference_model: nn.Module | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if not self.model.config.is_encoder_decoder:
            raise ValueError("Seq2SeqGRPOTrainer requires an encoder-decoder model")
        if self.args.deepspeed or self.args.fsdp:
            raise ValueError("This trainer supports single-device and DDP training, not DeepSpeed/FSDP")
        self.target_index = target_index
        self.grpo = grpo
        # Validate deterministic group capacity before the first rollout.
        grpo.rollout_generation_config(target_index)
        if self.eval_dataset is not None and grpo.eval_num_beams < min(10, len(target_index.sequence_to_target)):
            raise ValueError("eval_num_beams must cover min(10, corpus size) for MRR@10")
        self.reference_model = reference_model
        self._rollout_metrics: dict[str, list[float]] = defaultdict(list)
        self.model_accepts_loss_kwargs = False
        # Dropout must be disabled for on-policy sampling/scoring consistency.
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
            # T5Attention uses functional dropout with a float attribute.
            for name in ("dropout", "attention_dropout", "activation_dropout"):
                if isinstance(getattr(module, name, None), float):
                    setattr(module, name, 0.0)
        if grpo.beta and reference_model is None:
            raise ValueError("beta > 0 requires a frozen reference model")
        if reference_model is not None:
            reference_model.requires_grad_(False)
            reference_model.eval()
            self.reference_model = self.accelerator.prepare_model(reference_model, evaluation_mode=True)
        self.compute_metrics = retrieval_metrics
        self.update_monitor = _UpdateMonitor(self)
        self.add_callback(self.update_monitor)

    def generation_config(self, training: bool) -> GenerationConfig:
        # A fresh config prevents inherited beam/forced-token/repetition settings
        # from silently changing the rollout distribution.
        if training:
            return self.grpo.rollout_generation_config(self.target_index)
        beams = min(self.grpo.eval_num_beams, len(self.target_index.sequence_to_target))
        return beam_generation_config(self.target_index, self.grpo.max_target_length, beams, beams,
                                      self.grpo.score_normalization)

    def generate_sequences(self, model: nn.Module, inputs: dict[str, Any], training: bool) -> torch.Tensor:
        unwrapped = self.accelerator.unwrap_model(model)
        was_training = unwrapped.training
        unwrapped.eval()
        try:
            # The dry run generates before Trainer.train()/Accelerate.prepare
            # installs the mixed-precision forward wrapper. Enter autocast here
            # explicitly so its beam caches match calibration/training precision.
            with torch.no_grad(), self.accelerator.autocast():
                return unwrapped.generate(
                    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                    generation_config=self.generation_config(training),
                    prefix_allowed_tokens_fn=self.target_index.allowed,
                    synced_gpus=False,
                )
        finally:
            unwrapped.train(was_training)

    def compute_loss(self, model: nn.Module, inputs: dict[str, Any], return_outputs: bool = False,
                     num_items_in_batch: Any = None) -> Any:
        sequences = self.generate_sequences(model, inputs, training=True)
        group_size = self.grpo.num_generations
        repeated = {
            name: inputs[name].repeat_interleave(group_size, dim=0)
            for name in ("input_ids", "attention_mask")
        }
        mask = completion_mask(sequences[:, 1:], self.target_index.eos_id, self.target_index.pad_id)
        layout = self.target_index.score_layout(sequences)
        rewards = self.target_index.rewards(sequences, inputs["positive_targets"], group_size)
        advantages = group_advantages(rewards, group_size)
        decoder_inputs = sequences[:, :-1]
        # The decoder start can itself be PAD (T5), so its attention stays on.
        decoder_mask = torch.cat((torch.ones_like(mask[:, :1]), mask[:, :-1]), dim=1)
        outputs = model(**repeated, decoder_input_ids=decoder_inputs,
                        decoder_attention_mask=decoder_mask, use_cache=False)
        logps = token_logps(outputs.logits, layout, self.grpo.scoring_temperature, self.grpo.score_normalization)
        # One fresh rollout per microbatch: generation and scoring share weights.
        # Beam selection is biased; the ratio does not correct that selection.
        # detach retains the old-policy denominator, not its gradient.
        old_logps = logps.detach()
        reference_logps = None
        if self.grpo.beta:
            with torch.no_grad():
                ref_outputs = self.reference_model(
                    **repeated, decoder_input_ids=decoder_inputs,
                    decoder_attention_mask=decoder_mask, use_cache=False,
                )
                reference_logps = token_logps(ref_outputs.logits, layout, self.grpo.scoring_temperature,
                                             self.grpo.score_normalization)
        objective, kl = grpo_loss(logps, old_logps, advantages, mask, self.grpo.epsilon,
                                 self.grpo.beta, reference_logps)
        canonical = [self.target_index.canonical(row) for row in sequences.detach().cpu().tolist()]
        sequence_scores = (logps * mask).sum(1)
        ranking_loss = objective.new_zeros(())
        ranking_groups = objective.new_zeros(())
        if self.grpo.ranking_loss_weight:
            ranking_loss, ranking_groups = listwise_ranking_loss(sequence_scores, rewards, group_size, canonical)
        loss = objective + self.grpo.ranking_loss_weight * ranking_loss
        # With one fresh scoring pass, ratio == 1 and mean(advantages) == 0.
        # The surrogate value therefore cancels despite nonzero derivatives.
        # Log a detached score-function diagnostic, whose gradient would match
        # the unclipped policy term here, without changing the optimized loss.
        with torch.no_grad():
            trajectory_logps = (logps * mask).sum(1) / mask.sum(1).clamp_min(1)
            pg_loss = -(advantages * trajectory_logps).mean()
        grouped = rewards.reshape(-1, group_size)
        means = grouped.mean(1)
        stats = {
            "total_loss": loss.detach(),
            "policy_loss": objective.detach() - self.grpo.beta * kl.detach(),
            "kl_loss": self.grpo.beta * kl.detach(),
            "ranking_loss": ranking_loss.detach(), "ranking_groups": ranking_groups,
            "weighted_ranking_loss": self.grpo.ranking_loss_weight * ranking_loss.detach(),
            "pg_loss": pg_loss, "advantage_abs_mean": advantages.abs().mean(),
            "reward": rewards.mean(), "mixed_reward_groups": ((means > 0) & (means < 1)).float().mean(),
            "all_zero_groups": means.eq(0).float().mean(), "all_one_groups": means.eq(1).float().mean(),
            "completion_length": mask.sum(1).float().mean(), "kl": kl.detach(),
            "positive_rollouts_per_query": grouped.sum(1).mean(),
            "candidate_hit_any": means.gt(0).float().mean(),
            "decision_token_fraction": ((layout[1].sum(-1) > 1) & mask).sum() / mask.sum().clamp_min(1),
            "truncated_fraction": (~sequences[:, 1:].eq(self.target_index.eos_id).any(1)).float().mean(),
        }
        grouped_cpu = grouped.detach().cpu().tolist()
        duplicates, positive_counts, reciprocal_ranks, top_hits = [], [], [], []
        sequence_scores_cpu = sequence_scores.detach().cpu().tolist()
        for i in range(len(means)):
            group = canonical[i * group_size:(i + 1) * group_size]
            duplicates.append(1 - len(set(group)) / group_size)
            positive_counts.append(len({seq for j, seq in enumerate(group) if grouped_cpu[i][j] == 1}))
            order = sorted(range(group_size), key=lambda j: sequence_scores_cpu[i * group_size + j], reverse=True)
            seen, ranked_rewards = set(), []
            for j in order:
                if group[j] not in seen:
                    seen.add(group[j])
                    ranked_rewards.append(grouped_cpu[i][j])
            first = next((j + 1 for j, reward in enumerate(ranked_rewards) if reward > 0), None)
            reciprocal_ranks.append(1 / first if first else 0.)
            top_hits.append(float(first == 1))
        stats["duplicate_fraction"] = loss.new_tensor(sum(duplicates) / len(duplicates))
        stats["distinct_positives"] = loss.new_tensor(sum(positive_counts) / len(positive_counts))
        stats["candidate_mrr"] = loss.new_tensor(sum(reciprocal_ranks) / len(reciprocal_ranks))
        stats["candidate_hit_at_1"] = loss.new_tensor(sum(top_hits) / len(top_hits))
        # All ranks participate even when an entire local group has zero reward.
        values = torch.stack([value.detach().float() for value in stats.values()])
        values = self.accelerator.reduce(values, reduction="mean").cpu().tolist()
        for name, value in zip(stats, values):
            self._rollout_metrics[name].append(value)
        return (loss, {"loss": loss}) if return_outputs else loss

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
        if "loss" in logs:
            for name, values in self._rollout_metrics.items():
                if values:
                    logs[f"grpo/{name}"] = sum(values) / len(values)
            # Trainer rounds its built-in loss to four decimal places before
            # calling log(). Preserve the raw microbatch mean so small KL
            # penalties appear in scientific notation instead of always 0.0.
            if "grpo/total_loss" in logs:
                logs["loss"] = logs["grpo/total_loss"]
            self._rollout_metrics.clear()
        super().log(logs, *args, **kwargs)

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
        super().save_model(output_dir, _internal_call=_internal_call)
        if self.is_world_process_zero():
            path = Path(output_dir or self.args.output_dir) / "grpo_config.json"
            path.write_text(json.dumps({"implementation_version": 2, "grpo": asdict(self.grpo)}, indent=2) + "\n",
                            encoding="utf-8")

    def prediction_step(self, model: nn.Module, inputs: dict[str, Any], prediction_loss_only: bool,
                        ignore_keys: Any = None) -> tuple[Any, Any, Any]:
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            sequences = self.generate_sequences(model, inputs, training=False)
        beams = self.generation_config(False).num_return_sequences
        scores = []
        canonical = [self.target_index.canonical(row) for row in sequences.cpu().tolist()]
        for i, positives in enumerate(inputs["positive_targets"]):
            ranked = []
            for sequence in canonical[i * beams:(i + 1) * beams]:
                target = self.target_index.sequence_to_target.get(sequence)
                if target is not None and target not in ranked:
                    ranked.append(target)
            hits = [rank + 1 for rank, target in enumerate(ranked) if target in positives]
            reciprocal = 1 / hits[0] if hits and hits[0] <= 10 else 0.0
            recall = len(set(ranked) & set(positives)) / len(set(positives))
            first = hits[0] if hits else None
            hit1 = float(first == 1)
            scores.append([reciprocal, recall, hit1, float(first is not None and first <= 5),
                           float(first is not None and first <= 10), hit1,
                           1 / first if first is not None and first <= 5 else 0.])
            coverage = score_ranking(ranked, positives)
            scores[-1].extend([coverage["recall10"], coverage["ndcg10"], coverage["relevant_count10"]])
        predictions = torch.tensor(scores, dtype=torch.float32, device=sequences.device)
        # Dummy labels allow the standard Trainer evaluation loop to aggregate
        # per-query metrics, including distributed sampler padding removal.
        return None, predictions, torch.zeros(len(scores), dtype=torch.long, device=sequences.device)
