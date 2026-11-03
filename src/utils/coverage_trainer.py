"""Multi-positive preference post-training with freshly mined beam negatives.

Annotated positives are teacher-forced separately from decoding. They are not
presented as on-policy GRPO samples. All objectives average within each query
before averaging queries, and exclude every annotated positive from negatives.
"""

from dataclasses import dataclass
from math import ceil, isfinite
import random
from typing import Any, Sequence

import torch
from torch.nn import functional as F
from transformers import Trainer

from utils.grpo_trainer import GRPOSettings, Seq2SeqGRPOTrainer, completion_mask, token_logps


OBJECTIVES = ("multi_positive_dpo", "missed_positive_dpo", "multi_negative_dpo")


@dataclass
class CoverageSettings(GRPOSettings):
    objective: str = "multi_positive_dpo"
    preference_beta: float = .1
    sft_anchor_weight: float = .1
    positives_per_query: int = 2
    negatives_per_query: int = 4
    missed_positive_weight: float = 2.

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.objective not in OBJECTIVES:
            raise ValueError(f"Unknown preference objective: {self.objective}")
        if self.do_sample or self.beta or self.ranking_loss_weight:
            raise ValueError("Coverage experiments require deterministic negative mining, beta=0 and ranking_loss_weight=0")
        if self.positives_per_query < 1 or self.negatives_per_query < 1:
            raise ValueError("Positive and negative budgets must be positive")
        if not all(isfinite(x) for x in (self.preference_beta, self.sft_anchor_weight, self.missed_positive_weight)):
            raise ValueError("Preference settings must be finite")
        if self.preference_beta <= 0 or self.sft_anchor_weight < 0 or self.missed_positive_weight < 1:
            raise ValueError("Require preference_beta > 0, sft_anchor_weight >= 0, missed_positive_weight >= 1")


def select_candidates(positives: Sequence[str], ranked: Sequence[str], catalog: Sequence[str],
                      settings: CoverageSettings, rng: random.Random) -> dict[str, Any]:
    """Bounded, query-balanced positive sampling and false-negative-safe mining."""
    positive_set = set(positives)
    pool = sorted(positive_set)
    if not pool:
        raise ValueError("Each query must have a usable positive")
    ranked = list(dict.fromkeys(ranked))
    retrieved = set(ranked[:10])
    rng.shuffle(pool)
    budget = min(settings.positives_per_query, len(pool))
    chosen = []
    if settings.objective != "multi_positive_dpo":
        missed = [target for target in pool if target not in retrieved]
        chosen.extend(missed[:ceil(budget / 2)])
    chosen.extend(target for target in pool if target not in chosen)
    chosen = chosen[:budget]
    negatives = [target for target in ranked if target not in positive_set][:settings.negatives_per_query]
    mined = len(negatives)
    if len(negatives) < settings.negatives_per_query:
        # Avoid constructing a corpus-sized set/list for every query.
        positions = rng.sample(range(len(catalog)), min(len(catalog), settings.negatives_per_query * 4 + 16))
        for position in positions:
            target = catalog[position]
            if target not in positive_set and target not in negatives:
                negatives.append(target)
            if len(negatives) == settings.negatives_per_query:
                break
        if len(negatives) < settings.negatives_per_query:
            for target in catalog:
                if target not in positive_set and target not in negatives:
                    negatives.append(target)
                if len(negatives) == settings.negatives_per_query:
                    break
    weights = [settings.missed_positive_weight if settings.objective != "multi_positive_dpo"
               and target not in retrieved else 1. for target in chosen]
    return {"positives": chosen, "negatives": negatives, "weights": weights,
            "random_negatives": len(negatives) - mined,
            "missed_selected": sum(target not in retrieved for target in chosen),
            "retrieved_positives": len(retrieved & positive_set)}


def preference_loss(scores: torch.Tensor, reference_scores: torch.Tensor, positive_count: int,
                    weights: torch.Tensor, objective: str, beta: float) -> torch.Tensor:
    """DPO over all selected pairs, or per-positive multi-negative log contrast.

    Each positive has its own denominator: other positives never become its
    competitors. This avoids a loss that can be satisfied by only one easy
    positive carrying the whole positive-set probability mass.
    """
    if objective not in OBJECTIVES or not 0 < positive_count <= len(scores):
        raise ValueError("Invalid objective or positive count")
    if positive_count == len(scores):
        return scores.sum() * 0.
    rewards = beta * (scores - reference_scores.detach())
    differences = rewards[positive_count:][None, :] - rewards[:positive_count, None]
    if objective == "multi_negative_dpo":
        values = torch.logsumexp(torch.cat((torch.zeros_like(differences[:, :1]), differences), 1), 1)
    else:
        values = F.softplus(differences).mean(1)
    return (values * weights).sum() / weights.sum()


def score_sequences(model: Any, repeated: dict[str, torch.Tensor], sequences: torch.Tensor,
                    index: Any, normalization: str, positive_rows: torch.Tensor,
                    anchor: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Summed inference-compatible scores and full-vocabulary positive SFT NLL."""
    mask = completion_mask(sequences[:, 1:], index.eos_id, index.pad_id)
    decoder_mask = torch.cat((torch.ones_like(mask[:, :1]), mask[:, :-1]), 1)
    outputs = model(**repeated, decoder_input_ids=sequences[:, :-1], decoder_attention_mask=decoder_mask,
                    use_cache=False)
    logps = token_logps(outputs.logits, index.score_layout(sequences), 1., normalization)
    scores = (logps * mask).sum(1)
    nll = scores * 0.
    if anchor:
        labels = sequences[:, 1:].masked_fill(~(mask & positive_rows[:, None]), -100)
        terms = F.cross_entropy(outputs.logits.float().reshape(-1, outputs.logits.shape[-1]),
                                labels.reshape(-1), reduction="none").reshape_as(labels)
        nll = terms.sum(1) / mask.sum(1).clamp_min(1)
    return scores, nll


class CoveragePreferenceTrainer(Seq2SeqGRPOTrainer):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if self.reference_model is None:
            raise ValueError("Preference objectives require a frozen SFT reference")
        self.catalog = tuple(self.target_index.target_to_sequence)

    def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False,
                     num_items_in_batch: Any = None) -> Any:
        settings = self.grpo
        generated = self.generate_sequences(model, inputs, training=True)
        decoded = [self.target_index.sequence_to_target.get(self.target_index.canonical(row))
                   for row in generated.detach().cpu().tolist()]
        rng = random.Random(torch.randint(0, 2**31, ()).item())
        groups, owners, target_tokens, positive_rows = [], [], [], []
        for i, positives in enumerate(inputs["positive_targets"]):
            ranked = [target for target in decoded[i * settings.num_generations:(i + 1) * settings.num_generations]
                      if target is not None]
            group = select_candidates(positives, ranked, self.catalog, settings, rng)
            groups.append(group)
            targets = group["positives"] + group["negatives"]
            owners.extend([i] * len(targets))
            positive_rows.extend([True] * len(group["positives"]) + [False] * len(group["negatives"]))
            target_tokens.extend([[self.target_index.start_id, *self.target_index.target_to_sequence[t]] for t in targets])
        device = inputs["input_ids"].device
        sequences = torch.full((len(target_tokens), max(map(len, target_tokens))),
                               self.target_index.pad_id, dtype=torch.long, device=device)
        for i, tokens in enumerate(target_tokens):
            sequences[i, :len(tokens)] = torch.tensor(tokens, device=device)
        owner_index = torch.tensor(owners, device=device)
        repeated = {name: inputs[name].index_select(0, owner_index) for name in ("input_ids", "attention_mask")}
        positive_rows = torch.tensor(positive_rows, device=device)
        # Free reference logits before building the policy's backward graph.
        with torch.no_grad():
            reference_scores, _ = score_sequences(self.reference_model, repeated, sequences, self.target_index,
                                                  settings.score_normalization, positive_rows, False)
        scores, nll = score_sequences(model, repeated, sequences, self.target_index,
                                     settings.score_normalization, positive_rows, bool(settings.sft_anchor_weight))
        preferences, anchors = [], []
        offset = 0
        for group in groups:
            positives = len(group["positives"])
            end = offset + positives + len(group["negatives"])
            preferences.append(preference_loss(scores[offset:end], reference_scores[offset:end], positives,
                                               scores.new_tensor(group["weights"]), settings.objective,
                                               settings.preference_beta))
            anchors.append(nll[offset:offset + positives].mean())
            offset = end
        preference = torch.stack(preferences).mean()
        anchor = torch.stack(anchors).mean()
        loss = preference + settings.sft_anchor_weight * anchor
        stats = {"total_loss": loss.detach(), "preference_loss": preference.detach(), "sft_nll": anchor.detach(),
                 "selected_positives": loss.new_tensor(sum(len(g["positives"]) for g in groups) / len(groups)),
                 "selected_negatives": loss.new_tensor(sum(len(g["negatives"]) for g in groups) / len(groups)),
                 "missed_positives_selected": loss.new_tensor(sum(g["missed_selected"] for g in groups) / len(groups)),
                 "random_negatives": loss.new_tensor(sum(g["random_negatives"] for g in groups) / len(groups)),
                 "mining_hit_at_10": loss.new_tensor(sum(g["retrieved_positives"] > 0 for g in groups) / len(groups)),
                 "preference_queries": loss.new_tensor(sum(bool(g["negatives"]) for g in groups) / len(groups))}
        values = self.accelerator.reduce(torch.stack(list(stats.values())).float(), reduction="mean").cpu().tolist()
        for name, value in zip(stats, values):
            self._rollout_metrics[name].append(value)
        return (loss, {"loss": loss}) if return_outputs else loss

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
        if "loss" in logs:
            for name, values in self._rollout_metrics.items():
                if values:
                    logs[f"posttrain/{name}"] = sum(values) / len(values)
            logs["loss"] = logs.get("posttrain/total_loss", logs["loss"])
            self._rollout_metrics.clear()
        Trainer.log(self, logs, *args, **kwargs)
