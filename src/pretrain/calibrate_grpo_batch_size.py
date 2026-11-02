#!/usr/bin/env python3
"""Probe GRPO memory on one GPU before starting DDP; never train the checkpoint.

The least-free visible training GPU is used. Probes include generation to the
longest corpus target, full-length prompts, backward, and two AdamW steps so
optimizer state is resident during the second forward. Extra VRAM is reserved
for DDP buckets and runtime variation. Results are a starting batch estimate,
not a guarantee against changing GPU occupancy or backend-specific allocations.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from accelerate.utils import convert_to_fp32
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from utils.grpo_data import load_corpus
from utils.grpo_trainer import GRPOSettings, TargetIndex, completion_mask, constrained_logps, grpo_loss


def largest_fitting_batch(start: int, maximum: int, probe: Callable[[int], bool]) -> int:
    """Bracket the capacity, then search every integer size within the bracket."""
    if not 1 <= start <= maximum:
        raise ValueError("Require 1 <= starting batch <= maximum batch")
    candidate = start
    if probe(candidate):
        lower = candidate
        while candidate < maximum:
            candidate = min(maximum, candidate * 2)
            if not probe(candidate):
                upper = candidate - 1
                break
            lower = candidate
        else:
            return lower
    else:
        upper = candidate - 1
        lower = 0
        while candidate > 1:
            candidate = max(1, candidate // 2)
            if probe(candidate):
                lower = candidate
                break
            upper = candidate - 1
        if lower == 0:
            raise RuntimeError("Even one query does not fit; reduce NUM_GENERATIONS or use a GPU with more free VRAM")
    while lower < upper:
        middle = (lower + upper + 1) // 2
        if probe(middle):
            lower = middle
        else:
            upper = middle - 1
    return lower


def probe_once(args: argparse.Namespace, index: TargetIndex, device: torch.device,
               batch_size: int) -> tuple[bool, int]:
    """Use fresh model/optimizer objects so failed probes cannot retain state."""
    model = reference = optimizer = reserve = None
    try:
        torch.cuda.reset_peak_memory_stats(device)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint_path, local_files_only=True).to(device)
        if not model.config.is_encoder_decoder:
            raise ValueError("Calibration requires an encoder-decoder checkpoint")
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
            for name in ("dropout", "attention_dropout", "activation_dropout"):
                if isinstance(getattr(module, name, None), float):
                    setattr(module, name, 0.0)
        if args.beta:
            reference = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint_path, local_files_only=True).to(device)
            reference.requires_grad_(False)
            reference.eval()
        # Reserve at least two parameter-sized buffers for multi-GPU overhead.
        parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
        reserve_bytes = int(args.reserve_gib * 1024 ** 3)
        if args.num_gpus > 1:
            reserve_bytes = max(reserve_bytes, 2 * parameter_bytes)
        reserve = torch.empty(reserve_bytes, dtype=torch.uint8, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
        scaler = torch.cuda.amp.GradScaler(enabled=args.fp16, init_scale=128)
        longest = max(index.sequence_to_target, key=len)
        content_id = next((token for token in longest if token not in (index.pad_id, index.start_id, index.eos_id)), None)
        if content_id is None:
            raise ValueError("No content token is available for a full-length probe prompt")
        inputs = torch.full((batch_size, args.max_prompt_length), content_id, dtype=torch.long, device=device)
        inputs[:, -1] = index.eos_id
        attention = torch.ones_like(inputs)
        settings = GRPOSettings(
            num_generations=args.num_generations, do_sample=args.do_sample == "true",
            temperature=args.temperature, max_target_length=len(longest), beta=args.beta,
        )
        generation = settings.rollout_generation_config(index)

        def force_longest(batch_id: int, prefix: torch.Tensor) -> list[int]:
            position = prefix.shape[0] - 1
            return [longest[position]] if position < len(longest) else [index.pad_id]

        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=args.fp16):
                sequences = model.generate(
                    input_ids=inputs, attention_mask=attention, generation_config=generation,
                    prefix_allowed_tokens_fn=force_longest,
                )
            if not settings.do_sample:
                # A forced single path can leave -inf placeholder beam outputs.
                # Generation above exercises the configured beam cache; use the
                # full longest valid path for every backward row to probe memory.
                sequences = torch.tensor([index.start_id, *longest], device=device).repeat(
                    batch_size * args.num_generations, 1)
            mask = completion_mask(sequences[:, 1:], index.eos_id, index.pad_id)
            layout = index.score_layout(sequences)
            repeated = {
                "input_ids": inputs.repeat_interleave(args.num_generations, 0),
                "attention_mask": attention.repeat_interleave(args.num_generations, 0),
                "decoder_input_ids": sequences[:, :-1],
                "decoder_attention_mask": torch.cat((torch.ones_like(mask[:, :1]), mask[:, :-1]), 1),
                "use_cache": False,
            }
            model.train()
            with torch.autocast("cuda", dtype=torch.float16, enabled=args.fp16):
                # Accelerator's native-AMP wrapper converts ModelOutput tensors
                # back to FP32. Include that allocation in the capacity estimate.
                outputs = convert_to_fp32(model(**repeated))
                logps = constrained_logps(outputs.logits, layout, settings.scoring_temperature)
                ref_logps = None
                if reference is not None:
                    with torch.no_grad():
                        ref_outputs = convert_to_fp32(reference(**repeated))
                        ref_logps = constrained_logps(ref_outputs.logits, layout, settings.scoring_temperature)
                # Artificial nonzero advantages exercise backward allocations;
                # this is a capacity probe, not a source of training supervision.
                loss, kl = grpo_loss(logps, logps.detach(), torch.ones(len(sequences), device=device),
                                     mask, .2, args.beta, ref_logps)
            # Trainer.compute_loss returns before backward/optimizer.step; do
            # not keep otherwise-unused full-vocabulary outputs alive longer.
            del outputs, kl
            if reference is not None:
                del ref_outputs
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
            scaler.step(optimizer)
            scaler.update()
            if not optimizer.state:
                raise RuntimeError("Probe could not allocate optimizer state; check FP16 numerical stability")
            torch.cuda.synchronize(device)
            # Avoid retaining the preceding graph/logits into the next forward.
            del sequences, mask, layout, repeated, logps, ref_logps, loss
        return True, torch.cuda.max_memory_allocated(device)
    except torch.cuda.OutOfMemoryError:
        return False, torch.cuda.max_memory_allocated(device)
    finally:
        del model, reference, optimizer, reserve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--corpus_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--do_sample", type=str.lower, choices=("true", "false"), default="true")
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--invalid_target_policy", choices=("error", "skip"),
                        default=os.environ.get("GRPO_INVALID_TARGET_POLICY", "error"))
    parser.add_argument("--start_batch_size", type=int, default=16)
    parser.add_argument("--max_batch_size", type=int, default=32)
    parser.add_argument("--reserve_gib", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.num_gpus:
        raise ValueError(f"Calibration needs {args.num_gpus} visible CUDA GPUs")
    if min(args.num_gpus, args.max_prompt_length, args.max_target_length) < 1 or args.num_generations < 2:
        raise ValueError("Require positive GPU/token limits and at least two generations")
    if not 1 <= args.start_batch_size <= args.max_batch_size or args.reserve_gib < 0:
        raise ValueError("Require 1 <= start_batch_size <= max_batch_size and reserve_gib >= 0")
    if args.temperature <= 0 or args.beta < 0:
        raise ValueError("Require temperature > 0 and beta >= 0")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, local_files_only=True)
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.checkpoint_path, local_files_only=True)
    index = TargetIndex(tokenizer, config, load_corpus(args.corpus_file), args.max_target_length,
                        invalid_target_policy=args.invalid_target_policy)
    index.write_filter_report(Path(args.output_file).parent / "calibration_target_filtering")
    print("Target filtering:", json.dumps(index.filtering_summary), flush=True)
    free_memory = [torch.cuda.mem_get_info(i)[0] for i in range(args.num_gpus)]
    ordinal = min(range(args.num_gpus), key=lambda i: free_memory[i])
    device = torch.device("cuda", ordinal)
    torch.cuda.set_device(device)
    print(f"Calibrating on cuda:{ordinal}, the least-free training GPU; reserved margin >= {args.reserve_gib:g} GiB", flush=True)
    trials = []

    def probe(batch_size: int) -> bool:
        gc.collect()
        torch.cuda.empty_cache()
        fits, peak = probe_once(args, index, device, batch_size)
        gc.collect()
        torch.cuda.empty_cache()
        trials.append({"queries_per_gpu": batch_size, "trajectories_per_gpu": batch_size * args.num_generations,
                       "fits": fits, "peak_allocated_gib": peak / 1024 ** 3})
        print(f"  {batch_size} queries / {batch_size * args.num_generations} trajectories: "
              f"{'fits' if fits else 'OOM'}; peak incl. reserve {peak / 1024 ** 3:.2f} GiB", flush=True)
        return fits

    chosen = largest_fitting_batch(args.start_batch_size, args.max_batch_size, probe)
    result = {
        "batch_size": chosen, "num_generations": args.num_generations, "do_sample": args.do_sample == "true",
        "gpu": ordinal, "gpu_name": torch.cuda.get_device_name(device),
        "reserve_gib": args.reserve_gib, "max_batch_size": args.max_batch_size,
        "prompt_tokens": args.max_prompt_length,
        "target_tokens": max(map(len, index.sequence_to_target)), "trials": trials,
        "target_filtering": index.filtering_summary,
    }
    path = Path(args.output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Selected TRAIN_BATCH_SIZE={chosen}; report: {path}", flush=True)


if __name__ == "__main__":
    main()
