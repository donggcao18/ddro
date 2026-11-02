#!/usr/bin/env python3
"""Train a T5/CodeT5 generative retriever with binary multi-positive GRPO."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, set_seed

from utils.grpo_data import filter_query_targets, load_corpus, load_queries, split_queries, validate_query_sets
from utils.grpo_trainer import GRPOSettings, QueryCollator, Seq2SeqGRPOTrainer, TargetIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_path", required=True, help="Local SFT save_pretrained checkpoint.")
    parser.add_argument("--train_file", required=True, help="JSON/JSONL with prompt and positive_targets.")
    parser.add_argument("--eval_file")
    parser.add_argument("--corpus_file", required=True, help='Full corpus JSONL: {"target": "decoder ID"}.')
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--validation_split", type=float, default=0.01)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, required=True, help="Decoder token limit, including EOS, excluding decoder start.")
    parser.add_argument("--invalid_target_policy", choices=("error", "skip"),
                        default=os.environ.get("GRPO_INVALID_TARGET_POLICY", "error"))
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.0, help="Optional frozen-SFT KL penalty; zero loads no reference model.")
    parser.add_argument("--eval_num_beams", type=int, default=10)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Queries per device; trajectories = queries * num_generations.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--run_name")
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument("--fp16", action="store_true")
    precision.add_argument("--bf16", action="store_true")
    parser.add_argument("--resume_from_checkpoint", nargs="?", const=True, default=None)
    parser.add_argument("--dry_run", action="store_true", help="Validate data/model and sample one query batch; do not optimize.")
    return parser.parse_args()


def build_trainer(args: argparse.Namespace) -> Seq2SeqGRPOTrainer:
    if args.max_prompt_length < 1 or args.learning_rate <= 0 or args.num_train_epochs <= 0:
        raise ValueError("Prompt length, learning rate, and training epochs must be positive")
    if min(args.logging_steps, args.save_steps, args.eval_steps) < 1:
        raise ValueError("Logging, save, and evaluation steps must be positive")
    settings = GRPOSettings(
        num_generations=args.num_generations, temperature=args.temperature,
        max_target_length=args.max_target_length, epsilon=args.epsilon,
        beta=args.beta, eval_num_beams=args.eval_num_beams,
    )
    set_seed(args.seed)
    checkpoint = Path(args.checkpoint_path)
    if not checkpoint.is_dir():
        raise ValueError(f"Local checkpoint directory does not exist: {checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, local_files_only=True)
    if not model.config.is_encoder_decoder:
        raise ValueError("An encoder-decoder SFT checkpoint is required")
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        raise ValueError("Tokenizer vocabulary exceeds the checkpoint embedding size")
    targets = load_corpus(args.corpus_file)
    index = TargetIndex(tokenizer, model.config, targets, settings.max_target_length,
                        invalid_target_policy=args.invalid_target_policy)
    train = load_queries(args.train_file)
    if args.eval_file:
        validation = load_queries(args.eval_file)
    else:
        train, validation = split_queries(train, args.validation_split, args.seed)
    validate_query_sets(train, validation, set(targets))
    allowed_targets = set(index.target_to_sequence)
    train, train_filtering = filter_query_targets(train, allowed_targets)
    validation, validation_filtering = filter_query_targets(validation, allowed_targets)
    query_filtering = {"train": train_filtering, "validation": validation_filtering}
    if int(os.environ.get("RANK", "0")) == 0:
        index.write_filter_report(args.output_dir)
        (Path(args.output_dir) / "query_filtering.json").write_text(
            json.dumps(query_filtering, indent=2) + "\n", encoding="utf-8"
        )
        print("Target filtering:", json.dumps(index.filtering_summary))
        print("Query filtering:", json.dumps(query_filtering))
    if not train:
        raise ValueError("No training queries retain a usable positive target after filtering; see target/query filtering reports")
    validate_query_sets(train, validation, allowed_targets)
    # Distinct text queries can become identical after tokenization/truncation.
    validation_encodings = {
        tuple(tokenizer(row["prompt"], truncation=True, max_length=args.max_prompt_length)["input_ids"])
        for row in validation
    }
    for row in train if validation else []:
        encoded = tuple(tokenizer(row["prompt"], truncation=True, max_length=args.max_prompt_length)["input_ids"])
        if encoded in validation_encodings:
            raise ValueError(f"Train/validation prompts become identical after tokenization: {row['prompt']!r}")
    has_eval = bool(validation)
    if has_eval and args.save_steps % args.eval_steps:
        raise ValueError("With validation, save_steps must be a multiple of eval_steps")
    training_kwargs = dict(
        output_dir=args.output_dir, run_name=args.run_name,
        seed=args.seed, data_seed=args.seed,
        num_train_epochs=args.num_train_epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm, lr_scheduler_type="cosine",
        logging_steps=args.logging_steps, logging_first_step=True,
        eval_steps=args.eval_steps if has_eval else None,
        save_steps=args.save_steps, save_total_limit=args.save_total_limit,
        load_best_model_at_end=has_eval, metric_for_best_model="mrr@10" if has_eval else None,
        greater_is_better=True if has_eval else None,
        fp16=args.fp16, bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False, label_names=[],
        ddp_find_unused_parameters=False,
        report_to=[] if args.report_to == "none" else [args.report_to],
    )
    # Preserve the repository's Transformers 4.45 API and accept newer releases.
    strategy_key = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments).parameters else "evaluation_strategy"
    training_kwargs[strategy_key] = "steps" if has_eval else "no"
    training_args = TrainingArguments(**training_kwargs)
    reference = None
    if settings.beta:
        reference = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, local_files_only=True)
    if args.gradient_checkpointing:
        model.config.use_cache = False
    trainer_kwargs = dict(
        model=model, args=training_args,
        train_dataset=Dataset.from_list(train),
        eval_dataset=Dataset.from_list(validation) if has_eval else None,
        data_collator=QueryCollator(tokenizer, args.max_prompt_length),
        target_index=index, grpo=settings, reference_model=reference,
    )
    trainer_parameters = inspect.signature(Seq2SeqGRPOTrainer.__mro__[1].__init__).parameters
    trainer_kwargs["processing_class" if "processing_class" in trainer_parameters else "tokenizer"] = tokenizer
    trainer = Seq2SeqGRPOTrainer(**trainer_kwargs)
    if trainer.is_world_process_zero():
        destination = Path(args.output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        manifest = {
            "arguments": vars(args), "grpo": asdict(settings),
            "train_queries": len(train), "validation_queries": len(validation),
            "corpus_targets": len(allowed_targets),
            "target_filtering": index.filtering_summary, "query_filtering": query_filtering,
            "recall_k": min(settings.eval_num_beams, len(allowed_targets)),
            "reward": "1 iff completed target is in the query positive set; otherwise 0",
        }
        (destination / "grpo_run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(manifest, indent=2))
    return trainer


def main() -> None:
    args = parse_args()
    trainer = build_trainer(args)
    if args.dry_run:
        inputs = trainer._prepare_inputs(next(iter(trainer.get_train_dataloader())))
        with trainer.compute_loss_context_manager():
            sequences = trainer.generate_sequences(trainer.model, inputs, sampling=True)
        rewards = trainer.target_index.rewards(sequences, inputs["positive_targets"], args.num_generations)
        if trainer.is_world_process_zero():
            print("Dry-run targets:", [trainer.target_index.sequence_to_target.get(trainer.target_index.canonical(row)) for row in sequences.cpu().tolist()])
            print("Dry-run rewards:", rewards.cpu().tolist())
            print("Dry run successful; no optimizer steps were taken.")
        return
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if trainer.eval_dataset is not None:
        trainer.log_metrics("eval", trainer.evaluate())
    trainer.model.config.use_cache = True
    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    trainer.save_state()
    if trainer.is_world_process_zero():
        print(f"Saved final Hugging Face GRPO model to {final_dir}")


if __name__ == "__main__":
    main()
