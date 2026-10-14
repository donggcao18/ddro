# Branch-preference continuation after DPO

The `branch-preference` branch implements whole-ID sigmoid DPO plus a local
preference at the first divergent target token. Both actions are scored under
the same chosen-prefix history. Existing constrained on-policy mining,
multi-GPU generation, query-family splits, and retrieval evaluation are reused.

For `chosen=[A,B,C,D,EOS]` and `rejected=[A,B,X,Y,EOS]`, the local preference is
`C > X` under `[A,B]`. If X also leads to another known positive, the original
sequence pair remains in training but its branch loss is zero. Duplicate leaf
pairs representing the same decision share its auxiliary weight.

The objective is `L_sequence_DPO + branch_loss_weight * L_branch_DPO`.
The reference is initialized from the supplied completed DPO checkpoint and
kept frozen through the existing EMA-decay-1 mechanism. This implementation
uses uniform decision preferences with query/decision normalization; adaptive
weights and prefix SFT are separate follow-up experiments.

## Run the first pilot

Run in the repository root with the existing training environment (TRL 0.11.4,
Transformers 4.45.2). Supply the path to your **completed DPO checkpoint**:

```bash
conda activate ddro_env

CUDA_VISIBLE_DEVICES=0,1,2,3 bash src/scripts/ddro/run_vault_branch_preference.sh \
  --checkpoint /path/to/completed-dpo-checkpoint \
  --query-file /path/to/Ruby_merged_multilabel.jsonl \
  --target-type url_based_id \
  --output-dir outputs/vault-url-branch-C \
  --arm branch
```

For structure IDs, use the matching checkpoint/data and
`--target-type structure_id_v6`. The query file requires `positive_url_based_id`
or `positive_structure_id_v6`, respectively, as a nonempty list on every query
row. The ordinary ID column is also required. Missing positives do not fall
back to the provenance label.

If the query file does not cover the full intended retrieval collection, add
`--corpus-file /path/to/full-corpus.jsonl`; each corpus row must include the
selected ID column. Additional corpus files provide identities, not query
relevance labels.

Defaults: 4 GPUs, FP16, one round, up to 10,000 selected training queries,
1,000 optimizer updates, 30 beams, 4 negatives/query, mining batch 8/GPU,
training batch 4/GPU with accumulation 4, LR 1e-6, beta 0.4, branch coefficient
0.1, prompt limit 256, target limit 128. These are pilot settings, not verified
V100 memory/throughput measurements. Lower `--mining-batch-size` or `--batch-size`
and adjust accumulation if your checkpoint needs more memory.

The launcher writes a resolved config beside the output directory, e.g.
`outputs/vault-url-branch-C.config.json`. It launches the controller once;
the controller starts the mining and training workers. Do not wrap the launcher
in `torchrun`.

The launcher uses the on-policy compatibility defaults for every arm:
`--target-length-policy truncate --target-collision-policy allow
--target-token-policy allow`. Mining, branch extraction, and training use the
same effective token sequences, including tokenizer truncation and terminal EOS.
The target limit can remain 128 when some original IDs are longer. Alias groups
use the existing miner's representative IDs and positive filtering. Pairs whose
effective sequences are identical receive zero auxiliary loss. The branch audit
records overlength IDs, alias groups, excluded targets, and inactive pairs.

To require complete, distinct encodings, select `--target-length-policy error
--target-collision-policy error --target-token-policy error` and set an adequate
`--max-target-length`. The `skip` length/collision policies exclude affected IDs
from mining and branch-positive support; original relevance labels are preserved.

## Controls and refreshing negatives

Repeat the same command with identical settings and separate output directories:

```text
--arm dpo    --output-dir outputs/vault-url-branch-A
--arm zero   --output-dir outputs/vault-url-branch-C0
--arm branch --output-dir outputs/vault-url-branch-C
```

`dpo` continues ordinary DPO. `zero` builds branch annotations but sets the
auxiliary coefficient to zero. `branch` adds the local preference loss.
All arms use the same initial snapshot, selection seed, mining settings, and
frozen reference. Original sequence row order/content is preserved during
branch extraction.

For iterative on-policy mining, select `--rounds 3` when starting a **new**
experiment. Each round mines from the latest policy and re-extracts its branch
preferences. The existing controller restarts optimizer/scheduler state
between rounds; use the same lifecycle in comparison arms.

To resume an interruption, repeat the same command with `--resume`. You can
also run the saved config directly:

```bash
python src/pretrain/train_iterative_ddro_vault.py \
  --config outputs/vault-url-branch-C.config.json --resume
```

Use a new output directory when changing objective, coefficient, checkpoint,
data, target policies, or training settings. After this compatibility update,
restart a previously failed strict-policy run with a new output directory.
Existing pre-update manifests do not silently
migrate into branch experiments. `--prepare-only` prepares the split; continue
that run with `--resume` after removing `--prepare-only`.

## Outputs

Each round preserves `preferences.jsonl` and adds:

- `branch_preferences.jsonl`: all original sequence rows plus branch annotations.
- `branch_audit.json`: eligible rows/queries, skip reasons, distinct decisions,
  divergence positions, normalization weights, and source fingerprints.
- `training/latest/`: exported policy, tokenizer, and saved training arguments.

Training logs separate sequence/branch losses, eligible fraction, action margin,
and action accuracy. `best_checkpoint.json` selects by held-out MRR@10;
`run_manifest.json` records checkpoint identities and round retrieval metrics.
True multi-positive Recall is also reported by the existing evaluator.
If no pairs are branch-eligible, the auxiliary is zero and the run reports this
coverage explicitly. Action accuracy alone does not establish retrieval gains.

## Verification

The tests use locally created tiny T5 weights, with no model download:

```bash
python -m unittest discover -s src/pretrain/tests -p 'test_branch_preference*.py' -v
RUN_DPO_INTEGRATION=1 python -m unittest discover \
  -s src/pretrain/tests -p 'test_branch_preference*.py' -v
```

The integration flag enables real mining, annotation, training, frozen-reference
checks, interrupted checkpoint resume, and stale-artifact rejection under both
strict policies and truncated aliases. Tests also
verify zero-coefficient loss/gradient parity and same-state local gradients.
