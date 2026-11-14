"""Regression coverage for the encoder-decoder DPO tokenization cache."""

from functools import wraps
from importlib.util import find_spec
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class UnhashableModel:
    def __reduce__(self):
        raise TypeError("Model must not be hashed or sent to preprocessing workers")


def _tokenize(batch, tokenizer, args, processor, model):
    assert model is True
    return {
        "prompt_ids": [tokenizer(text, args.max_prompt_length) for text in batch["prompt"]],
        "chosen_ids": [tokenizer(text, args.max_completion_length) for text in batch["chosen"]],
    }


def token_ids(text, length):
    return list(map(ord, text[:length]))


@unittest.skipUnless(find_spec("datasets"), "requires Hugging Face Datasets")
class PreferenceCacheTest(unittest.TestCase):
    def test_reuse_across_ranks_preserves_indexed_order_and_token_limits(self):
        from datasets import Dataset, load_from_disk
        from pretrain.preference_cache import TokenizationCacheDataset

        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            Dataset.from_dict({
                "prompt": [f"query {i}" for i in range(70)],
                "chosen": [f"url/{i}" for i in range(70)],
            }).save_to_disk(temporary)
            source = load_from_disk(temporary).select(list(range(69, -1, -2)))
            old_layout = TokenizationCacheDataset.from_dataset(source, 10)
            new_layout = TokenizationCacheDataset.from_dataset(source, 1000)

            def options(rank, completion_length=5):
                return {
                    "batched": True,
                    "writer_batch_size": 10,
                    "fn_kwargs": {
                        "model": UnhashableModel(),
                        "processor": None,
                        "tokenizer": token_ids,
                        "args": SimpleNamespace(
                            max_prompt_length=7,
                            max_completion_length=completion_length,
                            local_rank=rank,
                        ),
                    },
                }

            with patch.object(_tokenize, "__module__", "trl.trainer.dpo_trainer"):
                small_batches = old_layout.map(_tokenize, **options(0))
                with patch.dict(os.environ, {"RANK": "0"}):
                    first = new_layout.map(_tokenize, **options(0))

                @wraps(Dataset._map_single)
                def unexpected_map(*args, **kwargs):
                    raise AssertionError("cache miss")

                with patch.dict(os.environ, {"RANK": "1"}):
                    with patch.object(Dataset, "_map_single", staticmethod(unexpected_map)):
                        reopened = new_layout.map(_tokenize, **options(1))
                different_length = new_layout.map(_tokenize, **options(1, completion_length=3))

            self.assertEqual(first.cache_files, reopened.cache_files)
            self.assertNotEqual(first.cache_files, different_length.cache_files)
            self.assertEqual(first.to_dict(), reopened.to_dict())
            self.assertEqual(first.to_dict(), small_batches.to_dict())
            self.assertEqual(first["prompt"], source["prompt"])
            self.assertTrue(all(len(ids) == 3 for ids in different_length["chosen_ids"]))
            self.assertLess(len(first.data.table.to_batches()), len(small_batches.data.table.to_batches()))
