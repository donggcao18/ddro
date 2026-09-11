"""CPU regression tests for TRL tokenization cache layout (no model downloads)."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from datasets import Dataset, load_from_disk
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast
from trl.trainer.dpo_trainer import _tokenize

from preference_cache import TokenizationCacheDataset


class PreferenceCacheTest(unittest.TestCase):
    def test_disk_cache_preserves_tokenization_and_indexed_order(self):
        impl = Tokenizer(models.WordLevel(
            {"<pad>": 0, "</s>": 1, "<unk>": 2, "query": 3, "chosen": 4, "rejected": 5},
            unk_token="<unk>",
        ))
        impl.pre_tokenizer = pre_tokenizers.Whitespace()
        impl.post_processor = processors.TemplateProcessing(single="$A </s>", special_tokens=[("</s>", 1)])
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=impl, pad_token="<pad>", eos_token="</s>", unk_token="<unk>"
        )
        config = SimpleNamespace(max_prompt_length=4, max_completion_length=3)
        kwargs = {
            "batched": True, "writer_batch_size": 10,
            "fn_kwargs": {"tokenizer": tokenizer, "args": config, "model": True},
        }
        with tempfile.TemporaryDirectory() as directory:
            Dataset.from_dict({
                "prompt": ["query " * (i % 5 + 1) for i in range(65)],
                "chosen": ["chosen " * (i % 3 + 1) for i in range(65)],
                "rejected": ["rejected"] * 65,
                "row_index": list(range(65)),
            }).save_to_disk(directory)
            base = load_from_disk(directory)
            for num_proc in [1, 2]:
                for source in [base, base.select(list(range(64, -1, -2)))]:
                    with self.subTest(num_proc=num_proc, indexed=source._indices is not None):
                        before = source.to_dict()
                        fingerprint = source._fingerprint
                        original_format = source.format
                        original = source.map(_tokenize, num_proc=num_proc, **kwargs)
                        wrapped = TokenizationCacheDataset.from_dataset(source, 1000)
                        actual = wrapped.map(_tokenize, num_proc=num_proc, **kwargs)
                        self.assertEqual(actual.to_dict(), original.to_dict())
                        self.assertEqual(actual["row_index"], before["row_index"])
                        self.assertLess(
                            len(actual.data.table.to_batches()), len(original.data.table.to_batches())
                        )
                        self.assertNotEqual(actual.cache_files, original.cache_files)
                        self.assertEqual(source.to_dict(), before)
                        self.assertEqual(source._fingerprint, fingerprint)
                        self.assertEqual(source.format, original_format)
                        for cache in actual.cache_files:
                            self.assertTrue(Path(cache["filename"]).is_file())

    def test_layout_fingerprint_is_stable_and_size_specific(self):
        source = Dataset.from_dict({"prompt": ["a", "b"]}).select([1, 0])
        first = TokenizationCacheDataset.from_dataset(source, 1000)
        second = TokenizationCacheDataset.from_dataset(source, 1000)
        legacy_size = TokenizationCacheDataset.from_dataset(source, 10)
        self.assertEqual(first._fingerprint, second._fingerprint)
        self.assertNotEqual(first._fingerprint, legacy_size._fingerprint)
        self.assertNotEqual(first._fingerprint, source._fingerprint)

    def test_unrelated_map_keeps_its_writer_options(self):
        wrapped = TokenizationCacheDataset.from_dataset(Dataset.from_dict({"x": [1]}), 1000)
        function = lambda row: row
        with patch.object(Dataset, "map", return_value="result") as mapped:
            self.assertEqual(wrapped.map(function, writer_batch_size=10), "result")
        mapped.assert_called_once_with(function, writer_batch_size=10)

    def test_invalid_writer_size(self):
        source = Dataset.from_dict({"x": [1]})
        for size in [0, -1]:
            with self.assertRaises(ValueError):
                TokenizationCacheDataset.from_dataset(source, size)


if __name__ == "__main__":
    unittest.main()
