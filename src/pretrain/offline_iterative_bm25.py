#!/usr/bin/env python3
"""Export training-only BM25 queries, retrieve in the BM25 env, and freeze a cache."""

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "scripts/bm25/the_vault"))
from common import clean_code, iter_json_records
from data.data_prep.prepare_vault_dpo_metadata import prepare
from pretrain.hybrid_preferences import query_signature
from pretrain.iterative_dpo_utils import assert_file_hashes, atomic_json, atomic_jsonl, file_hash, read_json


def export(cfg, work, code_files=()):
    work = Path(work).resolve()
    if work.exists() and any(work.iterdir()):
        raise ValueError("Offline preparation needs an empty work directory")
    if cfg["input_format"] != "multilabel":
        raise ValueError("Offline iterative BM25 currently requires direct multilabel IDs")
    metadata = work / "metadata"
    prepare(cfg["corpus_files"], cfg["query_file"], str(metadata), cfg["validation_fraction"],
            cfg["seed"], cfg["structure_id_sources"], cfg["id_mode"], cfg["input_format"], cfg["target_type"])
    documents = list(iter_json_records(metadata / "document_metadata.jsonl"))
    known = {row["text_id"] for row in documents}
    contents = {}
    sources = sorted({str(Path(p).resolve()) for p in [cfg["query_file"], *cfg["corpus_files"], *code_files]})
    for source in sources:
        for row in iter_json_records(source):
            doc = row.get(cfg["target_type"])
            original = row.get("original") or {}
            code = row.get("code") or (original.get("code") if isinstance(original, dict) else None)
            if doc not in known or not isinstance(code, str) or not clean_code(code):
                continue
            code = clean_code(code)
            if doc in contents and contents[doc] != code:
                raise ValueError(f"Conflicting code bodies for document: {doc}")
            contents[doc] = code
    if not contents:
        raise ValueError("No code bodies found; supply --code-file with document IDs and code/original.code")
    # All known IDs stay in the mapping; only documents with code enter Lucene.
    mapping = [{"id": i, "text_id": doc} for i, doc in enumerate(sorted(known))]
    map_path = work / "document_map.jsonl"
    atomic_jsonl(map_path, mapping)
    collection = work / "corpus/documents.jsonl"
    atomic_jsonl(collection, ({"id": str(row["id"]), "contents": contents[row["text_id"]]}
                             for row in mapping if row["text_id"] in contents))
    queries = list(iter_json_records(metadata / "train_queries.jsonl"))
    topics = work / "queries.tsv"
    with topics.open("w", encoding="utf-8", newline="\n") as handle:
        for i, row in enumerate(queries):
            handle.write(f"{i}\t{row['prompt']}\n")
    paths = [collection, map_path, topics, *metadata.glob("*.jsonl"), metadata / "split_manifest.json"]
    manifest = {"schema": 1, "target_type": cfg["target_type"],
                "artifacts": {str(p): file_hash(p) for p in paths},
                "source_inputs": {p: file_hash(p) for p in sources},
                "indexed_documents": len(contents), "missing_code_documents": len(known) - len(contents),
                "training_queries": len(queries), "contents_policy": "code only; no query or positive-label text"}
    atomic_json(work / "export_manifest.json", manifest)
    return manifest


def pack(work, output):
    work, output = Path(work).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Cache exists; use a new cache path rather than overwrite an experiment input")
    manifest = read_json(work / "export_manifest.json")
    assert_file_hashes(manifest["artifacts"])
    completion = read_json(work / "retrieval_complete.json")
    assert_file_hashes(completion["artifacts"])
    if completion["export_fingerprint"] != file_hash(work / "export_manifest.json"):
        raise ValueError("Retrieval belongs to a different offline export")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    if temporary.exists():
        raise ValueError(f"Incomplete cache build exists: {temporary}")
    connection = sqlite3.connect(temporary)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript("""
            CREATE TABLE metadata(value TEXT NOT NULL);
            CREATE TABLE documents(id INTEGER PRIMARY KEY, text_id TEXT UNIQUE NOT NULL);
            CREATE TABLE queries(id INTEGER PRIMARY KEY, query_key TEXT UNIQUE NOT NULL, signature TEXT NOT NULL);
            CREATE TABLE hits(query INTEGER REFERENCES queries(id), rank INTEGER CHECK(rank>0),
                              doc INTEGER REFERENCES documents(id), score REAL,
                              PRIMARY KEY(query,rank)) WITHOUT ROWID;
        """)
        connection.executemany("INSERT INTO documents VALUES (?,?)", (
            (r["id"], r["text_id"]) for r in iter_json_records(work / "document_map.jsonl")))
        connection.executemany("INSERT INTO queries VALUES (?,?,?)", (
            (i, r["query_key"], query_signature(r)) for i, r in enumerate(
                iter_json_records(work / "metadata/train_queries.jsonl"))))
        count = 0
        with (work / "bm25_run.trec").open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 6:
                    raise ValueError("Expected six-column TREC retrieval output")
                query, _, doc, rank, score, _ = fields
                rank, score = int(rank), float(score)
                if not math.isfinite(score) or rank > completion["hits"]:
                    raise ValueError("Invalid BM25 score/rank")
                connection.execute("INSERT INTO hits VALUES (?,?,?,?)", (int(query), rank, int(doc), score))
                count += 1
        metadata = {"schema": 1, "target_type": manifest["target_type"],
                    "train_hash": file_hash(work / "metadata/train_queries.jsonl"),
                    "documents_hash": file_hash(work / "metadata/document_metadata.jsonl"),
                    "export": manifest, "retrieval": completion, "candidate_rows": count}
        connection.execute("INSERT INTO metadata VALUES (?)", (json.dumps(metadata),))
        connection.commit()
        connection.close()
        temporary.replace(output)
    finally:
        connection.close()
        temporary.unlink(missing_ok=True)
    return metadata


def retrieve(work, hits=200, threads=16, batch_size=32, runner=None):
    work = Path(work).resolve()
    if min(hits, threads, batch_size) < 1:
        raise ValueError("Retrieval counts must be positive")
    manifest = read_json(work / "export_manifest.json")
    assert_file_hashes(manifest["artifacts"])
    index, run = work / "index", work / "bm25_run.trec"
    if index.exists() or run.exists():
        raise ValueError("Retrieval outputs already exist; pack a completed retrieval or prepare a fresh work directory")
    runner = runner or (lambda cmd: subprocess.run(cmd, check=True))
    commands = [
        [sys.executable, "-m", "pyserini.index.lucene", "--collection", "JsonCollection",
         "--input", str(work / "corpus"), "--index", str(index),
         "--generator", "DefaultLuceneDocumentGenerator", "--threads", str(threads),
         "--storePositions", "--storeDocvectors", "--storeRaw"],
        [sys.executable, "-m", "pyserini.search.lucene", "--index", str(index),
         "--topics", str(work / "queries.tsv"), "--output", str(run), "--output-format", "trec",
         "--hits", str(hits), "--bm25", "--k1", "0.82", "--b", "0.68",
         "--threads", str(threads), "--batch-size", str(batch_size)],
    ]
    for command in commands:
        print("Running:", " ".join(command), flush=True)
        runner(command)
    try:
        version = importlib.metadata.version("pyserini")
    except importlib.metadata.PackageNotFoundError:
        version = None
    atomic_json(work / "retrieval_complete.json", {
        "export_fingerprint": file_hash(work / "export_manifest.json"),
        "artifacts": {str(run): file_hash(run)}, "hits": hits,
        "k1": 0.82, "b": 0.68, "pyserini": version, "commands": commands})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    exporter = commands.add_parser("prepare")
    exporter.add_argument("--config", type=Path, required=True)
    exporter.add_argument("--work-dir", type=Path, required=True)
    exporter.add_argument("--code-file", action="append", default=[])
    builder = commands.add_parser("build")
    builder.add_argument("--work-dir", type=Path, required=True)
    builder.add_argument("--output-cache", type=Path, required=True)
    builder.add_argument("--hits", type=int, default=200)
    builder.add_argument("--threads", type=int, default=16)
    builder.add_argument("--batch-size", type=int, default=32)
    packer = commands.add_parser("pack")
    packer.add_argument("--work-dir", type=Path, required=True)
    packer.add_argument("--output-cache", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        from pretrain.train_iterative_ddro_vault import load_config
        result = export(load_config(args.config.resolve()), args.work_dir, args.code_file)
    else:
        if args.action == "build":
            if args.output_cache.exists():
                parser.error("Cache already exists; use a new output path")
            retrieve(args.work_dir, args.hits, args.threads, args.batch_size)
        result = pack(args.work_dir, args.output_cache)
    print(json.dumps({k: v for k, v in result.items() if k not in {"export", "retrieval", "artifacts", "source_inputs"}}, indent=2))


if __name__ == "__main__":
    main()
