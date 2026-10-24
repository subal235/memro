# LongMemEval memory experiments

This runner evaluates four isolated memory configurations against the same selected
LongMemEval-S questions. Each arm creates a fresh agent per question; arm outputs
are stored independently so the existing judge/report tools can grade each arm.

## Arms

- `baseline`: existing exchange-sized memories and hybrid retrieval.
- `graph-rerank`: reranks only the existing hybrid candidate pool using typed
  relations between candidates scoped to the benchmark agent and tenant. It does
  not inject graph-only memories. For a meaningful run, start the server with
  `GRAPH_WRITE_EDGES=true`; graph retrieval itself can remain disabled. If there
  are no edges among returned candidates, the arm reports a no-op rather than
  fabricating a benefit.
- `atomic-source`: stores each original user/assistant turn as its own memory,
  sets `is_atomic=true`, and attaches stable `source_ref`, source session,
  turn index, and role metadata. This is source-linked turn atomization, not
  LLM fact extraction; it deliberately isolates granularity/provenance from an
  additional model call.
- `temporal`: baseline ingestion plus `temporal_weight=0.30` anchored to the
  question date. Treat 0.30 as an experimental starting point, not a tuned value.

## Run

From `bench/longmemeval`, with the local server and benchmark database running:

```bash
python memory_experiments.py --pilot --name stage-d-pilot
python memory_experiments.py --subset --name stage-d-100
python memory_experiments.py --subset --no-llm --name stage-d-retrieval-only
```

Run a single arm with `--arms graph-rerank`, or any combination, e.g.
`--arms baseline graph-rerank`. The script resumes completed question IDs.

Grade each arm by pointing the existing judge/report scripts at the corresponding
`runs/<name>/<arm>/hyp.jsonl` and metadata directory. Compare paired question IDs;
do not compare aggregate percentages without confirming identical questions,
answer model, judge model, prompt, server commit, and dataset snapshot.

## Interpretation and limitations

- The graph reranker uses a conservative deterministic relation bonus on the
  current top-K candidate pool; it cannot recover a relevant memory absent from
  that pool. The base rank remains dominant and no new memories are inserted.
- Temporal ranking is a separate arm; it does not also enable graph reranking.
- The atomic arm increases memory count and embedding work, so compare cost,
  ingest time, evidence coverage, answer accuracy, and latency together.
- This runner does not claim a score improvement. Keep changes experimental until
  the benchmark demonstrates a repeatable gain with no material regression in
  aggregation, temporal reasoning, or multi-session categories.
