"""Four-arm LongMemEval experiment runner for Memro memory-quality work.

Arms:
  baseline       Existing exchange-sized ingestion + hybrid retrieval.
  graph-rerank   Same ingestion/retrieval; rerank only the existing hybrid pool
                 using scoped edges among candidates. Never injects candidates.
  atomic-source  One memory per source turn, marked atomic and annotated with
                 a stable LongMemEval source reference in metadata.
  temporal       Baseline ingestion + event-date-aware search (weight 0.30).

Run from bench/longmemeval:
  python memory_experiments.py --pilot
  python memory_experiments.py --subset
  python memory_experiments.py --subset --no-llm
  python memory_experiments.py --all --name stage-d

Each arm gets a fresh agent per question, so ingestion strategies do not share
memory state. Outputs are written to runs/<name>/<arm>/hyp.jsonl and meta.json.
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import pathlib
import threading
import time
import uuid

from common import RUNS, RunTracker, set_run_tracker
import run as base_run

ARMS = ("baseline", "graph-rerank", "atomic-source", "temporal")
ZERO_TENANT = "00000000-0000-0000-0000-000000000000"
GRAPH_WEIGHT = {"updates": 1.0, "extends": 0.8, "derives": 0.8, "associated": 0.35, "causes": 0.7}


def source_turns(session):
    """Yield (turn_index, role, text) without merging separate source turns."""
    for i, message in enumerate(session):
        role = str(message.get("role", "unknown"))
        content = str(message.get("content", ""))
        if content.strip():
            rendered = f"{role.capitalize()}: {content}"
            # Keep each memory under the server's 10k-character limit while
            # retaining the same stable source-turn reference for every chunk.
            for start in range(0, len(rendered), base_run.MAX_MEMORY_CHARS):
                yield i, role, rendered[start:start + base_run.MAX_MEMORY_CHARS]


def ingest_question(q, client, agent, atomic=False):
    meta = {}
    session_memories = {}
    started = time.time()
    for sid, date, session in zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"]):
        when = base_run.parse_date(date)
        session_memories[sid] = []
        session_uuid = str(uuid.uuid5(base_run.NS_SESSION, f"{q['question_id']}:{sid}"))
        if atomic:
            items = source_turns(session)
        else:
            items = ((i, "exchange", text) for i, text in enumerate(base_run.rounds(session)))
        for turn_index, role, content in items:
            source_ref = f"longmemeval:{q['question_id']}:{sid}:turn:{turn_index}"
            memory = client.create_memory(
                agent.agent_id,
                agent._private_key,
                content,
                "episodic",
                event_date=when.strftime("%Y-%m-%dT%H:%M:%SZ"),
                session_id=session_uuid,
                is_atomic=atomic,
                metadata={
                    "benchmark": "longmemeval-s",
                    "source_ref": source_ref,
                    "source_session_id": sid,
                    "source_turn_index": turn_index,
                    "source_role": role,
                    "source_session_date": date,
                    "event_date_basis": "session_date_proxy",
                    "question_id": q["question_id"],
                },
            )
            mid = memory["id"]
            meta[mid] = {"session": sid, "date": date, "content": content,
                         "source_ref": source_ref, "source_turn_index": turn_index}
            session_memories[sid].append(mid)
    ingest_s = time.time() - started
    deadline = time.time() + 1800
    while (left := base_run.pending_embeddings(agent.agent_id)) and time.time() < deadline:
        time.sleep(1.5)
    if left:
        raise RuntimeError(f"{left} memories still not embedded")
    return meta, session_memories, ingest_s, time.time() - started - ingest_s


def graph_rerank_only(hits, agent_id):
    """Reorder existing hybrid hits using relations within the candidate pool.

    No memory is added or removed. Base rank remains dominant; typed relation
    support from the top 3 seeds is a small, deterministic tie-breaking signal.
    """
    if len(hits) < 2:
        return hits, {"applied": False, "reason": "fewer_than_two_candidates", "edges": 0}
    ids = [h.get("memory_id", "") for h in hits]
    if any(not x or not all(c in "0123456789abcdefABCDEF-" for c in x) for x in ids):
        return hits, {"applied": False, "reason": "invalid_candidate_id", "edges": 0}
    quoted = ",".join("'" + x + "'" for x in ids)
    sql = (
        "SELECT source_memory_id::text, target_memory_id::text, relation_type, "
        "COALESCE(confidence,1.0)::text FROM memory_relations "
        f"WHERE tenant_id = '{ZERO_TENANT}' AND created_by = '{agent_id}' "
        f"AND source_memory_id IN ({quoted}) AND target_memory_id IN ({quoted})"
    )
    try:
        raw = base_run.psql(sql)
    except Exception as exc:
        return hits, {"applied": False, "reason": "edge_query_failed", "error": str(exc)[:200], "edges": 0}

    rank = {h["memory_id"]: i for i, h in enumerate(hits)}
    base = {h["memory_id"]: 1.0 / (60.0 + i + 1) for i, h in enumerate(hits)}
    bonus = {mid: 0.0 for mid in ids}
    edge_count = 0
    seed_ids = set(ids[:min(3, len(ids))])
    for line in raw.splitlines():
        cols = line.split("|")
        if len(cols) != 4:
            continue
        source, target, relation, confidence_text = cols
        try:
            confidence = max(0.0, min(1.0, float(confidence_text)))
        except ValueError:
            confidence = 0.0
        weight = GRAPH_WEIGHT.get(relation, 0.2)
        edge_count += 1
        if source in seed_ids and target not in seed_ids:
            bonus[target] += 0.03 * base[source] * confidence * weight
        elif relation == "associated" and target in seed_ids and source not in seed_ids:
            bonus[source] += 0.03 * base[target] * confidence * weight

    reranked = sorted(hits, key=lambda h: (base[h["memory_id"]] + bonus[h["memory_id"]],
                                            -rank[h["memory_id"]]), reverse=True)
    return reranked, {"applied": edge_count > 0, "reason": "ok" if edge_count else "no_candidate_edges",
                      "edges": edge_count, "candidates": len(hits), "injected": 0}


def retrieve_arm(q, client, agent, arm):
    atomic = arm == "atomic-source"
    meta, session_memories, ingest_s, embed_s = ingest_question(q, client, agent, atomic=atomic)
    search_started = time.time()
    options = {
        "agent_id": agent.agent_id,
        "query": q["question"][:500],
        "limit": base_run.TOP_K,
        "private_key": agent._private_key,
        "mode": "hybrid",
        "order": "score",
        "resolve_conflicts": True,
        "return_response": True,
    }
    if arm == "temporal":
        options["temporal_weight"] = 0.30
        options["anchor_date"] = base_run.parse_date(q["question_date"]).strftime("%Y-%m-%dT%H:%M:%SZ")
    response = client.search(**options)
    search_ms = (time.time() - search_started) * 1000
    hits = response.get("results", [])
    rerank_diag = None
    if arm == "graph-rerank":
        rerank_started = time.time()
        hits, rerank_diag = graph_rerank_only(hits, agent.agent_id)
        rerank_diag["rerank_ms"] = round((time.time() - rerank_started) * 1000, 2)
    return hits, meta, session_memories, {
        "ingest_s": round(ingest_s, 2),
        "embed_wait_s": round(embed_s, 2),
        "search_ms": round(search_ms, 2),
        "graph_rerank": rerank_diag,
        "memories_written": len(meta),
        "memories": len(meta),
        "atomic_ingestion": atomic,
        "temporal_weight": 0.30 if arm == "temporal" else 0.0,
    }


def run_arm(q, arm, no_llm=False):
    from memro import MemroClient

    client = MemroClient(base_run.API, api_key=None)
    agent = base_run.new_agent(client)
    try:
        hits, meta, session_memories, timings = retrieve_arm(q, client, agent, arm)
        hyp = "(no LLM: retrieval dry-run)" if no_llm else base_run.answer(q, hits, meta, session_memories)
        rows = []
        for h in hits:
            mid = h.get("memory_id")
            m = meta.get(mid, {})
            rows.append({
                "memory_id": mid,
                "session": m.get("session"),
                "source_ref": m.get("source_ref"),
                "score": h.get("score"),
                "content": h.get("content", m.get("content", ""))[:base_run.PER_ITEM_CHARS],
                "is_superseded": h.get("is_superseded", False),
            })
        return {
            "question_id": q["question_id"],
            "question_type": q["question_type"],
            "question": q["question"],
            "answer": q["answer"],
            "hypothesis": hyp,
            "arm": arm,
            "retrieved_sessions": [meta[h["memory_id"]]["session"] for h in hits if h.get("memory_id") in meta],
            "answer_session_ids": q["answer_session_ids"],
            "retrieved_evidence": rows,
            **timings,
        }
    finally:
        base_run.drop_agent(agent)


def select_questions(data, args):
    if args.pilot:
        by_type = {}
        for q in sorted(data, key=lambda item: item["question_id"]):
            by_type.setdefault(q["question_type"], [])
            if len(by_type[q["question_type"]]) < 2:
                by_type[q["question_type"]].append(q)
        return [q for values in by_type.values() for q in values]
    if args.subset:
        path = pathlib.Path(args.subset)
        if not path.exists():
            path = base_run.BENCH / args.subset
        ids = set(json.loads(path.read_text()))
        return [q for q in data if q["question_id"] in ids]
    return data


def main():
    parser = argparse.ArgumentParser(description="Controlled four-arm LongMemEval memory experiments")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot", action="store_true", help="12 questions (2 per type)")
    group.add_argument("--subset", nargs="?", const="subset_100.json", default=None, help="100-question subset")
    group.add_argument("--all", action="store_true", help="all available questions")
    parser.add_argument("--name", default="memory-experiments")
    parser.add_argument("--no-llm", action="store_true", help="retrieval-only dry run")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    args = parser.parse_args()

    data = select_questions(base_run.load_data(), args)
    root = RUNS / args.name
    root.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        out_dir = root / arm
        out_dir.mkdir(parents=True, exist_ok=True)
        tracker = RunTracker(out_dir)
        tracker.update_metadata(
            name=f"{args.name}/{arm}", arm=arm, commit=base_run.get_server_commit(),
            created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            questions_total=len(data), answer_model=base_run.ANSWER_MODEL,
            top_k=base_run.TOP_K, mode="hybrid", no_llm=args.no_llm,
            temporal_weight=0.30 if arm == "temporal" else 0.0,
            ingestion_strategy="source_turns" if arm == "atomic-source" else "exchange",
        )
        hyp_file = out_dir / "hyp.jsonl"
        done = set()
        if hyp_file.exists():
            for line in hyp_file.read_text().splitlines():
                if line.strip():
                    try:
                        done.add(json.loads(line)["question_id"])
                    except (ValueError, KeyError):
                        pass
        todo = [q for q in data if q["question_id"] not in done]
        print(f"[{arm}] {len(done)} complete, {len(todo)} remaining -> {hyp_file}", flush=True)
        for q in todo:
            row = run_arm(q, arm, args.no_llm)
            with hyp_file.open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{arm}] {q['question_id']} {q['question_type']} "
                  f"memories={row['memories_written']} search={row['search_ms']}ms", flush=True)
        tracker.update_metadata(completed=len(done) + len(todo), finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
    print(f"Done. Results: {root}", flush=True)


if __name__ == "__main__":
    main()
