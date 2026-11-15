"""
Runs the two evaluation runs for the report.

Run 1: 10 questions, no judge, graph on.
Run 2: 30 questions, judge on, graph vs vector-only.

Usage:
    uv run python src/graph_rag/evaluation/run_eval.py --run 1
    uv run python src/graph_rag/evaluation/run_eval.py --run 2
"""

import argparse
import json
import os
import statistics
from types import SimpleNamespace

import eval as ev  # eval.py in the same folder

QUESTIONS_FILE = "src/graph_rag/evaluation/eval_questions_50.json"

# Which questions to use (change these lists if you want)
RUN1_IDS = ["q01", "q04", "q06", "q08", "q11", "q13",   # specific (6)
            "q16", "q17",                                # multi (2)
            "q41", "q42"]                                # unanswerable (2)

RUN2_IDS = ([f"q{i:02d}" for i in range(1, 16)]          # specific (15)
            + ["q16", "q17", "q18", "q22", "q27",
               "q31", "q32", "q34", "q35"]               # multi (9)
            + [f"q{i}" for i in range(41, 47)])          # unanswerable (6)


def load_questions(ids):
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        all_q = json.load(f)
    return [q for q in all_q if q["id"] in ids]


def get_judge_client():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("No ANTHROPIC_API_KEY, judge is off.")
        return None
    import anthropic
    return anthropic.Anthropic()


def run_all(questions, use_graph, client):
    args = SimpleNamespace(
        no_graph=not use_graph,
        judge_model="claude-sonnet-5-5",
        min_keyword_recall=0.5,
    )
    results = []
    for q in questions:
        print(f"-> {q['id']} ({q['type']})")
        results.append(ev.evaluate_question(q, args, client))
    return results


def summary(results):
    """Return the numbers we put in the report table."""
    total = len(results)
    ran = [r for r in results if "error" not in r]
    passed = sum(r["passed"] for r in results)

    out = {"pass": f"{ev.pct(passed, total)} ({passed} of {total})"}

    for t in ["specific", "multi", "unanswerable"]:
        rs = [r for r in results if r["type"] == t]
        p = sum(r["passed"] for r in rs)
        out[t] = f"{p} of {len(rs)}"

    hits = [r["hit"] for r in ran if "hit" in r]
    out["hit_rate"] = ev.pct(sum(hits), len(hits))

    for key in ["coverage", "mrr", "precision", "keyword_recall"]:
        out[key] = ev.mean([r.get(key) for r in ran])[0]

    answerable = [r for r in ran if r["type"] != "unanswerable"]
    cited = sum(r["has_citation"] for r in answerable)
    out["citation"] = f"{ev.pct(cited, len(answerable))} ({cited} of {len(answerable)})"

    unans = [r for r in ran if r["type"] == "unanswerable"]
    refused = sum(r["refused"] for r in unans)
    out["refusals"] = f"{ev.pct(refused, len(unans))} ({refused} of {len(unans)})"

    out["judge"] = ev.mean([r.get("judge") for r in ran])[0] or "not run"

    lats = [r["latency_s"] for r in ran]
    if lats:
        out["latency"] = (f"average {statistics.mean(lats):.1f} s, "
                          f"median {statistics.median(lats):.1f} s, "
                          f"max {max(lats):.1f} s")
    return out


def save(name, data):
    with open(name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {name}")


def run1():
    questions = load_questions(RUN1_IDS)
    results = run_all(questions, use_graph=True, client=None)  # no judge
    save("run1_results.json", results)
    print("\n===== RUN 1 =====")
    for k, v in summary(results).items():
        print(f"{k:15s} {v}")


def run2():
    questions = load_questions(RUN2_IDS)
    client = get_judge_client()

    print("\n--- with graph ---")
    graph = run_all(questions, use_graph=True, client=client)
    print("\n--- vector only ---")
    vector = run_all(questions, use_graph=False, client=client)

    save("run2_graph.json", graph)
    save("run2_vector.json", vector)

    sg, sv = summary(graph), summary(vector)
    print("\n===== RUN 2: graph vs vector-only =====")
    print(f"{'metric':15s} {'with graph':28s} vector only")
    for k in sg:
        print(f"{k:15s} {str(sg[k]):28s} {sv.get(k)}")

    # Questions where the two runs gave a different result
    print("\nQuestions that differed:")
    v_by_id = {r["id"]: r for r in vector}
    for g in graph:
        v = v_by_id[g["id"]]
        if g["passed"] != v["passed"] or g.get("judge") != v.get("judge"):
            print(f"  {g['id']}: graph passed={g['passed']} judge={g.get('judge')} | "
                  f"vector passed={v['passed']} judge={v.get('judge')}")
            print(f"     graph fail reasons : {g.get('fail_reasons')}")
            print(f"     vector fail reasons: {v.get('fail_reasons')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, choices=[1, 2], required=True)
    a = ap.parse_args()
    run1() if a.run == 1 else run2()