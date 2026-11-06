#!/usr/bin/env python3
"""
RAG eval runner for Graph RAG Studio.

Usage:
    uv add anthropic python-dotenv
    $env:ANTHROPIC_API_KEY="sk-ant-..."     # needed only for the LLM judge (or put it in .env)
    uv run python src/graph_rag/evaluation/eval.py --questions src/graph_rag/evaluation/eval_questions_50.json --out eval_results.json

questions.json format (list of objects):
[
  {
    "id": "q01",
    "type": "specific",                # specific | multi | unanswerable
    "question": "What limitations are mentioned in the short text topic modeling survey?",
    "expected_papers": ["Short Text Topic Modeling Techniques, Applications, and Performance: A Survey"],
    "keywords": ["sparsity", "co-occurrence", "external knowledge"],
    "reference": "optional reference answer for the judge",
    "require_citation": true           # optional, default true (set false for aggregate questions)
  },
  {
    "id": "q09",
    "type": "unanswerable",
    "question": "What does the survey say about the best topic model for Arabic short texts?"
  }
]

This script uses the QueryEngine directly (no HTTP API needed).
"""

import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:  # optional: load ANTHROPIC_API_KEY etc. from .env
    from dotenv import load_dotenv

    # Load .env from project root (same as config.py)
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)
except ImportError:
    pass

# --------------------------------------------------------------------------
# Detectors
# --------------------------------------------------------------------------

CITATION_RE = re.compile(
    "|".join(
        [
            r"\[[^\]]*chunk\s*\d+[^\]]*\]",       # [Paper, chunk 3] / [chunk 7]
            r"\[Source:[^\]]+\]",                  # [Source: ...]
            r"【[^】]*chunk\s*\d+[^】]*】",          # 【...chunk 14】
        ]
    ),
    re.IGNORECASE,
)


def has_citation(answer: str) -> bool:
    return bool(CITATION_RE.search(answer or ""))


REFUSAL_RE = re.compile(
    r"^\W*(?:"
    r"i (?:could not|couldn't|can't|cannot|was unable to) find"
    r"|there is no (?:information|mention|evidence)"
    r"|the (?:provided |retrieved )?(?:context|excerpts?|documents?|sources?|library|papers?|survey)"
    r"(?:\s+\w+){0,4}\s+(?:do(?:es)? not|don't|doesn't) "
    r"(?:contain|mention|include|provide|discuss|cover|recommend|address)"
    r"|no (?:relevant|mention|information|paper|papers|document|documents|source|sources)"
    r"|not (?:mentioned|covered|found) in"
    r"|i don't have (?:enough )?information"
    r"|none of the (?:papers|documents|sources)"
    r")",
    re.IGNORECASE,
)


REFUSAL_ANYWHERE_RE = re.compile(
    r"(?:"
    r"(?:could not|couldn't|can't|cannot|unable to|did not|didn't)\s+(?:find|identify|locate)"
    r"|(?:do(?:es)? not|don't|doesn't)\s+(?:contain|mention|include|provide|discuss|cover|address)"
    r"|no (?:mention|information|evidence)"
    r"|(?:insufficient|not enough) (?:evidence|information)"
    r")",
    re.IGNORECASE,
)


def _clean_for_refusal(answer: str) -> str:
    # models often write curly apostrophes, which never match a plain one in the patterns
    answer = (answer or "").replace("\u2019", "'").replace("\u2018", "'")
    answer = re.sub(r"[*#>_`]", "", answer).strip()
    answer = re.sub(r"^\s*answer\s*[:\-]?\s*", "", answer, flags=re.IGNORECASE)
    return answer.strip()


def is_refusal(answer: str) -> bool:
    answer = _clean_for_refusal(answer)
    head = answer[:300]
    if REFUSAL_RE.search(head):
        return True
    # a negative statement about the evidence in the first sentence (works with long paper titles)
    first = re.split(r"(?<=[.!?])\s", answer, maxsplit=1)[0][:250]
    if REFUSAL_ANYWHERE_RE.search(first):
        return True
    # short answer with no citation that contains a negation -> likely a refusal
    return (
        len(answer.split()) < 60
        and not has_citation(answer)
        and re.search(r"\b(not|no)\b", head.lower()) is not None
    )


# --------------------------------------------------------------------------
# Retrieval metrics
# --------------------------------------------------------------------------

SOURCE_RE = re.compile(r"^(?P<paper>.*?)\s*\(chunk\s*(?P<chunk>\d+)\)\s*$", re.IGNORECASE)


def source_paper(src: str) -> str:
    m = SOURCE_RE.match(src.strip())
    return (m.group("paper") if m else src).strip()


def norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def title_key(s: str) -> str:
    """Letters/digits only, lowercased: ignores spaces, hyphens, punctuation (PDF-extraction glitches)."""
    return re.sub(r"[\W_]+", "", s.lower())


def titles_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    from difflib import SequenceMatcher

    return SequenceMatcher(None, a, b).ratio() >= 0.9


def retrieval_metrics(sources, expected_papers):
    """hit, precision, coverage, mrr. Returns None for questions with no expected papers."""
    if not expected_papers:
        return None
    expected = [title_key(p) for p in expected_papers]
    papers = [title_key(source_paper(s)) for s in sources]
    relevant = [any(titles_match(p, e) for e in expected) for p in papers]

    hit = any(relevant)
    precision = sum(relevant) / len(papers) if papers else 0.0
    covered = sum(any(titles_match(p, e) for p in papers) for e in expected)
    coverage = covered / len(expected)
    mrr = 0.0
    for i, r in enumerate(relevant, start=1):
        if r:
            mrr = 1.0 / i
            break
    return {
        "hit": hit,
        "precision": round(precision, 3),
        "coverage": round(coverage, 3),
        "mrr": round(mrr, 3),
    }


def keyword_recall(answer: str, keywords):
    if not keywords:
        return None
    def clean(s):
        # The model often writes non-breaking hyphens (U+2011) that never match a plain "-".
        return re.sub(r"[\u2010-\u2015\u2212]", "-", s).lower()

    a = clean(answer)
    found = [k for k in keywords if clean(k) in a]
    return round(len(found) / len(keywords), 3), [k for k in keywords if k not in found]


# --------------------------------------------------------------------------
# QueryEngine client (direct call, no HTTP)
# --------------------------------------------------------------------------

from graph_rag.retriever.query_engine import query_engine
from graph_rag.config import settings


def _format_source(s) -> str:
    """Normalize a source to 'Paper title (chunk N)'."""
    return str(s).strip()


def call_query_engine(question: str, use_graph: bool = True):
    """Call QueryEngine directly. Returns (answer, sources, latency_s, error)."""
    import time
    from graph_rag.config import settings
    
    # Temporarily override USE_GRAPH setting
    original_use_graph = settings.USE_GRAPH
    settings.USE_GRAPH = use_graph
    
    t0 = time.time()
    try:
        response = query_engine.query(
            question=question,
            session_id=None,
            user_id=None,
            message_id=None,
            run_evaluation=False,  # We'll do our own evaluation
        )
        latency = time.time() - t0
        return response.answer, response.sources, latency, None
    except Exception as e:
        latency = time.time() - t0
        return None, [], latency, {"kind": "query_engine_error", "detail": str(e)[:300]}
    finally:
        # Restore original setting
        settings.USE_GRAPH = original_use_graph


# --------------------------------------------------------------------------
# LLM judge (runs on EVERY question)
# --------------------------------------------------------------------------

ANSWERABLE_JUDGE_PROMPT = """You are grading a RAG system's answer about a library of research papers.

Question: {question}
Reference answer / key points (may be empty): {reference}
System answer: {answer}
Sources returned: {sources}

Score 1-5:
5 = Fully correct, complete, consistent with the reference; no fabricated specifics.
4 = Mostly correct; minor omissions or slightly imprecise details.
3 = Partially correct; notable gaps or some doubtful claims.
2 = Mostly wrong, off-topic, or built on irrelevant sources.
1 = Wrong or fabricated.

Be skeptical of specific numbers, metrics, dataset names that look unsupported.
Return ONLY JSON: {{"score": int, "suspicious_claims": [str], "reason": str}}"""

UNANSWERABLE_JUDGE_PROMPT = """You are grading a RAG system on a question that CANNOT be answered from the library.

Question: {question}
System answer: {answer}
Sources returned: {sources}

Score 1-5:
5 = Clearly states the library doesn't contain this information; no fabricated claims;
    optionally mentions what the papers DO cover.
4 = Declines correctly but with minor unnecessary hedging or tangential info.
3 = Declines but also presents loosely related content as if it were an answer.
2 = Gives a confident answer built mostly from tangential sources.
1 = Fabricates specifics (methods, numbers, paper names) to answer.

Return ONLY JSON: {{"score": int, "suspicious_claims": [str], "reason": str}}"""


def judge(anthropic_client, model: str, q: dict, answer: str, sources):
    tmpl = UNANSWERABLE_JUDGE_PROMPT if q["type"] == "unanswerable" else ANSWERABLE_JUDGE_PROMPT
    prompt = tmpl.format(
        question=q["question"],
        reference=q.get("reference", ""),
        answer=answer,
        sources=json.dumps(sources, ensure_ascii=False),
    )
    try:
        if hasattr(anthropic_client, "messages"):  # Anthropic client
            msg = anthropic_client.messages.create(
                model=model, max_tokens=500,
                messages=[{"role": "user", "content": prompt}])
            text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        else:  # OpenAI-compatible client (Gemini, Groq, Ollama...)
            res = anthropic_client.chat.completions.create(
                model=model, max_tokens=1500, temperature=0,
                messages=[{"role": "user", "content": prompt}])
            text = res.choices[0].message.content or ""
        a, b = text.find("{"), text.rfind("}")  # tolerate extra text around the JSON
        j = json.loads(text[a:b + 1])
        return int(j["score"]), j.get("reason", ""), j.get("suspicious_claims", [])
    except Exception as e:  # judge failure must not kill the run
        return None, f"judge_error: {e}", []


# --------------------------------------------------------------------------
def evaluate_question(q, args, anthropic_client):
    rec = {"id": q["id"], "type": q["type"], "question": q["question"]}

    answer, sources, latency, err = call_query_engine(q["question"], use_graph=not args.no_graph)
    rec["latency_s"] = round(latency, 2)
    if err:
        rec["error"] = err
        rec["passed"] = False
        return rec
    rec.update(
        answer=answer,
        sources=sources,
        n_sources=len(sources),
        has_citation=has_citation(answer),
        refused=is_refusal(answer),
    )

    rm = retrieval_metrics(sources, q.get("expected_papers", []))
    if rm:
        rec.update(rm)
    kr = keyword_recall(answer, q.get("keywords", []))
    if kr:
        rec["keyword_recall"], rec["missing_keywords"] = kr

    if anthropic_client:
        score, reason, suspicious = judge(anthropic_client, args.judge_model, q, answer, sources)
        rec["judge"], rec["judge_reason"], rec["suspicious_claims"] = score, reason, suspicious
    else:
        rec["judge"] = None

    # ---- pass/fail ----
    j = rec["judge"]
    reasons = []
    if q["type"] == "unanswerable":
        if not rec["refused"]:
            reasons.append("did not refuse")
        if j is not None and j < 4:
            reasons.append(f"judge score {j} < 4")
    else:
        if rec["refused"]:
            reasons.append("answer is a refusal")
        if q.get("require_citation", True) and not rec["has_citation"]:
            reasons.append("no citation")
        if "hit" in rec and not rec["hit"]:
            reasons.append("no expected paper retrieved")
        if "keyword_recall" in rec and rec["keyword_recall"] < args.min_keyword_recall:
            reasons.append(f"keyword recall {rec['keyword_recall']} < {args.min_keyword_recall}")
        if j is not None and j < 4:
            reasons.append(f"judge score {j} < 4")
    rec["fail_reasons"] = reasons
    rec["passed"] = not reasons
    return rec


def mean(vals):
    vals = [v for v in vals if v is not None]
    return (round(statistics.mean(vals), 3), len(vals)) if vals else (None, 0)


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "n/a"


def summarize(results):
    total = len(results)
    ran = [r for r in results if "error" not in r]
    errored = [r for r in results if "error" in r]

    print("\n===== SUMMARY =====")
    print(f"Questions: {total}   ran: {len(ran)}   errored: {len(errored)}")
    for r in errored:
        e = r["error"]
        print(f"  ERROR {r['id']} ({r['type']}): {e['kind']} {e.get('status', '')} after {e.get('attempts', 1)} attempt(s)")
        if e.get("body"):
            print(f"        body: {e['body'][:300]}")

    n_pass = sum(r["passed"] for r in results)
    print(f"\nPass rate (errors count as FAIL): {n_pass}/{total}  ({pct(n_pass, total)})")
    for t in ["specific", "multi", "unanswerable"]:
        rs = [r for r in results if r["type"] == t]
        if rs:
            p = sum(r["passed"] for r in rs)
            print(f"  {t:13s} pass {p}/{len(rs)}  ({pct(p, len(rs))})   (ran {sum('error' not in r for r in rs)}/{len(rs)})")

    print("\nRetrieval metrics (answerable only):")
    for k in ["precision", "coverage", "mrr"]:
        m, n = mean([r.get(k) for r in ran])
        print(f"  {k:10s} {m}  (n={n})")
    hits = [r["hit"] for r in ran if "hit" in r]
    if hits:
        print(f"  hit-rate   {sum(hits)/len(hits):.3f}  ({pct(sum(hits), len(hits))}, n={len(hits)})")

    m, n = mean([r.get("keyword_recall") for r in ran])
    print(f"Keyword recall: {m}  (n={n})")
    ans = [r for r in ran if r["type"] != "unanswerable"]
    if ans:
        c = sum(r["has_citation"] for r in ans)
        print(f"Citation rate (answerable): {c}/{len(ans)}  ({pct(c, len(ans))})")
    un = [r for r in ran if r["type"] == "unanswerable"]
    if un:
        c = sum(r["refused"] for r in un)
        print(f"Correct refusals (unanswerable): {c}/{len(un)}  ({pct(c, len(un))})")
    m, n = mean([r.get("judge") for r in ran])
    print(f"Judge score (1-5): {m}  (n={n} of {len(ran)} ran)")

    lats = [r["latency_s"] for r in ran]
    if lats:
        print(f"Latency  avg={statistics.mean(lats):.1f}s  p50={statistics.median(lats):.1f}s  max={max(lats):.1f}s  (n={len(lats)})")

    flagged = [r for r in ran if r.get("suspicious_claims")]
    if flagged:
        print("\nQuestions with suspicious claims flagged by judge:")
        for r in flagged:
            print(f"  {r['id']}: {r['suspicious_claims'][:3]}")

    failed = [r for r in results if not r["passed"]]
    if failed:
        print("\nFailed questions and why:")
        for r in failed:
            why = r.get("fail_reasons") or ([f"error: {r['error']['kind']}"] if "error" in r else ["unknown"])
            print(f"  {r['id']} ({r['type']}): {', '.join(why)}")

    # ---- overall percentages ----
    ran_pass = sum(r["passed"] for r in ran)
    print("\n===== OVERALL EVALUATION =====")
    print(f"Overall pass rate (all questions):  {pct(n_pass, total)}  ({n_pass}/{total})")
    print(f"Pass rate (questions that ran):     {pct(ran_pass, len(ran))}  ({ran_pass}/{len(ran)})")
    if hits:
        print(f"Retrieval hit-rate:                 {pct(sum(hits), len(hits))}")
    if ans:
        print(f"Citation rate:                      {pct(sum(r['has_citation'] for r in ans), len(ans))}")
    if un:
        print(f"Correct refusals:                   {pct(sum(r['refused'] for r in un), len(un))}")
    jm, jn = mean([r.get("judge") for r in ran])
    if jm is not None:
        print(f"Judge score:                        {pct(jm, 5)}  (avg {jm}/5, n={jn})")
    else:
        print("Judge score:                        n/a (judge did not run)")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="src/graph_rag/evaluation/eval_questions_50.json")
    ap.add_argument("--out", default="eval_results.json")
    ap.add_argument("--judge-model", default="claude-sonnet-5-5")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated question ids to run, e.g. q08,q09,q10 "
                    "(results are merged into the existing --out file)")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to wait between questions")
    ap.add_argument("--min-keyword-recall", type=float, default=0.5)
    ap.add_argument("--stop-on-error", action="store_true",
                    help="stop at the first failed request (saves results collected so far)")
    ap.add_argument("--no-graph", action="store_true", help="vector-only baseline")
    ap.add_argument("--judge-provider", default="anthropic", choices=["anthropic", "openai"],
                    help="'openai' = any OpenAI-compatible judge (Gemini free tier, Groq, Ollama)")
    ap.add_argument("--limit", type=int, default=0, help="limit to first N questions (0 = no limit)")
    ap.add_argument("--abort-after-consecutive-errors", type=int, default=3,
                    help="stop eval after N consecutive errors (0 = never abort)")
    args = ap.parse_args()

    with open(args.questions, encoding="utf-8") as f:
        questions = json.load(f)

    anthropic_client = None
    if not args.no_judge and args.judge_provider == "openai":
        if not (os.environ.get("JUDGE_API_KEY") and os.environ.get("JUDGE_BASE_URL")):
            print("[warn] JUDGE_API_KEY / JUDGE_BASE_URL not set in .env -> judge disabled.")
        else:
            from openai import OpenAI
            anthropic_client = OpenAI(api_key=os.environ["JUDGE_API_KEY"],
                                      base_url=os.environ["JUDGE_BASE_URL"])
            args.judge_model = os.environ.get("JUDGE_MODEL", args.judge_model)
            print(f"[info] judge: {args.judge_model} via {os.environ['JUDGE_BASE_URL']}")
    elif not args.no_judge:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("[warn] ANTHROPIC_API_KEY is not set -> judge disabled. "
                  "Set it (or use .env) or pass --no-judge to silence this.")
        else:
            try:
                import anthropic
                anthropic_client = anthropic.Anthropic()
            except Exception as e:
                print(f"[warn] judge disabled: {e}")

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    to_run = [q for q in questions if not only or q["id"] in only]
    
    # Apply limit if specified
    if args.limit > 0:
        to_run = to_run[:args.limit]
        print(f"[info] Limited to first {args.limit} questions")

    # Warm up LLM to avoid counting model loading time
    print("[info] Warming up LLM...")
    try:
        query_engine.warm_up()
    except Exception as e:
        print(f"[warn] LLM warm-up failed: {e}")

    new_results = []
    consecutive_errors = 0
    try:
        for i, q in enumerate(to_run):
            print(f"-> {q['id']} ({q['type']}) {q['question'][:70]}")
            rec = evaluate_question(q, args, anthropic_client)
            status = "ERROR" if "error" in rec else ("PASS" if rec["passed"] else "FAIL")
            print(f"   {status}  {rec['latency_s']}s")
            new_results.append(rec)
            if "error" in rec:
                e = rec["error"]
                consecutive_errors += 1
                print(f"   -> {e.get('kind')}: {str(e.get('detail', ''))[:300]}")
                if args.stop_on_error:
                    print("[stop] first error reached. Fix the cause, then re-run the failed "
                          "questions with --only.")
                    break
                if args.abort_after_consecutive_errors > 0 and consecutive_errors >= args.abort_after_consecutive_errors:
                    print(f"[abort] {consecutive_errors} consecutive errors reached. Stopping evaluation.")
                    break
            if args.delay and i < len(to_run) - 1:
                import time
                time.sleep(args.delay)
    except KeyboardInterrupt:
        print("\n[interrupted] saving the results collected so far...")

    results = new_results
    if only and os.path.exists(args.out):
        # Merge re-run questions into the previous results, keeping the question-file order.
        try:
            with open(args.out, encoding="utf-8") as f:
                old = {r["id"]: r for r in json.load(f)}
        except Exception:
            old = {}
        old.update({r["id"]: r for r in new_results})
        results = [old[q["id"]] for q in questions if q["id"] in old]

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    summarize(results)
    print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()