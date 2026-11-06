"""Render-Reading Instruments (RRI) on SFE.

RRI reads the registered items with its instruments and commits the reading;
the MLLM answers every other item with the official SFE prompt, and the LLM
judges score every answer with the official SFE rubric.  The MLLM and the
judges are served by any OpenAI-compatible endpoint:

    export OPENAI_API_KEY=...
    export OPENAI_BASE_URL=https://api.openai.com/v1
    python run_rri.py --data /path/to/SFE --model <mllm>
"""
import argparse
import base64
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
from PIL import Image

import rri

FIELDS = ("astronomy", "chemistry", "earth", "life", "material")

# The paper scores the 819 items of all tasks except E018.
SKIP_TASKS = ("E018",)

TYPE_NOTE = {
    "mcq": "The question is a multiple-choice question. "
           "Answer with the option letter from the given choices.",
    "exact_match": "The question is an exact match question. "
                   "Answer the question using a single word or phrase.",
    "open_ended": "The question is an open-ended question. Answer the question using a phrase.",
}

JUDGE_SYSTEM = "You are a helpful and precise assistant for checking the correctness of the answer."

JUDGE_RULES = """You are a strict evaluator assessing answer correctness. You must score the model's prediction on a scale from 0 to 10, where 0 represents an entirely incorrect answer and 10 indicates a highly correct answer.
# Input
Question:
```
{question}
```
Ground Truth Answer:
```
{answer}
```
Model Prediction:
```
{pred}
```


# Evaluation Rules
- The model prediction may contain the reasoning process, you should spot the final answer from it.
- For multiple-choice questions: Assign a higher score if the predicted answer matches the ground truth, either by option letters or content. Include partial credit for answers that are close in content.
- For exact match and open-ended questions:
  * Assign a high score if the prediction matches the answer semantically, considering variations in format.
  * Deduct points for partially correct answers or those with incorrect additional information.
- Ignore minor differences in formatting, capitalization, or spacing since the model may explain in a different way.
- Treat numerical answers as correct if they match within reasonable precision
- For questions requiring units, both value and unit must be correct

# Scoring Guide
Provide a single integer from 0 to 10 to reflect your judgment of the answer's correctness.

# Strict Output format example
4"""


def official_prompt(doc):
    """Official SFE prompt."""
    q = "You are an expert in %s and need to solve the following question.\n%s\n\n%s" % (
        doc["field"], TYPE_NOTE[doc["question_type"]], doc["question"])
    if doc["question_type"] == "mcq":
        q += "\n" + "\n".join(doc["options"])
    return q


def encode(path):
    """PNG data URL of the image at its original resolution."""
    buf = io.BytesIO()
    Image.open(path).convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def chat(model, messages, retries=8):
    """Text of one chat completion, or None when every attempt fails."""
    url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url, data=json.dumps({"model": model, "messages": messages}).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "")})
    for k in range(retries + 1):
        code = None
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                content = json.load(r)["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(p.get("text", "") for p in content)
            return content or ""
        except urllib.error.HTTPError as e:
            code = e.code
        except Exception:
            pass
        if code is not None and 400 <= code < 500 and code not in (408, 429):
            break
        if k < retries:
            time.sleep(min(5 * 2 ** k, 60) if code == 429 else 3 + 4 * k)
    return None


def load(path):
    out = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    o = json.loads(line)
                    out[o["id"]] = o
    return out


def run(fn, todo, path, store, workers, what):
    print("%s: %d to do" % (what, len(todo)), flush=True)
    with open(path, "a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as ex:
        for i, rec in enumerate(ex.map(fn, todo), 1):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            store[rec["id"]] = rec
            if i % 100 == 0:
                print("  %d / %d" % (i, len(todo)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="SFE root with 20250611-<lang>/ and images/")
    ap.add_argument("--lang", default="en", choices=["en", "zh"])
    ap.add_argument("--model", required=True, help="MLLM that answers the items RRI does not read")
    ap.add_argument("--judges", nargs="+", default=["gpt-6-astra", "gpt-6-luna"])
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tag = lambda s: re.sub(r"[^\w.-]+", "_", s)  # noqa: E731

    items, missing = [], 0
    for doc in pq.read_table(os.path.join(args.data, "20250611-" + args.lang,
                                          "test.parquet")).to_pylist():
        if doc["id"].split("/")[0] in SKIP_TASKS:
            continue
        paths = [os.path.join(args.data, "images", os.path.basename(str(p).replace("\\", "/")))
                 for p in doc["images"] or []]
        missing += sum(not os.path.exists(p) for p in paths)
        items.append((doc, paths))
    if missing:
        raise SystemExit("%d images are missing under %s" % (missing, os.path.join(args.data, "images")))

    # 1. answers: committed RRI reading on registered items, MLLM answer otherwise
    def answer(item):
        doc, paths = item
        reading = rri.read(doc, paths)
        if reading is not None and reading["answer"] is not None:
            return {"id": doc["id"], "answer": reading["answer"], "source": "rri"}
        content = [{"type": "image_url", "image_url": {"url": encode(p)}} for p in paths]
        content.append({"type": "text", "text": official_prompt(doc)})
        pred = chat(args.model, [{"role": "user", "content": content}])
        return {"id": doc["id"], "answer": pred or "", "source": "mllm", "failed": pred is None}

    ans_path = os.path.join(args.out, "%s_answers.jsonl" % tag(args.model))
    answers = load(ans_path)
    run(answer, [it for it in items if it[0]["id"] not in answers or answers[it[0]["id"]].get("failed")],
        ans_path, answers, args.workers, "answers")

    # 2. judges: official rubric; an empty answer or a failed judge call scores 0
    results = {}
    for judge in args.judges:
        def score(item, judge=judge):
            doc = item[0]
            pred = answers[doc["id"]]["answer"]
            if not pred:
                return {"id": doc["id"], "answer": pred, "score": 0.0}
            prompt = JUDGE_RULES.format(question=official_prompt(doc), answer=str(doc["answer"]), pred=pred)
            text = chat(judge, [{"role": "system", "content": JUDGE_SYSTEM},
                                {"role": "user", "content": prompt}])
            m = re.search(r"\d+", text or "")
            return {"id": doc["id"], "answer": pred, "score": float(m.group()) if m else 0.0,
                    "failed": text is None}

        sc_path = os.path.join(args.out, "%s_%s_scores.jsonl" % (tag(args.model), tag(judge)))
        scores = load(sc_path)
        todo = [it for it in items
                if scores.get(it[0]["id"], {}).get("answer") != answers[it[0]["id"]]["answer"]
                or scores[it[0]["id"]].get("failed")]
        run(score, todo, sc_path, scores, args.workers, "judge " + judge)
        by = defaultdict(list)
        for doc, _ in items:
            by[doc["field"]].append(scores[doc["id"]]["score"])
            by["overall"].append(scores[doc["id"]]["score"])
        results[judge] = {k: 10.0 * sum(v) / len(v) for k, v in by.items()}

    # 3. score (0-100) per discipline and overall
    keys = [f for f in FIELDS if f in results[args.judges[0]]] + ["overall"]
    if len(args.judges) > 1:
        results["average"] = {k: sum(results[j][k] for j in args.judges) / len(args.judges) for k in keys}
    print("\nRRI answers: %d, MLLM answers: %d, scored items: %d"
          % (sum(a["source"] == "rri" for a in answers.values()),
             sum(a["source"] == "mllm" for a in answers.values()), len(items)))
    print("%-16s" % "" + "".join("%11s" % k for k in keys))
    for name, row in results.items():
        print("%-16s" % name + "".join("%11.2f" % row[k] for k in keys))
    with open(os.path.join(args.out, "%s_results.json" % tag(args.model)), "w") as f:
        json.dump({"model": args.model, "lang": args.lang, "n_items": len(items), "scores": results},
                  f, indent=1)


if __name__ == "__main__":
    main()
