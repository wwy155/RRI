"""Render-Reading Instruments on SFE.

    python run_rri.py --data /path/to/SFE --lang en --pred mllm_predictions.jsonl --out outputs
"""
import argparse
import ast
import json
import os

import pyarrow.parquet as pq

import rri


def exact(answer, gold):
    gold = ast.literal_eval(gold) if isinstance(gold, str) and gold[:1] in "[{" else gold
    if isinstance(gold, dict):
        return json.loads(answer) == {k: int(v) for k, v in gold.items()}
    gold = gold[0] if isinstance(gold, (list, tuple)) else gold
    return answer == str(gold).strip("() ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="SFE root with 20250611-<lang>/ and images/")
    ap.add_argument("--lang", default="en", choices=["en", "zh"])
    ap.add_argument("--pred", help="MLLM predictions, JSONL with 'id' and 'pred'")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    rows = pq.read_table(os.path.join(args.data, "20250611-%s" % args.lang,
                                      "test.parquet")).to_pylist()
    os.makedirs(args.out, exist_ok=True)

    readings = {}
    with open(os.path.join(args.out, "rri_readings.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            paths = [os.path.join(args.data, "images", os.path.basename(str(p).replace("\\", "/")))
                     for p in (r["images"] or [])]
            out = rri.read(r, paths)
            if out is not None:
                readings[r["id"]] = out
                f.write(json.dumps({"id": r["id"], **out}, ensure_ascii=False) + "\n")

    if args.pred:
        preds = {}
        with open(args.pred, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    o = json.loads(line)
                    preds[o["id"]] = o.get("pred")
        with open(os.path.join(args.out, "predictions.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                ans = readings.get(r["id"], {}).get("answer")
                f.write(json.dumps({"id": r["id"], "pred": preds.get(r["id"]) if ans is None
                                    else ans}, ensure_ascii=False) + "\n")

    committed = [i for i, o in readings.items() if o["answer"] is not None]
    gold = {r["id"]: r["answer"] for r in rows}
    n_exact = sum(exact(readings[i]["answer"], gold[i]) for i in committed)
    print("instrumented items: %d, committed: %d, exact readings: %d"
          % (len(readings), len(committed), n_exact))


if __name__ == "__main__":
    main()
