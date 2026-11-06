# Render-Reading Instruments

Code for *Render-Reading Instruments for Scientific Visual Question Answering*.

## Setup

```bash
pip install -r requirements.txt
```

Download [SFE](https://huggingface.co/datasets/InternScience/SFE) (`20250611-en/`, `20250611-zh/`, `images/`).

## Run

```bash
python run_rri.py --data /path/to/SFE --lang en --pred mllm_predictions.jsonl --out outputs
```

`--pred` (optional) is a JSONL file with one `{"id": ..., "pred": ...}` per item. The readings are written to `outputs/rri_readings.jsonl` and the final answers to `outputs/predictions.jsonl`.
