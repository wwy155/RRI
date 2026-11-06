# Render-Reading Instruments

Code for *Render-Reading Instruments for Scientific Visual Question Answering*.

## Setup

```bash
pip install -r requirements.txt
```

Download [SFE](https://huggingface.co/datasets/InternScience/SFE) (`20250611-en/`, `20250611-zh/`, `images/`).

## Run

The MLLM and the LLM judges are called through an OpenAI-compatible API.

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.openai.com/v1
python run_rri.py --data /path/to/SFE --model <mllm>
```

Answers, judge scores and per-discipline results are written to `outputs/`.
