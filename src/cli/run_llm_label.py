import os
import argparse
import json
import asyncio
import random
import numpy as np
from dotenv import load_dotenv
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from openai import AsyncOpenAI, APIError, RateLimitError

from ..config import load_config
from ..utils import ensure_dirs
from ..data.load import load_or_generate, load_or_generate_llm

def build_prompt(text: str, class_names: list[str], style: str, df: pd.DataFrame) -> str:
    base = (
        f"Your task is to label the following text as expressing one of the {class_names}. "
        f"Provide only the class name and a 10 word or less 'thought' justifying the choice of label. "
        f"Provide your response in JSON format, only as "
        f'{{\"label\": <label>, \"thought\": <thought>}}.\n\n'
    )

    if style == "zero":
        return base + f"Text: {text}"

    elif style == "extended":
        # Few-shot examples: up to 3 per class
        examples = []
        for cls in class_names:
            samples = df[df["label"] == cls]["text"].sample(min(3, len(df[df["label"] == cls])), random_state=42)
            for s in samples:
                examples.append(f'Example: {{\"text\": \"{s}\", \"label\": \"{cls}\"}}')

        examples_str = "\n".join(examples)
        return base + f"Here are some examples:\n{examples_str}\n\nNow classify this new text:\nText: {text}"

    else:
        raise ValueError(f"Unknown prompt style: {style}")

async def call_with_backoff(client, model, prompt, max_retries=5):
    delay = 1
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return resp.choices[0].message.content.strip()
        except (RateLimitError, APIError) as e:
            wait = delay + random.uniform(0, 0.5)
            print(f"Rate limit/API error (attempt {attempt+1}), retrying in {wait:.1f}s...")
            await asyncio.sleep(wait)
            delay *= 2
        except Exception as e:
            print("Unexpected error:", e)
            return None
    return None

async def process_text(semaphore, client, model, text, class_names, style, df):
    async with semaphore:
        prompt = build_prompt(text, class_names, style, df)
        content = await call_with_backoff(client, model, prompt)
        if not content:
            return {"label": "ERROR", "thought": ""}
        try:
            parsed = json.loads(content)
            return {"label": str(parsed.get("label", "")), "thought": parsed.get("thought", "")}
        except Exception:
            return {"label": "PARSE_ERROR", "thought": content[:50]}

async def run_labelling(df, class_names, model, style, concurrency=5):
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        process_text(semaphore, client, model, text, class_names, style, df)
        for text in df["text"].tolist()
    ]
    return await asyncio.gather(*tasks)

def main():
    parser = argparse.ArgumentParser(description="Run LLM-based labelling on synthetic tweets (parallelised, few-shot extended)")
    parser.add_argument("--data-source", choices=["lexicon", "llm"], default="lexicon")
    parser.add_argument("--prompt-style", choices=["zero", "extended"], default=None,
                        help="Prompting style: zero-shot or extended few-shot. Defaults to config.yaml.")
    parser.add_argument("--regen-data", action="store_true", help="Regenerate synthetic dataset before labelling.")
    parser.add_argument("--regen-labels", action="store_true", help="Force re-run of LLM labelling even if cached labels exist.")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of concurrent API calls.")
    parser.add_argument("--llm-model", type=str, default="gpt-4.1-nano-2025-04-14", help="LLM model to use for labelling.")
    args = parser.parse_args()

    load_dotenv()
    cfg = load_config()
    ensure_dirs()

    llm_cfg = cfg.get("llm_label", {})
    model = args.llm_model or llm_cfg.get("model", "gpt-4.1-nano-2025-04-14")
    style = args.prompt_style or llm_cfg.get("prompt_style", "zero")

    # Load dataset
    if args.data_source == "llm":
        df = load_or_generate_llm(
            n_per_class=cfg["data"]["n_per_class"],
            start=cfg["data"]["start"],
            end=cfg["data"]["end"],
            add_label_noise=cfg["data"]["add_label_noise"],
            noise_rate=cfg["data"]["noise_rate"],
            seed=cfg["random_seed"],
            regen=args.regen_data,
            path="data/synthetic_llm.csv",
            openai_model=model,
        )
    else:
        df = load_or_generate(
            n_per_class=cfg["data"]["n_per_class"],
            start=cfg["data"]["start"],
            end=cfg["data"]["end"],
            add_label_noise=cfg["data"]["add_label_noise"],
            noise_rate=cfg["data"]["noise_rate"],
            seed=cfg["random_seed"],
            regen=args.regen_data,
        )

    class_names = sorted(df["label"].unique().tolist())

    # Paths for caching
    labels_path = os.path.join("outputs", "tables", f"llm_labels_{args.data_source}_{style}_{model}.csv")
    metrics_path = os.path.join("outputs", "tables", f"metrics_summary_{args.data_source}.csv")
    os.makedirs(os.path.dirname(labels_path), exist_ok=True)

    if os.path.exists(labels_path) and not args.regen_labels:
        print(f" --› Using cached LLM labels from {labels_path}")
        df = pd.read_csv(labels_path)
    else:
        print(f" --› Running parallel LLM labelling with {model} ({style} prompt, concurrency={args.concurrency}) ...")
        results = asyncio.run(run_labelling(df, class_names, model, style, concurrency=args.concurrency))
        df["llm_label"] = [r["label"] for r in results]
        df["llm_thought"] = [r["thought"] for r in results]
        # Ensure string type before saving
        df["llm_label"] = df["llm_label"].astype(str)
        df["label"] = df["label"].astype(str)
        df.to_csv(labels_path, index=False)
        print(f" --› Saved LLM labels to {labels_path}")

    # Ensure consistent types for metrics
    df["label"] = df["label"].astype(str)
    df["llm_label"] = df["llm_label"].astype(str)

    # Compute metrics
    acc = accuracy_score(df["label"], df["llm_label"])
    f1 = f1_score(df["label"], df["llm_label"], average="weighted")

    # Integrate into summary CSV
    row = {
        "Method": f"llm-{style}",
        "Data": args.data_source,
        "TrainFrac": np.nan,
        "ARI": np.nan,
        "NMI": np.nan,
        "Silhouette": np.nan,
        "LR_Accuracy": np.nan,
        "LR_F1_weighted": np.nan,
        "LLM_Accuracy": acc,
        "LLM_F1_weighted": f1,
        "LLM_model_name": model
    }

    if os.path.exists(metrics_path):
        existing = pd.read_csv(metrics_path)
        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        header = ["Method", "Data", "TrainFrac", "ARI", "NMI", "Silhouette",
                  "LR_Accuracy", "LR_F1_weighted", "LLM_Accuracy", "LLM_F1_weighted", "LLM_model_name"]
        updated = pd.DataFrame([row], columns=header)

    updated.to_csv(metrics_path, index=False)

    print(f" --› Saved integrated metrics to {metrics_path}")
    print(f"LLM Accuracy={acc:.3f}  F1_w={f1:.3f}")

if __name__ == "__main__":
    main()
