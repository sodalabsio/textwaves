from __future__ import annotations
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

# Reuse class names and time-series utilities from the lexicon generator
from .generate_synthetic import (
    CLASSES,
    _daily_range,
    _simulate_bias_curve,
    _class_probs_from_bias,
    _sample_times_for_class,
)


def _have_openai_key() -> bool:
    return bool(os.getenv('OPENAI_API_KEY'))


def _openai_client():
    from openai import OpenAI  # imported lazily
    return OpenAI()


def _build_prompt(klass: str, n: int) -> List[Dict[str, str]]:
    system = (
        "You generate short, social-media style political posts for a synthetic dataset. "
        "Write in plain English, neutral register for mainstream, more deliberative for deep democracy, "
        "and more outraged/populist for anti_democratic. Keep each post concise and self-contained. "
        "Avoid slurs, personal data, or calls to violence."
    )

    style = {
        'deep_democracy': (
            "Voice: civic, deliberative, pro-institutions. Emphasize rule of law, pluralism, transparency, rights."
        ),
        'mainstream_democracy': (
            "Voice: pragmatic, policy-focused, problem-solving. Emphasize budgets, public services, stability, cross-party work."
        ),
        'anti_democratic': (
            "Voice: populist, distrustful of institutions and media, calling things rigged or corrupt. No hate speech."
        ),
    }[klass]

    user = (
        f"Produce {n} distinct posts labelled class={klass}.\n"
        "Requirements:\n"
        "- Each post is 1 sentence; 10-28 words; feels like a social post.\n"
        "- Vary wording to increase lexical overlap across classes (reuse nouns like courts, media, elections, borders).\n"
        "- Do NOT include quotes, numbering, explanations, or extra text.\n"
        "- Output strictly as a JSON array named 'posts' with objects {\"text\": <string>}.\n\n"
        f"Style: {style}"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _parse_posts(content: str) -> List[str]:
    # Expect a JSON object {"posts": [{"text": ...}, ...]}
    try:
        obj = json.loads(content)
        if isinstance(obj, dict) and 'posts' in obj and isinstance(obj['posts'], list):
            posts = []
            for item in obj['posts']:
                if isinstance(item, dict) and 'text' in item and isinstance(item['text'], str):
                    t = item['text'].strip()
                    if t:
                        posts.append(t)
            return posts
        # Sometimes model returns a list directly
        if isinstance(obj, list):
            posts = []
            for item in obj:
                if isinstance(item, dict) and 'text' in item and isinstance(item['text'], str):
                    t = item['text'].strip()
                    if t:
                        posts.append(t)
                elif isinstance(item, str) and item.strip():
                    posts.append(item.strip())
            return posts
    except Exception:
        pass

    # Fallback: try to extract lines that look like JSONL or quoted strings
    lines = [ln.strip().strip('-').strip() for ln in content.splitlines() if ln.strip()]
    out: List[str] = []
    for ln in lines:
        try:
            obj = json.loads(ln)
            if isinstance(obj, dict) and isinstance(obj.get('text'), str):
                t = obj['text'].strip()
                if t:
                    out.append(t)
        except Exception:
            # Heuristic: quoted string
            if ln.startswith('"') and ln.endswith('"') and len(ln) > 2:
                out.append(ln[1:-1])
            elif ln and not ln.lower().startswith(('posts', 'class=')):
                out.append(ln)
    return out


def _generate_posts_for_class(klass: str, total: int, model: str, temperature: float = 0.8,
                              max_per_call: int = 24) -> List[str]:
    """Call OpenAI in batches to get `total` posts for a class."""
    client = _openai_client()
    posts: List[str] = []
    while len(posts) < total:
        want = min(max_per_call, total - len(posts))
        messages = _build_prompt(klass, want)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or ""
            batch = _parse_posts(content)
            # Basic cleanup and filters
            clean: List[str] = []
            for t in batch:
                t = t.strip().replace('\n', ' ')
                # remove surrounding quotes if any
                if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
                    t = t[1:-1].strip()
                # length filter
                wc = len(t.split())
                if 6 <= wc <= 32 and not t.lower().startswith(('class=', 'label=')):
                    clean.append(t)
            posts.extend(clean)
        except Exception as e:
            # On transient errors, just retry with a smaller batch
            if max_per_call > 5:
                max_per_call = max(5, max_per_call // 2)
            else:
                raise e
    # Deduplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for t in posts:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq[:total]


def generate_llm(
    n_per_class: int = 100,
    start: str = '2022-01-01',
    end: str = '2024-12-31',
    add_label_noise: bool = False,
    noise_rate: float = 0.05,
    seed: int = 42,
    balance_classes: bool = True,
    enable_timeseries: bool = True,
    sharpness: float = 2.5,
    season_period: int = 45,
    season_amp: float = 0.35,
    openai_model: str = 'gpt-4.1-nano-2025-04-14',
    temperature: float = 0.8,
) -> pd.DataFrame:
    """Generate a synthetic corpus using an LLM to author short posts.

    Falls back to the lexicon generator if no OPENAI_API_KEY is set.
    """
    # Fallback if key missing
    if not _have_openai_key():
        from .generate_synthetic import generate as generate_lex
        print('[llm] No OPENAI_API_KEY found; falling back to lexicon generator')
        return generate_lex(n_per_class=n_per_class, start=start, end=end,
                            add_label_noise=add_label_noise, noise_rate=noise_rate, seed=seed,
                            balance_classes=balance_classes, enable_timeseries=enable_timeseries,
                            sharpness=sharpness, season_period=season_period, season_amp=season_amp)

    rng = np.random.default_rng(seed)
    start_dt, end_dt = pd.to_datetime(start), pd.to_datetime(end)

    # Time series setup (same logic as lexicon version)
    print(' --> Setting up synthetic timeseries ...')
    days = _daily_range(start_dt.to_pydatetime(), end_dt.to_pydatetime())
    if enable_timeseries and len(days) >= 2:
        print('   --> Simulating bias curve ...')
        bias = _simulate_bias_curve(rng, len(days), season_period=season_period, season_amp=season_amp)
        probs_by_day = np.vstack([_class_probs_from_bias(bias[i], sharpness=sharpness) for i in range(len(days))])
    else:
        print('   --> No bias curve; using uniform class probabilities.')
        probs_by_day = np.ones((len(days), len(CLASSES))) / len(CLASSES)

    records = []

    if balance_classes:
        print(' --> Generating texts using balanced classes ...')
        # Pre-generate texts per class using the LLM
        per_class_texts: Dict[str, List[str]] = {}
        for klass in CLASSES:
            per_class_texts[klass] = _generate_posts_for_class(
                klass, n_per_class, model=openai_model, temperature=temperature
            )
        for k, klass in enumerate(CLASSES):
            day_probs = probs_by_day[:, k]
            day_probs = day_probs / day_probs.sum() if day_probs.sum() > 0 else np.ones_like(day_probs) / len(day_probs)
            ts_list = _sample_times_for_class(rng, days, day_probs, n_per_class)
            for i, ts in enumerate(ts_list):
                text = per_class_texts[klass][i % len(per_class_texts[klass])]
                records.append({'text': text, 'label_name': klass, 'timestamp': ts})
    else:
        print(' --> Generating texts using unbalanced classes ...')
        # Approximate total, then draw class each time and request posts lazily per small batches
        total_docs = 3 * n_per_class
        # Maintain small buffers so we don't over-call the API
        buffers: Dict[str, List[str]] = {c: [] for c in CLASSES}
        for _ in range(total_docs):
            i = int(rng.integers(0, len(days)))
            p = probs_by_day[i]
            k = int(rng.choice(np.arange(len(CLASSES)), p=p))
            klass = CLASSES[k]
            if not buffers[klass]:
                buffers[klass] = _generate_posts_for_class(klass, 12, model=openai_model, temperature=temperature)
            text = buffers[klass].pop(0)
            ts = days[i] + timedelta(seconds=int(rng.integers(0, 24*3600)))
            records.append({'text': text, 'label_name': klass, 'timestamp': ts})

    df = pd.DataFrame.from_records(records)
    label_map = {name: i for i, name in enumerate(CLASSES)}
    df['label'] = df['label_name'].map(label_map).astype(int)

    if add_label_noise and noise_rate > 0:
        print(f' --> Adding label noise ({noise_rate*100:.1f}%) ...')
        m = len(df)
        n_flip = int(m * noise_rate)
        idx = rng.choice(df.index, size=n_flip, replace=False)
        for j in idx:
            true = df.at[j, 'label']
            choices = [x for x in sorted(label_map.values()) if x != true]
            df.at[j, 'label'] = int(rng.choice(choices))

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def save_csv(df: pd.DataFrame, path: str = 'synthetic_data/synthetic_llm.csv') -> None:
    print(' --> Saving synthetic data to CSV ...')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
