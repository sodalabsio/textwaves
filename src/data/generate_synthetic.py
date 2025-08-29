from __future__ import annotations
import os
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Design goals of this improved generator
# - Produce more natural short texts using clause-level templates and agreement
# - Maximise lexical overlap across classes to avoid easy TF-IDF wins
# - Provide a controllable time series so class prevalence varies over time
# - Remain dependency-light (numpy, pandas only)
# -----------------------------------------------------------------------------

# Shared nouns (appear in all classes) — anchor vocabulary overlap
SHARED_NOUNS = [
    'courts', 'media', 'elections', 'borders', 'schools', 'communities', 'workers',
    'families', 'economy', 'healthcare', 'police', 'parliament', 'constitution',
    'rights', 'security', 'freedoms', 'votes', 'taxes', 'public services',
    'local councils', 'institutions', 'process', 'law', 'budget', 'candidates'
]

# Class frames mostly differ in adjectives/verbs/claims, not in nouns
FRAME_LEX = {
    'deep_democracy': {
        'adjectives': ['independent', 'transparent', 'plural', 'inclusive', 'accountable',
                       'free', 'fair', 'deliberative', 'open', 'constitutional'],
        'verbs': ['protect', 'strengthen', 'uphold', 'safeguard', 'expand', 'defend',
                  'renew', 'restore', 'deepen'],
        'goals': ['protect rights', 'build trust', 'support participation',
                  'keep power accountable', 'ensure fair representation'],
        'modals': ['must', 'should', 'will', 'need to']
    },
    'mainstream_democracy': {
        'adjectives': ['balanced', 'practical', 'responsible', 'efficient', 'lawful',
                       'stable', 'sensible', 'measured', 'workable'],
        'verbs': ['deliver', 'improve', 'manage', 'review', 'modernise', 'consult',
                  'fund', 'balance', 'maintain'],
        'goals': ['deliver services', 'keep budgets in line', 'solve problems',
                  'work across parties', 'support local priorities'],
        'modals': ['will', 'aim to', 'plan to', 'should']
    },
    'anti_democratic': {
        'adjectives': ['rigged', 'corrupt', 'broken', 'fake', 'weak', 'globalist',
                       'traitorous', 'crooked', 'captured'],
        'verbs': ['expose', 'fight', 'crush', 'defy', 'punish', 'end', 'smash',
                  'shut down', 'take back'],
        'goals': ['take back control', 'end the scam', 'put the people first',
                  'drain the swamp', 'stop the betrayal'],
        'modals': ['will', 'must', 'won\'t', 'have to']
    }
}

CLASSES = list(FRAME_LEX.keys())

# Discourse scaffolds; each element is a callable that returns a sentence string
# given RNG and a class key. All patterns use shared nouns to boost lexical overlap.

# Basic subject choices produce agreement info
SUBJECTS = [
    ('we', 'pl'),
    ('our community', 'sg'),
    ('this government', 'sg'),
    ('people', 'pl'),
    ('I', 'sg1'),
    ('they', 'pl')
]

# Lightweight verb agreement for present tense
# Only handles a few person/number cases for our chosen subjects
_DEF_S3 = set(['he', 'she', 'it', 'this government', 'our community'])

def _conj_present(verb: str, subject: str) -> str:
    # Compound verbs like 'shut down' need agreement on first token
    parts = verb.split(' ')
    head = parts[0]
    tail = ' '.join(parts[1:])
    s = subject.lower()
    is_s3 = (s in _DEF_S3) or (s.endswith('y') and ' ' not in s)
    if head in ['will', 'must', 'should', 'won\'t', 'have', 'need', 'plan', 'aim']:
        # Modal/aux keep base verb
        return verb
    if s in ['we', 'people', 'they']:
        v = head
    elif s == 'I' or s == 'i' or s == 'sg1':
        v = head
    else:
        # naive 3rd singular
        if head.endswith('y') and len(head) > 1 and head[-2] not in 'aeiou':
            v = head[:-1] + 'ies'
        elif head.endswith('sh') or head.endswith('ch') or head.endswith('x') or head.endswith('s') or head.endswith('z'):
            v = head + 'es'
        else:
            v = head + 's'
    return (v + (' ' + tail if tail else ''))


def _np(rng: np.random.Generator, klass: str) -> str:
    # Noun phrase with class-influenced adjective but shared noun
    noun = rng.choice(SHARED_NOUNS)
    # With some probability, borrow adjective from another class to increase overlap
    if rng.random() < 0.2:  # overlap rate
        other = rng.choice([c for c in CLASSES if c != klass])
        adj = rng.choice(FRAME_LEX[other]['adjectives'])
    else:
        adj = rng.choice(FRAME_LEX[klass]['adjectives'])
    # Sometimes no adjective
    if rng.random() < 0.25:
        return f"{noun}"
    # Handle a/an
    art = 'an' if adj[0].lower() in 'aeiou' else 'a'
    return f"{art} {adj} {noun}"


def _vp(rng: np.random.Generator, klass: str, subject: str) -> str:
    modal = rng.choice(FRAME_LEX[klass]['modals']) if rng.random() < 0.8 else ''
    verb = rng.choice(FRAME_LEX[klass]['verbs'])
    v = _conj_present(verb, subject)
    if modal:
        return f"{modal} {verb}"
    return v


def _goal(rng: np.random.Generator, klass: str) -> str:
    return rng.choice(FRAME_LEX[klass]['goals'])


def _clause_claim(rng: np.random.Generator, klass: str) -> str:
    subject, _num = rng.choice(SUBJECTS)
    vp = _vp(rng, klass, subject)
    obj = _np(rng, klass)
    if rng.random() < 0.6:
        return f"{subject.capitalize()} {vp} the {obj}."
    else:
        return f"{subject.capitalize()} {vp} {obj}."


def _clause_goal(rng: np.random.Generator, klass: str) -> str:
    subject, _ = rng.choice(SUBJECTS)
    vp = _vp(rng, klass, subject)
    goal = _goal(rng, klass)
    return f"{subject.capitalize()} {vp} to {goal}."


def _clause_because(rng: np.random.Generator, klass: str) -> str:
    subject, _ = rng.choice(SUBJECTS)
    vp = _vp(rng, klass, subject)
    reason_np = _np(rng, klass)
    obj = _np(rng, klass)
    if rng.random() < 0.5:
        return f"Because the {reason_np} matters, {subject} {vp} the {obj}."
    else:
        return f"Because the {reason_np} is at stake, {subject} {vp} {obj}."


def _clause_time_to(rng: np.random.Generator, klass: str) -> str:
    v = rng.choice(FRAME_LEX[klass]['verbs'])
    np1 = _np(rng, klass)
    np2 = _np(rng, klass)
    conj = rng.choice(['and', 'while we'])
    return f"It\'s time to {v} the {np1} {conj} {v} the {np2}."


def _clause_no_more(rng: np.random.Generator, klass: str) -> str:
    np1 = _np(rng, klass)
    subject, _ = rng.choice(SUBJECTS)
    vp = _vp(rng, klass, subject)
    np2 = _np(rng, klass)
    return f"No more {np1}: {subject.capitalize()} {vp} the {np2}."


def _clause_question_then(rng: np.random.Generator, klass: str) -> str:
    np1 = _np(rng, klass)
    subject, _ = rng.choice(SUBJECTS)
    vp = _vp(rng, klass, subject)
    np2 = _np(rng, klass)
    return f"Who benefits when the {np1} is ignored? {subject.capitalize()} {vp} the {np2}."


SENTENCE_BUILDERS = [
    _clause_claim,
    _clause_goal,
    _clause_because,
    _clause_time_to,
    _clause_no_more,
    _clause_question_then,
]

# Light social-media style noise for realism
EMOJIS = ['🙂', '🙌', '⚖️', '🗳️', '🛡️', '🚫', '💬']
HASHTAGS = ['#democracy', '#community', '#rights', '#security', '#accountability', '#elections', '#media']
URLS = ['http://example.com', 'https://news.example.org/article']


def _add_style_noise(rng: np.random.Generator, text: str, klass: str) -> str:
    # punctuation emphasis
    if rng.random() < 0.25:
        text = text.rstrip('.') + rng.choice(['!', '!!', '!?'])
    # occasional hashtag or emoji
    tail_bits = []
    if rng.random() < 0.3:
        tail_bits.append(rng.choice(HASHTAGS))
    if rng.random() < 0.15:
        tail_bits.append(rng.choice(EMOJIS))
    if rng.random() < 0.08:
        tail_bits.append(rng.choice(URLS))
    if tail_bits:
        text = text + ' ' + ' '.join(tail_bits)
    # tiny typo chance: swap a vowel in a random word
    if rng.random() < 0.05:
        words = text.split(' ')
        i = int(rng.integers(0, len(words)))
        w = words[i]
        vowels = 'aeiou'
        for _ in range(3):
            pos = int(rng.integers(0, len(w))) if w else 0
            if pos < len(w) and w[pos].lower() in vowels:
                alt = rng.choice([v for v in vowels if v != w[pos].lower()])
                w = w[:pos] + (alt.upper() if w[pos].isupper() else alt) + w[pos+1:]
                break
        words[i] = w
        text = ' '.join(words)
    return text


# ----------------------- Time-series utilities --------------------------------

def _daily_range(start: datetime, end: datetime) -> List[datetime]:
    days = []
    cur = datetime(start.year, start.month, start.day)
    last = datetime(end.year, end.month, end.day)
    while cur <= last:
        days.append(cur)
        cur += timedelta(days=1)
    return days


def _simulate_bias_curve(rng: np.random.Generator, n_days: int, rho: float = 0.95,
                         noise_sd: float = 0.15, season_amp: float = 0.4, season_period: int = 30,
                         shocks: Optional[Dict[int, float]] = None) -> np.ndarray:
    """Simulate a latent ideology bias b_t in [-2, 2] approximately.
    Anchors: -1 deep_democracy, 0 mainstream_democracy, +1 anti_democratic.
    """
    b = np.zeros(n_days, dtype=float)
    b[0] = float(rng.normal(0.0, 0.5))
    for t in range(1, n_days):
        b[t] = rho * b[t-1] + float(rng.normal(0.0, noise_sd))
    # Add seasonality
    tgrid = np.arange(n_days)
    b += season_amp * np.sin(2 * np.pi * tgrid / max(2, season_period))
    # Add shocks if provided: dict of day_index -> delta
    if shocks:
        for idx, delta in shocks.items():
            if 0 <= idx < n_days:
                b[idx:] += float(delta)  # persistent shift
    # Clip
    b = np.clip(b, -2.0, 2.0)
    return b


def _class_probs_from_bias(bias: float, sharpness: float = 2.5) -> np.ndarray:
    """Map scalar bias to class probs via softmax on squared distance to anchors.
    Anchors at -1, 0, +1 for the three classes respectively.
    """
    anchors = np.array([-1.0, 0.0, 1.0])
    # Higher when closer to anchor
    scores = -sharpness * (anchors - bias) ** 2
    # Stabilise with small floor so no class goes to zero
    scores = scores - np.max(scores)
    exps = np.exp(scores)
    probs = exps / np.sum(exps)
    # Mix with a small uniform component
    eps = 0.05
    probs = (1 - eps) * probs + eps * (np.ones_like(probs) / len(probs))
    return probs


def _sample_times_for_class(rng: np.random.Generator, days: List[datetime],
                            day_probs: np.ndarray, n: int) -> List[datetime]:
    # Choose days according to probabilities, then choose a random intra-day time
    idxs = rng.choice(np.arange(len(days)), size=n, replace=True, p=day_probs)
    ts_list = []
    for i in idxs:
        day = days[i]
        seconds = int(rng.integers(0, 24*3600))
        ts_list.append(day + timedelta(seconds=seconds))
    return ts_list


# -------------------------- Text generation -----------------------------------

def _make_sentence(rng: np.random.Generator, klass: str) -> str:
    builder = rng.choice(SENTENCE_BUILDERS)
    sent = builder(rng, klass)
    # Capitalise and ensure end punctuation
    if not sent.endswith(('.', '!', '?')):
        sent += '.'
    return sent


def _make_text(rng: np.random.Generator, klass: str) -> str:
    # 1–3 sentences, with a mild chance of style noise at the end
    n_sent = int(rng.integers(1, 4))
    sents = [_make_sentence(rng, klass) for _ in range(n_sent)]
    text = ' '.join(sents)
    text = _add_style_noise(rng, text, klass)
    return text


def _random_date(rng: np.random.Generator, start: datetime, end: datetime) -> datetime:
    delta = end - start
    seconds = rng.integers(0, int(delta.total_seconds()) + 1)
    return start + timedelta(seconds=int(seconds))


def generate(
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
) -> pd.DataFrame:
    """Generate a synthetic corpus.

    Parameters
    - n_per_class: number of samples per class (if balance_classes=True) else approximate per class
    - start, end: date range (inclusive of days)
    - add_label_noise: if True, flip a fraction of labels at random
    - noise_rate: fraction of labels to flip
    - seed: RNG seed
    - balance_classes: if True, produce exactly n_per_class for each class
    - enable_timeseries: if True, allocate timestamps using a time-varying class prevalence
    - sharpness: how peaky class probabilities are around bias anchor points
    - season_period, season_amp: seasonality for the latent bias
    """
    rng = np.random.default_rng(seed)
    start_dt, end_dt = pd.to_datetime(start), pd.to_datetime(end)

    # Time series setup
    print(' --> Setting up synthetic timeseries ...')

    days = _daily_range(start_dt.to_pydatetime(), end_dt.to_pydatetime())
    if enable_timeseries and len(days) >= 2:
        print('   --> Simulating bias curve ...')

        bias = _simulate_bias_curve(rng, len(days), season_period=season_period, season_amp=season_amp)
        probs_by_day = np.vstack([_class_probs_from_bias(bias[i], sharpness=sharpness) for i in range(len(days))])
        # probs_by_day[:, 0] -> deep, 1 -> main, 2 -> anti
    else:
        print('   --> No bias curve; using uniform class probabilities.')

        probs_by_day = np.ones((len(days), len(CLASSES))) / len(CLASSES)

    records = []

    if balance_classes:
        print(' --> Generating texts using balanced classes ...')
        # For each class, sample days weighted by its prevalence curve, then sample times
        for k, klass in enumerate(CLASSES):
            day_probs = probs_by_day[:, k]
            # Normalise
            if day_probs.sum() <= 0:
                day_probs = np.ones_like(day_probs) / len(day_probs)
            else:
                day_probs = day_probs / day_probs.sum()
            ts_list = _sample_times_for_class(rng, days, day_probs, n_per_class)
            for ts in ts_list:
                text = _make_text(rng, klass)
                records.append({'text': text, 'label_name': klass, 'timestamp': ts})
    else:
        print(' --> Generating texts using unbalanced classes ...')
        # Draw documents day-by-day with multinomial class draws; total approx 3*n_per_class
        total_docs = 3 * n_per_class
        for _ in range(total_docs):
            # pick a day uniformly, then class by that day's probs
            i = int(rng.integers(0, len(days)))
            p = probs_by_day[i]
            k = int(rng.choice(np.arange(len(CLASSES)), p=p))
            klass = CLASSES[k]
            ts = days[i] + timedelta(seconds=int(rng.integers(0, 24*3600)))
            text = _make_text(rng, klass)
            records.append({'text': text, 'label_name': klass, 'timestamp': ts})

    df = pd.DataFrame.from_records(records)
    # Stable label coding in class order
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

    # Shuffle for downstream tasks
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def save_csv(df: pd.DataFrame, path: str = 'synthetic_data/synthetic.csv') -> None:
    print(' --> Saving synthetic data to CSV ...')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
