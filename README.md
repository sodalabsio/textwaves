# 🌊 TextWaves: simple NLP analysis on a tiny synthetic dataset

🖊️ [Simon D. Angus](https://research.monash.edu/en/persons/simon-angus), authored with AI-augmentation via SoDa Laboratory's [Assistant AI](assistant.sodalabs.io) tooling.

[SoDa Laboratories](sodalabs.io), Monash Business Schoool, Monash University

Please fork to begin your project -- if using these tools for academic work, please cite as below.

## TL;DR
Lean, hands-on repo to demonstrate three representation “waves” for text-as-data in social science, plus optional LLM-labelling, accompanying workshops/talks at U Greenwich, LDN, Sep 1–2 2025 — _Methods in Social Change and Inertia Workshops_.

- Wave 1: TF‑IDF

- Wave 2: Average GloVe embeddings

- Wave 3: Transformer/LLM embeddings (OpenAI or SBERT fallback)

- Optional: LLM labelling (zero-shot or few-shot) to directly assign labels to texts

We compare methods on a tiny, labelled synthetic corpus (3 classes × ~100 each). You can:

- Cluster and assess separation/quality in 2D.
- Train a simple logistic regression baseline (temporal train/test split: train on “history”, predict the “future”).
- Run an LLM labelling pass and compare its accuracy/F1 to classic ML.

All runs write to a unified results CSV you can plot in one line.

## 🤔 Assumptions
You’re comfortable running a few terminal commands, have basic Python familiarity, and want a clear, minimal path to try NLP/AI on texts for social science questions.

## 🚀 Quick start (fastest path)

1) Install Python 3.10+

2) Create/activate a virtual environment and install deps

   ```bash
   python -m venv myvenv
   source myvenv/bin/activate
   pip install -r requirements.txt
   ```

3) (Optional) Add an OpenAI API key for OpenAI embeddings, LLM data synthesis and LLM labelling

   - Copy `.env.example` to `.env` and set `OPENAI_API_KEY=...`
   - No key? The pipeline still runs end-to-end:
     - OpenAI embeddings automatically fall back to SBERT (`all-MiniLM-L6-v2`).
     - LLM-based data synthesis falls back to the lexicon generator.

4) Run everything in one go, writing a unified metrics file

   Classic ML only (TF‑IDF, GloVe, SBERT/OpenAI embeddings), lexicon synthetic data:

   ```bash
   python -m src.cli.build_all
   ```

   Also include LLM labelling (zero-shot by default):

   ```bash
   python -m src.cli.build_all --include-llm
   ```

   Few-shot LLM labelling instead (uses in-repo examples):

   ```bash
   python -m src.cli.build_all --include-llm --llm-prompt-style extended
   ```

   Any of the above, but generating/using llm created synthetic data instead of lexicon:

   ```bash
   python -m src.cli.build_all --data-source llm ...
  ```

5) Visualise the results

   This reads the unified CSV and creates a comparison bar chart of weighted F1 across LR (by representation) and LLM:

   ```bash
   python -m src.viz.results --data-source lexicon
   ```

   Use `--data-source llm` if you built the LLM-authored synthetic dataset.

6) (Optional) Visuals on the fly

   You can add these flags to any single-method run to save figures immediately:

   ```bash
   python -m src.cli.run_pipeline --method tfidf --data-source lexicon \
       --plot-scatter --plot-timeseries --plot-future
   ```

That’s it. You now have:

- A single CSV at `outputs/tables/metrics_summary_{lexicon|llm|byod}.csv`
- A figure at `outputs/figures/f1_comparison_{lexicon|llm}.png`
- Optional figures (if you used the flags): scatter/timeseries/future under `outputs/figures/`
- A cache of run products you can plot later without recomputing under `outputs/cache/`

---

## ⚙ Pipeline approach (what’s happening under the hood)
The pipeline has five components you can mix and match:

1. **Synthesise — tiny dataset (default 300 texts, 3 topics) using a lexicon or an LLM. Texts are timestamped so class prevalence ebbs/flows over time.

2. **Vectorise** — TF‑IDF, GloVe, or transformer-based embeddings (OpenAI with SBERT fallback).

3. **Cluster and reduce** — cluster (KMeans) and reduce to 2D (UMAP/PCA) for intuition and visualisation.

4. **Train / predict** — logistic regression trained on the earliest slice of data (temporal split) and evaluated on future texts.

5. **Compare** — a unified results CSV with clustering metrics and classification metrics, plus optional LLM labelling metrics.

Together these mirror common text-as-data workflows for computational social science and are easy to adapt to your own data.

## Generate data explicitly (optional)
You can regenerate the synthetic data without running a full method:

Lexicon dataset (default):

```bash
python -m src.cli.run_pipeline --regen-data --data-source lexicon
```

LLM-authored dataset (requires key; defaults to `gpt-4.1-nano-2025-04-14`):

```bash
python -m src.cli.run_pipeline --regen-data --data-source llm
```

Each writes a CSV to `synthetic_data/` (or `data/` where noted) with columns `text,label_name,timestamp,label`.

## Run a single method by hand (optional)
Choose your representation and run end-to-end on either dataset source:

```bash
# TF‑IDF example on the lexicon dataset
python -m src.cli.run_pipeline --method tfidf --data-source lexicon

# SBERT/OpenAI embeddings (auto-falls back to SBERT if no key)
python -m src.cli.run_pipeline --method openai --data-source llm
```

Use `--reduce pca` to switch from UMAP, or `--clusterer kmeans` to be explicit. See all options:

```bash
python -m src.cli.run_pipeline --help
```

## Optional visualisations (from cache or on the fly)
You have two simple ways to get the nice visuals.

1) On the fly during a run (no extra commands):

```bash
python -m src.cli.run_pipeline --method tfidf --data-source lexicon \
    --plot-scatter --plot-timeseries --plot-future
```

2) Later, without recomputing, from cached outputs:

```bash
# After any prior run for a given (data_source, method)
# Uses outputs/cache/last_{data_source}_{method}.npz
python -m src.cli.plot_cached --what timeseries --data-source lexicon     # - no method required, just plots synthetic data timeseries
python -m src.cli.plot_cached --what scatter --data-source lexicon --method tfidf  # -- scatter of lowD vectors
python -m src.cli.plot_cached --what future --data-source lexicon --method tfidf   # -- future label predictions vs. groundtruth

# Or point to a specific cache (e.g., run_<hash>.npz)
python -m src.cli.plot_cached --what scatter --data-source llm --method openai \
    --cache outputs/cache/run_aaaaaaaaaaaaaaaa.npz
```

Generated figures go to `outputs/figures/` with sensible names.

## BYOD: Bring Your Own Data
You can plug in your own CSV with almost no friction.

Minimum you need is a `text` column. If you also provide `label` and/or `timestamp`, we’ll use them; otherwise we create a synthetic timestamp and, if no labels, put everything into a single class `unlabeled` so clustering and visuals still work.

Quick examples:

```bash
# Labeled data with custom column names
python -m src.cli.run_pipeline \
  --data-source byod \
  --byod-path my_texts.csv \
  --byod-text-col body \
  --byod-label-col stance \
  --byod-timestamp-col posted_at \
  --method tfidf \
  --plot-scatter --plot-timeseries

# Unlabeled data (clustering + visuals still work)
python -m src.cli.run_pipeline \
  --data-source byod \
  --byod-path my_texts.csv \
  --byod-text-col text \
  --method glove \
  --plot-scatter --plot-timeseries

# Later, plot from cache without recomputing
python -m src.cli.plot_cached --what scatter --data-source byod --method glove
```

Under the hood this uses `src/data/byod.py` to normalize your CSV to the standard schema: `text`, `label_name`, `label` (int-coded), `timestamp`.

## Outputs (unified and simple)
- Unified metrics CSV (always produced):
  - `outputs/tables/metrics_summary_{lexicon|llm|byod}.csv`
  - Columns include: representation and settings, clustering metrics (ARI, NMI, Silhouette), logistic regression metrics (Accuracy, weighted F1), and (if run) LLM labelling metrics.

- LLM labelling cache (if run):
  - `outputs/tables/llm_labels_{data_source}_{prompt_style}_{model}.csv`
  - Stores the LLM’s chosen label and short “thought” per text.

- Visualisations (optional, either on-the-fly or from cache):
  - `outputs/figures/scatter_{method}_{data}.png`
  - `outputs/figures/timeseries_{method}_{data}.png`
  - `outputs/figures/future_{method}_{data}.png`

- Cache of run products (enables instant plotting later, no recompute):
  - `outputs/cache/run_<hash>.npz` (contains coords2d, clabels, labels, label_names, timestamps, test_idx, y_pred, cutoff_ts)
  - Convenience link: `outputs/cache/last_{data_source}_{method}.npz`

## Inference train/test controls (temporal split)
Logistic regression is trained on the earliest fraction of the dataset (by timestamp) and evaluated on the future portion — a realistic setting where future distributions are not leaked into training.

- Default train fraction: `0.3` (30% history; 70% future).
- Change it with, e.g.:

```bash
python -m src.cli.run_pipeline --method tfidf --train-frac 0.8
```

## LLM labelling (zero-shot and few-shot)
You can ask an LLM to assign labels directly and compare that to classic ML baselines.

- Zero-shot (default):

```bash
python -m src.cli.run_llm_label --data-source lexicon --prompt-style zero
```

- Few-shot ("extended"): draws small in-repo examples per class as scaffolding

```bash
python -m src.cli.run_llm_label --data-source lexicon --prompt-style extended
```

All LLM-labelling runs write metrics into the same unified CSV used by classic ML, so your `src.viz.results` figure will include both.

Tips:
- Add `--regen-labels` to force relabelling even if a cache exists.
- Use `--concurrency` to control parallel API calls.
- Select labelling model with `--llm-model` (defaults to `gpt-4.1-nano-2025-04-14`).

## Dataset sources (lexicon vs LLM)
- Lexicon generator (default): short texts created from clause-level templates with shared nouns to keep lexical overlap across classes.

- LLM generator: short, social-media-like posts authored by an LLM. Defaults to `gpt-4.1-nano-2025-04-14`; override with `--llm-model`. If `OPENAI_API_KEY` is missing, it falls back to the lexicon generator.

Classes are the same in both datasets: `deep_democracy`, `mainstream_democracy`, `anti_democratic`.

## Configuration
Edit `configs/config.yaml` (sensible defaults apply if missing):

- Vectorizers
  - TF‑IDF params
  - GloVe model
  - OpenAI embedding model and batch size
  - SBERT fallback model (used automatically if no OpenAI key)

- Data
  - Counts, time span, optional label noise

- Reduce/Cluster
  - `reduce.method` (`umap` or `pca`), `reduce.n_components`, `reduce.n_neighbors` (UMAP)
  - `cluster.k` for KMeans

- LLM labelling
  - `llm_label.model` and `llm_label.prompt_style` (`zero` or `extended`)

Notes:
- GloVe is loaded via `gensim.downloader` (`glove-wiki-gigaword-50`) — requires internet on first use.
- SBERT fallback (`all-MiniLM-L6-v2`) downloads on first use.
- OpenAI models require an API key set in `.env` (or your shell env). Without a key, the pipeline degrades gracefully.

## Bring your own data (BYOD)
Once you’re comfortable, adapt this as a swiss‑army‑knife starter:

- Minimal expected columns (if you mirror the synthetic format): `text`, `label_name` (or `label`), and ideally a `timestamp` for temporal splits and time‑series visuals.
- Start by replacing the data loader in `src/data/load.py` to point at your CSV; or add a new loader.
- Keep the same CLI shape so you can still run `build_all` and produce unified results/plots without changing your workflow.

For convenience, the built-in BYOD loader (`src/data/byod.py`) already handles these cases via the CLI flags in the section above.

## What this repo shows
- How representation choices affect cluster structure and downstream performance (TF‑IDF → GloVe → LLM embeddings).
- How different dataset sources (lexicon vs LLM‑authored) influence separability while keeping labels identical.
- A single, unified results file combining classic ML and LLM labelling so you can compare quickly and communicate clearly.
- Simple, modular code you can reuse (data generation, vectorisation, clustering, reduction, metrics, visualisation).
- Graceful key handling via `.env` and caching to avoid repeated API calls.

## License
MIT.

## To cite
If you fork/apply this repository for your research, please cite as:

Bibtex:

```latex
@misc{angus2025textwaves,
  author = {Angus, SD},
  title = {TextWaves: simple NLP analysis on a tiny synthetic dataset},
  year = {2025},
  url = {https://github.com/sodalabsio/textwaves},
  note = {GitHub repository}
}
```

APA:

```text
Angus, S. (2025). TextWaves: simple NLP analysis on a tiny synthetic dataset [Computer software]. GitHub. https://github.com/sodalabsio/textwaves
```
