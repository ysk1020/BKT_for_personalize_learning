# Knowledge Tracing & Personalized Skill Recommendation (ASSISTments)

This project uses **Bayesian Knowledge Tracing (BKT)** to estimate how well each student has mastered each math skill, then turns those estimates into a **personalized, explainable list of skills to revisit**. It is built on the ASSISTments *Skill Builder* dataset.

## Pipeline at a glance

```
data/skill_builder_data.csv
        │  01_explore        → understand & validate the raw data
        ▼
        │  02_preprocess     → clean data
outputs/clean_data.csv
        │  03_bkt            → fit BKT per skill, estimate per-student mastery
outputs/mastery_estimates.csv
        │  04_evaluation     → held-out (by student) predictive validity
        │  05_recommendations→ knowledge gaps, ranked recommendations, scenario comparison
outputs/recommendations.csv, scenario_comparison.csv, ...
```

## Project structure

| Path | Contents |
|---|---|
| `data/` | Raw dataset `skill_builder_data.csv` (525,534 rows × 30 columns) |
| `notebook/` | The five Jupyter notebooks, run in order (`01` → `05`) |
| `markdown/` | Notebooks exported to Markdown (with figures) for easy reading without Jupyter |
| `outputs/` | Generated CSVs and `figures/` produced by the notebooks |

## Notebooks

1. **[01_explore](notebook/01_explore.ipynb)** – Loads the raw data and checks missing values (15% of rows lack a skill label), original vs. scaffold rows, duplicates, and student/skill statistics.
2. **[02_preprocess](notebook/02_preprocess.ipynb)** – Cleaning steps:
   - drop rows without `skill_name`
   - keep only original problems (`original = 1`), removing scaffold/hint steps
   - remove duplicate events per `(user_id, skill_name, order_id)`
   - drop rare skills (< 100 interactions) and students with too few interactions

   Result: `outputs/clean_data.csv` (294,418 rows, 3,053 students, 92 skills).
3. **[03_bkt](notebook/03_bkt.ipynb)** – Fits one BKT model per skill with [pyBKT](https://github.com/CAHLR/pyBKT) (EM algorithm; parameters: prior knowledge L0, learn T, slip S, guess G) and computes each student's mastery probability per skill → `mastery_estimates.csv`.
4. **[04_evaluation](notebook/04_evaluation.ipynb)** – Splits **by student** (80/20, seed 42), fits BKT on training students only, and predicts the next response for held-out students. Reports AUC, accuracy and RMSE against a baseline, plus per-skill AUC. This establishes *predictive validity* only; it does not prove causal learning gains.
5. **[05_recommendations](notebook/05_recommendations.ipynb)** – Flags a **knowledge gap** when mastery < 0.8, ranks gaps lowest-mastery first, and generates plain-language explanations. Compares a personalized path against a non-personalized one and measures "wasted effort" (skills already mastered that a student would be asked to practice).

## Key results

| Metric | BKT | Baseline |
|---|---|---|
| AUC (held-out students) | 0.720 | 0.621 |
| Accuracy | 0.713 | 0.659 |
| RMSE | 0.440 | 0.466 |

- 611 held-out students, 88 skills evaluated.
- 28.2% of student–skill pairs are knowledge gaps (avg. 4.3 gaps per student).
- In the non-personalized scenario, on average ~64% of a student's assigned skills were already mastered (median ~67%), which personalization avoids.

(See `outputs/evaluation_results.csv` and `outputs/scenario_comparison.csv`.)

## Outputs

| File | Description |
|---|---|
| `clean_data.csv` | Preprocessed interactions |
| `mastery_estimates.csv` | Mastery probability per student × skill |
| `test_predictions.csv` | BKT predictions on held-out students |
| `evaluation_results.csv` | Overall metrics vs. baseline |
| `per_skill_auc.csv` | AUC per skill |
| `recommendations.csv` | Ranked recommended skills with explanations |
| `scenario_comparison.csv` | Wasted-effort comparison per student |
| `figures/` | Plots used in the thesis |

## Setup & reproduction

Requires Python 3 and Jupyter.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook
```

Then run the notebooks in `notebook/` in order, `01` through `05`. Each notebook reads the previous stage's saved output, so they must be run sequentially. Paths are relative (`../data`, `../outputs`), so launch Jupyter from the project root or open notebooks from within `notebook/`.

## Notes & limitations

- The raw CSV is read with `ISO-8859-1` encoding.
- Mastery is a *latent estimate*; recommendations are decision support and flagged as requiring teacher review.
- Evaluation shows predictive accuracy, not that following recommendations improves learning.
