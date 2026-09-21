# code/

All analyses were run in Python 3.11.6. Scripts use relative paths and expect
the raw extracts in `data/raw/` — see [`../data/README.md`](../data/README.md).
Those data are not in this repository and cannot be.

## 01_cleaning/

Builds the analytic files from the raw extracts.

| Script | What it does |
|---|---|
| `ridoc_56d_mice_cleaning.py` | Collapses the charge-level RIDOC extract (229,017 rows) to one row per incarceration event (109,430), applies exclusions, links ACS 2019 ZIP median household income, imputes missing values by MICE (5 imputations), and derives the 56-day long-stay outcome. Produces the primary analytic file (n=51,492). |
| `ridoc_28d_mice_cleaning.py` | Same, 28-day threshold. |
| `ridoc_56d_completecase_cleaning.py` | Complete-case alternative (Appendix C). |
| `ridoc_28d_completecase_cleaning.py` | Same, 28-day threshold. |
| `jdi_56d_cleaning.py` | Cleans the JDI national extract, drops states missing top charge (CT, MA, PA), links CDC county SVI, imputes by MICE. Produces the JDI analytic file (n=892,215). |
| `jdi_28d_cleaning.py` | Same, 28-day threshold. |
| `national_dataset_clean1.py`, `national_dataset_clean2.py` | Earlier passes assembling the state-level JDI files into one national dataset. Retained for provenance. |

## 02_models_ridoc/

Hyperparameters selected by nested stratified 5-fold cross-validation with
grid search maximizing average precision, stratified on the long-stay outcome.
Five models at each threshold; the stacked ensemble's meta-model is a logistic
regression fit on the four base model outputs.

`ridoc_56d_logistic.py`, `ridoc_56d_balanced_rf.py`, `ridoc_56d_catboost.py`,
`ridoc_56d_hgbm.py`, `ridoc_56d_stacked.py`, and the five 28-day equivalents.

`ridoc_56d_stacked_balanced.py` is the balanced variant of the 56-day stacked
ensemble.

## 03_models_jdi/

The same structure on the JDI file, with two substitutions for tractability at
~3 million observations: logistic regression fit by stochastic gradient
descent, and LightGBM in place of HGBM.

`jdi_56d_logistic_sgd.py`, `jdi_56d_balanced_rf.py`, `jdi_56d_catboost.py`,
`jdi_56d_lightgbm.py`.

## 04_sensitivity/

Named by the supplemental appendix each one supports.

| Appendix | Scripts |
|---|---|
| C — complete case, pre-COVID period, and both combined | `appC_completecase_fulltime_*.py` (4), `appC_completecase_precovid_*.py` (4), `appC_mice_precovid_*.py` (4) |
| D — predictive probability threshold tuning | `appD_threshold_tuning_*.py` (3) |

## mappings/

`ridoc_encoding_mappings.csv` and `jdi_encoding_mappings.csv` record how each
categorical variable's levels were encoded for modeling. Labels and codes
only — no counts, no records.

## Known gaps

These are not yet in the repository and should be added before submission:

- **JDI 28-day models and the JDI stacked ensemble.** These exist only as
  Jupyter notebooks and are being exported to `.py`.
- **Undersampling sensitivity scripts** (the other half of Appendix D).
  Being reconstructed; results to be re-verified against the reported numbers.
- **Appendices E through I** — continuous and ordinal outcome models,
  criminal history score, bail encoding, and feature importance. No scripts
  for these are currently in hand.
- **Table and figure generation.** Nothing here writes out the published
  tables. Add as `05_tables_figures.py` if a script exists.

## Before committing

- Clear all notebook outputs before converting or committing any `.ipynb`
- Confirm nothing in `results/` reports a cell count under 11
