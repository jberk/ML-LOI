# Predicting Length of Incarceration in Jail to Guide HCV Treatment

Analytic code for:

> Berk J, Wolfer-Jenkins C, Allen B, Paul A, Martin M, James M-E, Akiyama MJ,
> Bovell-Ammon BJ. *Evaluating the Application of Machine Learning to Predict
> Length of Incarceration and Guide HCV Treatment in a Jail Setting.* (under review)

## What this study did

Hepatitis C treatment takes about 8 weeks. In jails, most people are released
long before that, and no one knows at admission who will still be there. We
asked whether machine learning applied to data routinely available at booking
could identify people likely to stay long enough to complete treatment.

Two datasets were used:

- **RIDOC** — all pretrial detentions ending between January 2012 and
  February 2023 in Rhode Island's unified jail-prison system
  (109,430 incarceration events; 51,492 in the analytic sample).
- **JDI** — daily jail roster data collected by the NYU Public Safety Lab's
  Jail Data Initiative, January 2020 to May 2025, 42 states
  (892,215 events in the analytic sample).

Models: logistic regression, balanced random forest, CatBoost, histogram-based
gradient boosting (RIDOC) / LightGBM (JDI), and a stacked ensemble.
Outcomes: stay of at least 56 days (primary) and at least 28 days.

The headline result is negative: prediction was poor across every model and
both datasets (PR-AUC 0.28-0.35 in RIDOC). This repository exists so that
result can be checked and built on.

## Data availability

**No individual-level data are in this repository, and none can be added.**
See [`data/README.md`](data/README.md) for how to request each dataset, and
[`docs/data_access.md`](docs/data_access.md) for the public data sources
(ACS income, CDC SVI) that the code also uses.

What *is* here: the analysis code, a full data dictionary describing every
variable and how it was derived, and the aggregate results reported in the
paper.

## Repository contents

```
code/      analysis scripts, numbered in run order
docs/      data dictionaries, data access instructions, TRIPOD+AI checklist
results/   aggregate tables and figures reported in the paper
data/      empty by design - see data/README.md
```

## Reproducing the analysis

Requires Python 3.11.6.

```bash
git clone https://github.com/<username>/<repo>.git
cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place the raw extracts in `data/raw/` as described in `data/README.md`, then
run the scripts in `code/` in numbered order.

## Ethics

Approved by the Brown University Health Prisoner IRB and the Rhode Island
Department of Corrections Medical Research Advisory Group. The IRB waived
informed consent given the retrospective design. Reporting follows TRIPOD+AI.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

Code is released under the MIT License (see [`LICENSE`](LICENSE)).
Documentation and results tables are released under CC BY 4.0.

## Contact

Justin Berk, MD MPH MBA - justin_berk@brown.edu
ORCID: [0000-0002-2865-7464](https://orcid.org/0000-0002-2865-7464)
