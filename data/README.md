# data/

**This folder is intentionally empty in the public repository.**

Neither of the two datasets used in this study can be redistributed here.

## RIDOC administrative data

Individual-level pretrial detention records were provided to the research team
by the Rhode Island Department of Corrections under an approved research
agreement and are not ours to redistribute. Researchers who want these data
should submit a request to RIDOC. The corresponding author
(justin_berk@brown.edu) can advise on the request process.

## Jail Data Initiative (JDI) data

The national jail roster data were provided by the NYU Public Safety Lab's
Jail Data Initiative. Access is by request through
https://jaildatainitiative.org/

## What you need in this folder to run the code

The scripts expect the raw extracts to be placed here locally (they are
blocked from git by `.gitignore`):

```
data/
  raw/
    ridoc_charges.csv        # one row per charge
    jdi_bookings.csv         # one row per booking
    acs2019_zip_income.csv   # public - ACS 2019 median household income by ZIP
    cdc_svi_county.csv       # public - CDC Social Vulnerability Index by FIPS
  processed/                 # created by the cleaning scripts
```

The two public files (ACS and CDC SVI) are freely downloadable; see
`docs/data_access.md` for the exact sources.

Variable definitions for every field the code expects are in
`docs/data_dictionary_ridoc.csv` and `docs/data_dictionary_jdi.csv`.
