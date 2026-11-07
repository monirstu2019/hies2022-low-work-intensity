# Explaining Low Work Intensity among Working Women in Bangladesh

Reproducibility materials for the manuscript **“Explaining Low Work Intensity among Working Women in Bangladesh: An Explainable Machine Learning Analysis of National Survey Data.”**

## Important data-access notice

This repository package includes the **2,110-record analytical extract**, a full 720-PSU stratified survey-design frame, analysis scripts and documentation. The original HIES 2022 raw modules and the author-constructed **full 62,407-record** merged dataset are **not** included. The analytical extract retains respondent-level HIES-derived information, so **the authors must verify that their BBS data-use agreement or written permission allows sharing this particular extract** and perform a disclosure-risk review before uploading it publicly. Pseudonymized cluster numbers and removed identifiers alone do not guarantee anonymity. If sharing is not permitted, remove the `data/` folder before publishing this repository.

The authors' Stata file `HIES2022_Data_Preparation_and_Merging.do` records how the merged dataset was constructed. The original survey modules must be obtained separately under BBS access rules:

- BBS microdata catalogue: <https://nsds.bbs.gov.bd/en/microdata>
- BBS e-service / microdata application portal: <https://eservice.bbs.gov.bd/>

**Reproducibility limitation:** the released analytical extract plus 720-PSU frame support inspection of sample selection, variables, cleaning and study design, but are **not a drop-in replacement** for the full merged dataset in the supplied `main_analysis.py` and supporting scripts. Exact end-to-end reproduction needs authorized input data and the original index/design context, or further tested adaptations. The full workflow was not rerun end-to-end on the reduced release.

## What is included

- `data/Analytical_Dataset_2110.csv`: reduced analytical subset with 2,110 records and 59 relevant variables; names, source household/person identifiers and original source PSU labels are excluded.
- `data/Survey_Design_PSU_Frame_720.csv`: the full 16-stratum, 720-cluster frame with the same pseudonymous PSU values as the analytical file.
- `docs/Analytical_Variable_Dictionary.csv`, `docs/REVIEW_BEFORE_SHARING.md`, and `docs/VERIFICATION.json`: released-variable definitions, processing and data-sharing limitations, and numerical integrity checks.

- `code/stata/HIES2022_Data_Preparation_and_Merging.do`: author-written Stata workflow that merges and prepares the HIES modules into the person-level merged file used for the analysis.
- `code/python/main_analysis.py`: main Python workflow (the supplied `TTT(7).py`) covering the primary descriptive, survey-design, nested-CV, Boruta, model comparison, calibration, SHAP, and sensitivity steps.
- `code/python/Step1_40...` through `Step1_48...`: supporting tuning, outer-fold prediction, bootstrap, SHAP, figure, sensitivity, and final QA scripts invoked by the main workflow.
- `docs/VARIABLE_DICTIONARY.csv`: variable names, Stata variable labels, storage types, missingness counts, and sensitivity notes from the author-supplied merged file. This file contains metadata only, not observations.
- `docs/MANUSCRIPT_VARIABLE_DEFINITIONS.csv`: the manuscript's predictor definition/coding table.
- `docs/STATA_VALUE_LABELS.csv`: categorical display labels extracted from the Stata file metadata; no respondent-level values are included.
- `docs/INPUT_MODULES.csv`: source HIES modules referenced by the Stata preparation script.
- `docs/ANALYTICAL_WORKFLOW.md` and `docs/DATA_ACCESS_AND_RESTRICTIONS.md`: workflow and access notes.
- `archive/earlier_workflows/`: earlier exploratory scripts, retained for provenance only; they are not invoked by the final reported-results workflow.

## Dataset construction and analytical sample

The author-supplied merged file `final_hies_women_lfp_earnings_ml_ready.dta` contains 62,407 records and 324 variables. It was constructed from multiple HIES 2022 modules using `HIES2022_Data_Preparation_and_Merging.do`; it is not the original raw HIES release. The main Python workflow applies `sample_earnings_women_15plus == 1` to select 2,110 working women aged 15 years or older. The supplied dataset check yields 553 cases with annual working hours at or below 1,440. The 1,440-hour cut-off is the unweighted 25th percentile in the analytical sample.

The supplied final analysis code was written against the full merged 62,407-row file, not solely the 2,110-row subset. In particular, the full survey PSU frame (720 PSU–stratum units across 16 strata) is used for the primary PSU bootstrap and the fully nested sensitivity workflow; the analytical sample itself spans 589 PSU–stratum units. Therefore, a 2,110-row-only file is not a drop-in replacement for the full analysis input.

## Reproduction requirements

1. Obtain authorized HIES 2022 source modules from BBS.
2. Place the required source modules in a local folder and set the Stata `HIES_RAW_DIR` environment variable, or edit the fallback path at the top of `code/stata/HIES2022_Data_Preparation_and_Merging.do`.
3. Run `HIES2022_Data_Preparation_and_Merging.do` in Stata 17 or later. It creates the merged file `final_hies_women_lfp_earnings_ml_ready.dta` in the derived-output folder.
4. Set `HIES_MERGED_DATA` to the path of that merged `.dta` file and `HIES_OUTPUT_DIR` to a writable output folder.
5. Install the Python packages listed in `requirements.txt` and run `code/python/main_analysis.py` in the same Python environment. The main script invokes the supporting scripts in `code/python/`.
6. For final verification, run `Step1_48_Final_Results_QA.py` after the preceding outputs have been generated.

The analysis is computationally intensive. The nested tuning, outer-fold prediction, SHAP, and 1,000-replicate bootstrap steps may require substantial memory and runtime. The complete pipeline has not been re-executed from raw HIES modules as part of preparing this public code package because those source modules are not included.

## Environment variables

- `HIES_RAW_DIR`: folder containing the authorized original HIES 2022 `.dta` modules required by `HIES2022_Data_Preparation_and_Merging.do`.
- `HIES_DERIVED_DIR`: folder where the Stata preparation workflow writes intermediate and merged files.
- `HIES_MERGED_DATA`: full merged 62,407-row `.dta` file used by the Python analysis.
- `HIES_OUTPUT_DIR`: folder for generated analysis outputs.

## Analysis design summary

- Analytical sample: 2,110 working women aged 15+.
- Outcome: `low_work_intensity = 1` when annual working hours are `<= 1,440`; otherwise 0.
- Outcome cases: 553 (26.21% unweighted); weighted prevalence reported in the manuscript: 23.78%.
- Survey design: 720 PSU–stratum units in the full HIES frame and 16 strata; 589 PSU–stratum units represented in the analytical domain.
- Candidate predictor groups: 35.
- Primary modelling: six classifiers; five outer and three inner PSU-grouped folds; training-fold preprocessing and Boruta feature selection.
- Evaluation: survey-weighted average precision, ROC-AUC, and Brier score.
- Explainability: held-out Random Forest SHAP grouped to original predictors.
- Sensitivity: fully nested RF configurations and paired, stratified PSU bootstrap.

## Important reproducibility qualification

This repository provides the analytical dataset comprising 2,110 working women aged 15 years or older, the accompanying 720-PSU survey-design frame, Stata data-preparation code, Python analysis scripts, variable definitions, and supporting documentation. The analytical dataset was derived from an author-constructed merged dataset containing 62,407 observations and 324 variables, prepared from the original HIES 2022 survey modules. The original HIES survey modules and the full merged dataset are not redistributed in this repository. Researchers requiring the original survey data may obtain access through the Bangladesh Bureau of Statistics (BBS), subject to its access procedures. While the shared analytical dataset supports verification of the study's variables and principal analytical inputs, certain survey-design and sensitivity analyses require the full merged dataset or additional saved computational outputs. Therefore, complete numerical reproduction of all reported findings is not claimed from the reduced dataset alone. Direct links to the shared analytical data and supporting materials are provided in the repository and should be cited in the manuscript.
