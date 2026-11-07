# Analytical workflow and key formulas

## 1. Data construction

`code/stata/HIES2022_Data_Preparation_and_Merging.do` imports and merges HIES 2022 modules at person and household levels. The final output is `final_hies_women_lfp_earnings_ml_ready.dta`. It contains 62,407 records and 324 variables in the supplied author file. The do-file depends on the original BBS HIES 2022 source modules listed in `INPUT_MODULES.csv`; those modules are not redistributed here.

## 2. Analytical sample

The Python workflow selects records with `sample_earnings_women_15plus == 1`, producing 2,110 working women aged 15 years or older. The selected domain contains 589 PSU–stratum groups across 16 strata. The full source file contains the 720 PSU–stratum units used in the full-frame variance/bootstrap routines.

## 3. Outcome definition

Let `h_i` be annual working hours. The threshold is the unweighted sample 25th percentile, `Q_0.25(h) = 1,440`.

`low_work_intensity_i = 1` if `h_i <= 1,440`; otherwise `0`.

This is a sample-relative classification and is not an internationally standardized underemployment threshold. There are 553 positive cases (26.21% unweighted). The survey-weighted prevalence is calculated as:

`P_w = sum_i(w_i * Y_i) / sum_i(w_i)`

where `w_i` is the HIES population sampling weight and the sum is over the 2,110-person analytical domain.

## 4. Predictors and preprocessing

The manuscript defines 35 candidate predictor groups across demographic, education, digital/financial access, functional limitation, geography, household, food security, wealth/living standards, housing/infrastructure, and employment structure. See `MANUSCRIPT_VARIABLE_DEFINITIONS.csv`. Structural missingness from questionnaire routing is distinguished from ordinary missingness. Structural zeroes are recoded where the questionnaire logic supports zero; not-applicable responses are retained as special categories (commonly -99). Rare-category handling, imputation, scaling, PCA, and Boruta selection are learned within training folds to avoid information leakage.

The household living-standard index uses 14 housing, property, expenditure, and durable-asset indicators. Winsorization at training-fold 1st/99th percentiles, log(1+x), median imputation, standardization, and PCA are fitted separately within training partitions.

## 5. Validation and models

The primary workflow evaluates six classifiers using five outer and three inner PSU-grouped folds. Entire PSU clusters are assigned to a single fold. Boruta feature selection and hyperparameter tuning are confined to training partitions. Survey weights are used for model fitting and weighted evaluation.

Primary metrics are survey-weighted Average Precision (AP), ROC-AUC, and Brier score. The manuscript reports approximate confidence intervals based on stratified PSU bootstrap with out-of-fold predictions held fixed; they do not include uncertainty from refitting the entire modelling pipeline.

## 6. Explainability and sensitivity

Random Forest is selected for SHAP interpretation based on the highest observed weighted AP. SHAP values are generated for held-out outer-fold observations and aggregated to original predictor groups. SHAP values explain model predictions and are not causal effects.

Sensitivity analysis compares the full 35-predictor configuration with configurations excluding three core or all six employment predictors. The fully nested sensitivity routine repeats preprocessing, PCA, Boruta, and tuning within each training partition. Paired PSU bootstrap uses 1,000 replicates and fixed out-of-fold predictions.
