
# %% STEP 1.1: LOAD DATA AND VERIFY ANALYTICAL SAMPLE

# ============================================================
# Research: Low Work Intensity among Working Women
# Dataset: HIES 2022, Bangladesh
# ============================================================

# 1. Import essential libraries
import os
import numpy as np
import pandas as pd

# Repository path configuration (added for portability; does not alter analysis methods)
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT_ROOT / "code" / "python"
DATA_FILE = Path(os.environ.get("HIES_MERGED_DATA", str(PROJECT_ROOT / "restricted_data" / "final_hies_women_lfp_earnings_ml_ready.dta")))
OUTPUT_DIR_PATH = Path(os.environ.get("HIES_OUTPUT_DIR", str(PROJECT_ROOT / "outputs")))
OUTPUT_DIR_PATH.mkdir(parents=True, exist_ok=True)

# 2. Define dataset path
file_path = str(DATA_FILE)

# 3. Create output directory
output_dir = str(OUTPUT_DIR_PATH)
os.makedirs(output_dir, exist_ok=True)

# 4. Verify dataset exists
assert os.path.exists(file_path), "Dataset file not found!"

# 5. Load Stata dataset
df = pd.read_stata(
    file_path,
    convert_categoricals=False
)

print("========== ORIGINAL DATASET ==========")
print("Total observations:", df.shape[0])
print("Total variables:", df.shape[1])

# 6. Select working women aged 15+
women = df.loc[
    df["sample_earnings_women_15plus"] == 1
].copy()

print("\n========== ANALYTICAL SAMPLE ==========")
print("Working women:", len(women))
print("Number of variables:", women.shape[1])

# 7. Verify required survey design variables
required_vars = [
    "earn_annual_hours",
    "population_weight",
    "psu",
    "stratum16"
]

missing_vars = [
    v for v in required_vars
    if v not in women.columns
]

assert not missing_vars, (
    f"Missing required variables: {missing_vars}"
)

print("\n========== DATA QUALITY CHECK ==========")

print("Missing annual hours:",
      women["earn_annual_hours"].isna().sum())

print("Missing population weights:",
      women["population_weight"].isna().sum())

print("Missing PSU:",
      women["psu"].isna().sum())

print("Missing strata:",
      women["stratum16"].isna().sum())

print("Number of strata:",
      women["stratum16"].nunique())

print("Number of PSU-stratum groups:",
      women[["stratum16", "psu"]]
      .drop_duplicates().shape[0])

print("\nSTEP 1.1 COMPLETED.")



# %% STEP 1.2: OUTCOME CONSTRUCTION AND WEIGHTED PREVALENCE

# ============================================================
# Research: Low Work Intensity among Working Women
# Dataset: HIES 2022, Bangladesh
# ============================================================

# 1. Convert working hours and survey weights to numeric
annual_hours = pd.to_numeric(
    women["earn_annual_hours"],
    errors="coerce"
)

population_weights = pd.to_numeric(
    women["population_weight"],
    errors="coerce"
)

# 2. Validate working hours and population weights
assert annual_hours.notna().all(), \
    "Missing annual working hours detected!"

assert (annual_hours >= 0).all(), \
    "Negative annual working hours detected!"

assert np.isfinite(population_weights).all(), \
    "Invalid population weights detected!"

assert (population_weights > 0).all(), \
    "Zero or negative population weights detected!"

# 3. Calculate the unweighted 25th percentile
p25_hours = annual_hours.quantile(0.25)

# 4. Construct the binary outcome
# 1 = Low work intensity (hours <= 25th percentile)
# 0 = Not low work intensity

women["low_work_intensity"] = (
    annual_hours <= p25_hours
).astype(int)

# 5. Calculate unweighted distribution
frequency = (
    women["low_work_intensity"]
    .value_counts()
    .reindex([0, 1], fill_value=0)
)

unweighted_percent = (
    frequency / len(women) * 100
)

# 6. Calculate survey-weighted distribution
weighted_percent = []

for outcome_code in [0, 1]:

    mask = women["low_work_intensity"] == outcome_code

    percentage = (
        population_weights.loc[mask].sum()
        / population_weights.sum()
        * 100
    )

    weighted_percent.append(percentage)

# 7. Create results table
outcome_table = pd.DataFrame({
    "Outcome": [
        "Not low work intensity",
        "Low work intensity"
    ],
    "Frequency": frequency.to_numpy(),
    "Unweighted_Percent": unweighted_percent.to_numpy(),
    "Survey_Weighted_Percent": weighted_percent
})

# 8. Print verification results
print("========== STEP 1.2 RESULTS ==========")

print("Analytical sample:", len(women))
print("25th percentile annual hours:", p25_hours)

print("Observations exactly at threshold:",
      int((annual_hours == p25_hours).sum()))

print("\nOUTCOME DISTRIBUTION:")
print(outcome_table.round(2).to_string(index=False))

print("\nValid population weights:",
      population_weights.notna().sum())

print("Unweighted low work intensity (%):",
      round(unweighted_percent.loc[1], 2))

print("Survey-weighted low work intensity (%):",
      round(weighted_percent[1], 2))

# 9. Final checks
assert len(women) == 2110
assert frequency.sum() == len(women)
assert women["low_work_intensity"].isin([0, 1]).all()

assert np.isclose(
    sum(weighted_percent), 100
)

print("\nSTEP 1.2 COMPLETED SUCCESSFULLY.")




# %% STEP 1.3: SURVEY DESIGN AND WEIGHT AUDIT

# ============================================================
# HIES 2022: Survey design verification
# Analytical population: Working women aged 15+
# ============================================================

import numpy as np
import pandas as pd

# 1. Prepare survey design dataset
survey_audit = women[
    [
        "stratum16",
        "psu",
        "population_weight",
        "low_work_intensity"
    ]
].copy()

survey_audit["population_weight"] = pd.to_numeric(
    survey_audit["population_weight"],
    errors="coerce"
)

# 2. Check validity
assert survey_audit.notna().all().all(), \
    "Missing survey design information detected."

assert np.isfinite(
    survey_audit["population_weight"]
).all(), "Non-finite survey weights detected."

assert (
    survey_audit["population_weight"] > 0
).all(), "Non-positive survey weights detected."

# 3. Identify unique PSU-stratum combinations
psu_summary = (
    survey_audit
    .groupby(["stratum16", "psu"])
    .agg(
        N_Women=("low_work_intensity", "size"),
        N_Low=("low_work_intensity", "sum"),
        Weight_Sum=("population_weight", "sum")
    )
    .reset_index()
)

# 4. Summarize strata
stratum_summary = (
    psu_summary
    .groupby("stratum16")
    .agg(
        PSUs=("psu", "size"),
        Analytical_Women=("N_Women", "sum"),
        Low_Cases=("N_Low", "sum")
    )
    .reset_index()
)

# 5. Sampling-weight diagnostics
w = survey_audit["population_weight"].to_numpy(
    dtype=float
)

# Kish effective sample size based on weights only
kish_effective_n = (w.sum() ** 2) / np.sum(w ** 2)

weight_cv = np.std(w, ddof=1) / np.mean(w)

# 6. Print survey structure
print("========== STEP 1.3 RESULTS ==========")

print("\nSURVEY DESIGN:")
print("Analytical observations:", len(survey_audit))
print("Number of strata:", len(stratum_summary))
print("PSU-stratum combinations:", len(psu_summary))

print("\nPSUs per stratum:")
print(stratum_summary.to_string(index=False))

print("\nPSU SIZE:")
print("Minimum women per PSU:",
      psu_summary["N_Women"].min())

print("Median women per PSU:",
      psu_summary["N_Women"].median())

print("Maximum women per PSU:",
      psu_summary["N_Women"].max())

print("\nSAMPLING WEIGHTS:")
print("Minimum:", round(w.min(), 4))
print("Median:", round(np.median(w), 4))
print("Mean:", round(w.mean(), 4))
print("Maximum:", round(w.max(), 4))
print("Coefficient of variation:",
      round(weight_cv, 4))

print("\nKish effective sample size (weights only):",
      round(kish_effective_n, 2))

# 7. Check PSU availability for variance estimation
single_psu_strata = stratum_summary.loc[
    stratum_summary["PSUs"] < 2,
    "stratum16"
].tolist()

print("\nStrata with fewer than two PSUs:",
      single_psu_strata)

# 8. Final verification
assert len(survey_audit) == 2110
assert len(stratum_summary) == 16
assert len(psu_summary) == 589

assert stratum_summary["Analytical_Women"].sum() == 2110

print("\nSTEP 1.3 VERIFICATION SUCCESSFUL.")






# %% STEP 1.4: SURVEY-WEIGHTED PREVALENCE AND 95% CI

# ============================================================
# HIES 2022: Domain-based survey variance estimation
# Outcome: Low work intensity
#
# Uses:
#   1. Full available survey PSU/stratum structure
#   2. Population sampling weights
#   3. Stratified PSU-level Taylor linearization
#
# Assumptions:
#   - First-stage PSU variance approximation
#   - No finite population correction
#   - 1440-hour cutoff treated as fixed
# ============================================================

import numpy as np
import pandas as pd

from scipy.stats import t
from scipy.special import expit, logit

# ------------------------------------------------------------
# 1. Identify analytical domain in the full dataset
# ------------------------------------------------------------

domain = (
    pd.to_numeric(
        df["sample_earnings_women_15plus"],
        errors="coerce"
    ) == 1
)

assert df.index.is_unique
assert women.index.is_unique

assert domain.sum() == 2110

assert set(df.index[domain]) == set(women.index), (
    "Analytical sample indices do not match."
)

# ------------------------------------------------------------
# 2. Preserve full survey design structure
# ------------------------------------------------------------

survey_full = df[
    ["stratum16", "psu"]
].copy()

assert survey_full.notna().all().all(), (
    "Missing PSU/stratum values in full survey data."
)

# Initialize zero domain contributions
survey_full["weighted_domain"] = 0.0
survey_full["weighted_low"] = 0.0

# Only analytical-domain observations contribute
domain_weights = pd.to_numeric(
    women["population_weight"],
    errors="coerce"
)

domain_outcome = women[
    "low_work_intensity"
].astype(int)

survey_full.loc[
    women.index, "weighted_domain"
] = domain_weights.to_numpy()

survey_full.loc[
    women.index, "weighted_low"
] = (
    domain_weights.to_numpy()
    * domain_outcome.to_numpy()
)

# ------------------------------------------------------------
# 3. Estimate weighted prevalence
# ------------------------------------------------------------

weighted_total = survey_full[
    "weighted_domain"
].sum()

weighted_low_total = survey_full[
    "weighted_low"
].sum()

p_hat = weighted_low_total / weighted_total

assert np.isclose(p_hat, 0.23780001, atol=0.0001)

# ------------------------------------------------------------
# 4. Linearized contribution for domain ratio
# ------------------------------------------------------------

survey_full["linearized"] = (
    survey_full["weighted_low"]
    - p_hat * survey_full["weighted_domain"]
)

# Sum linearized contributions within complete PSUs
psu_linearized = (
    survey_full
    .groupby(
        ["stratum16", "psu"],
        observed=True
    )["linearized"]
    .sum()
    .reset_index()
)

# ------------------------------------------------------------
# 5. Stratified PSU-level variance estimation
# ------------------------------------------------------------

variance_total = 0.0
design_df = 0
psu_counts = []

for stratum, group in psu_linearized.groupby(
    "stratum16",
    observed=True
):

    values = group["linearized"].to_numpy(
        dtype=float
    )

    n_psu = len(values)

    assert n_psu >= 2, (
        f"Insufficient PSUs in stratum {stratum}"
    )

    stratum_variance = (
        n_psu / (n_psu - 1)
    ) * np.sum(
        (values - values.mean()) ** 2
    )

    variance_total += stratum_variance

    design_df += n_psu - 1
    psu_counts.append(n_psu)

variance_p = variance_total / weighted_total**2

se_p = np.sqrt(variance_p)

# ------------------------------------------------------------
# 6. Approximate 95% confidence interval
# Logit transformation keeps limits between 0 and 1
# ------------------------------------------------------------

t_critical = t.ppf(
    0.975,
    df=design_df
)

se_logit = se_p / (
    p_hat * (1 - p_hat)
)

ci_lower = expit(
    logit(p_hat) - t_critical * se_logit
)

ci_upper = expit(
    logit(p_hat) + t_critical * se_logit
)

# ------------------------------------------------------------
# 7. Design effect relative to a simple SRS variance
# This is an approximate comparison only
# ------------------------------------------------------------

n_domain = len(women)

srs_variance = (
    p_hat * (1 - p_hat) / n_domain
)

approx_design_effect = (
    variance_p / srs_variance
)

# ------------------------------------------------------------
# 8. Final results
# ------------------------------------------------------------

print("========== STEP 1.4 RESULTS ==========")

print("\nSURVEY STRUCTURE:")
print("Full dataset observations:", len(df))
print("Analytical-domain observations:", n_domain)
print("Full dataset PSU groups:", len(psu_linearized))
print("Full dataset strata:", len(psu_counts))
print("Design degrees of freedom:", design_df)

print("\nSURVEY-WEIGHTED PREVALENCE:")
print("Weighted prevalence (%):",
      round(p_hat * 100, 4))

print("Taylor-linearized SE (% points):",
      round(se_p * 100, 4))

print("95% Confidence Interval (%):",
      f"[{ci_lower*100:.4f}, {ci_upper*100:.4f}]")

print("\nAPPROXIMATE DESIGN EFFECT:")
print(round(approx_design_effect, 4))

# ------------------------------------------------------------
# 9. Verification
# ------------------------------------------------------------

assert 0 < p_hat < 1
assert np.isfinite(se_p)
assert se_p > 0
assert 0 <= ci_lower < p_hat < ci_upper <= 1
assert len(psu_counts) == 16

print("\nSTEP 1.4 VERIFICATION SUCCESSFUL.")







# %% STEP 1.5: PREDICTOR AND MISSINGNESS AUDIT

# ============================================================
# HIES 2022 - Publication-oriented data quality assessment
# No imputation, encoding or feature selection in this step
# ============================================================

import numpy as np
import pandas as pd
import os

# ------------------------------------------------------------
# 1. Define 35 planned candidate predictors
# Eight grouped predictors are mapped to their raw variables.
# Wealth index and transport missingness flag will be created
# in later preprocessing steps.
# ------------------------------------------------------------

candidate_features = [
    "age",
    "marital_status",
    "relation_head_grp",
    "literate_basic",
    "highest_class_completed",
    "has_mobile",
    "has_bank_account",
    "has_mfs_account",
    "internet_used",
    "has_computer",
    "any_functional_difficulty",
    "rural_urban",
    "division_code",
    "rmo_code",
    "domain16_grp",
    "hh_MEMBER",
    "fies_raw_score",
    "wealth_living_index",
    "toilet_type_grp",
    "drinking_water_source_grp",
    "light_source",
    "cooking_stove_type",
    "cooking_fuel_grp",
    "electricity_hours_24h",
    "transport_stop_halfkm",
    "transport_stop_time_min",
    "transport_stop_time_min_missing",
    "regular_waste_collection",
    "slum_household",
    "earn_sector",
    "earn_emp_status_ag",
    "earn_emp_status_nonag",
    "earn_org_type_grp",
    "earn_ind_code_grp",
    "earn_occ_code_grp"
]

grouped_to_raw = {
    "relation_head_grp": "relation_head",
    "domain16_grp": "domain16",
    "toilet_type_grp": "toilet_type",
    "drinking_water_source_grp": "drinking_water_source",
    "cooking_fuel_grp": "cooking_fuel",
    "earn_org_type_grp": "earn_org_type",
    "earn_ind_code_grp": "earn_ind_code",
    "earn_occ_code_grp": "earn_occ_code"
}

derived_features = [
    "wealth_living_index",
    "transport_stop_time_min_missing"
]

raw_predictors = list(dict.fromkeys(
    grouped_to_raw.get(v, v)
    for v in candidate_features
    if v not in derived_features
))

# ------------------------------------------------------------
# 2. Wealth index inputs and structural-missingness flags
# ------------------------------------------------------------

wealth_vars = [
    "hh_rooms",
    "covered_area",
    "house_value",
    "land_property_value",
    "other_asset_value",
    "sec9a1_food_value",
    "sec9b2_value_total",
    "sec9c_total_value",
    "sec9d1_value_total",
    "sec9d2_value_total",
    "durable_owned_count",
    "durable_total_number",
    "durable_total_value",
    "sec9_total_observed_value"
]

structural_flags = [
    "owns_land_property_nonoperated",
    "owns_other_assets",
    "ever_attended_education"
]

required_inputs = list(dict.fromkeys(
    raw_predictors + wealth_vars + structural_flags
))

# ------------------------------------------------------------
# 3. Check variable availability
# ------------------------------------------------------------

missing_variables = [
    v for v in required_inputs
    if v not in women.columns
]

print("========== STEP 1.5 RESULTS ==========")

print("\nPLANNED PREDICTORS:")
print("Original candidate predictors:",
      len(candidate_features))

print("Raw predictor inputs:",
      len(raw_predictors))

print("Features to create later:",
      derived_features)

print("Wealth PCA input variables:",
      len(wealth_vars))

print("\nMissing required dataset columns:",
      missing_variables)

assert len(candidate_features) == 35
assert len(raw_predictors) == 33
assert len(wealth_vars) == 14
assert not missing_variables, (
    "Required inputs missing. Check variable names."
)

# ------------------------------------------------------------
# 4. Examine missingness in candidate predictor inputs
# ------------------------------------------------------------

audit_rows = []

for var in raw_predictors:

    s = women[var]

    numeric = pd.to_numeric(s, errors="coerce")

    audit_rows.append({
        "Variable": var,
        "Missing_N": int(s.isna().sum()),
        "Missing_Percent": 100 * s.isna().mean(),
        "Unique_Values": int(s.nunique(dropna=True)),
        "Negative_N": int((numeric < 0).sum()),
        "Code_minus99_N": int((numeric == -99).sum()),
        "Code_minus88_N": int((numeric == -88).sum())
    })

predictor_audit = pd.DataFrame(audit_rows)

predictor_audit = predictor_audit.sort_values(
    ["Missing_Percent", "Variable"],
    ascending=[False, True]
).reset_index(drop=True)

print("\nRAW PREDICTOR MISSINGNESS:")
print(
    predictor_audit.round(2).to_string(index=False)
)

# ------------------------------------------------------------
# 5. Check wealth PCA input missingness
# ------------------------------------------------------------

wealth_audit = pd.DataFrame({
    "Variable": wealth_vars,
    "Missing_N": [
        int(women[v].isna().sum())
        for v in wealth_vars
    ],
    "Missing_Percent": [
        100 * women[v].isna().mean()
        for v in wealth_vars
    ]
})

wealth_audit = wealth_audit.sort_values(
    "Missing_Percent",
    ascending=False
)

print("\nWEALTH INDEX INPUT MISSINGNESS:")
print(
    wealth_audit.round(2).to_string(index=False)
)

# ------------------------------------------------------------
# 6. Recalculate weighted prevalence with float64
# ------------------------------------------------------------

w64 = women["population_weight"].to_numpy(
    dtype=np.float64
)

y64 = women["low_work_intensity"].to_numpy(
    dtype=np.float64
)

weighted_prevalence_64 = np.average(
    y64,
    weights=w64
)

print("\nFLOAT64 WEIGHT VERIFICATION:")
print(
    "Weighted prevalence (%):",
    round(weighted_prevalence_64 * 100, 6)
)

print(
    "Difference from Step 1.4:",
    round(
        weighted_prevalence_64 - p_hat,
        12
    )
)

assert np.isclose(
    weighted_prevalence_64,
    p_hat,
    atol=1e-10
)

# ------------------------------------------------------------
# 7. Save audit tables for reproducibility
# ------------------------------------------------------------

predictor_audit.to_csv(
    os.path.join(
        output_dir,
        "Step1_5_Predictor_Audit.csv"
    ),
    index=False
)

wealth_audit.to_csv(
    os.path.join(
        output_dir,
        "Step1_5_Wealth_Input_Audit.csv"
    ),
    index=False
)

print("\nSTEP 1.5 VERIFICATION SUCCESSFUL.")







# %% STEP 1.6: STRUCTURAL MISSINGNESS VERIFICATION

# ============================================================
# HIES 2022 - Survey routing and structural missingness audit
#
# Purpose:
# Distinguish potentially structural missingness from
# ordinary missing data BEFORE preprocessing.
#
# No imputation or recoding is performed in this step.
# ============================================================

import numpy as np
import pandas as pd
import os

# ------------------------------------------------------------
# 1. Define survey-routing relationships
# ------------------------------------------------------------

routing_checks = [
    ("earn_sector", "earn_emp_status_ag"),
    ("earn_sector", "earn_emp_status_nonag"),
    ("earn_sector", "earn_org_type"),
    ("ever_attended_education",
     "highest_class_completed"),
    ("transport_stop_halfkm",
     "transport_stop_time_min"),
    ("owns_land_property_nonoperated",
     "land_property_value"),
    ("owns_other_assets", "other_asset_value"),
    ("rural_urban", "transport_stop_halfkm"),
    ("rural_urban", "electricity_hours_24h")
]

# ------------------------------------------------------------
# 2. Verify all required variables exist
# ------------------------------------------------------------

all_variables = set()

for group_var, target_var in routing_checks:
    all_variables.add(group_var)
    all_variables.add(target_var)

missing_columns = sorted(
    all_variables - set(women.columns)
)

assert not missing_columns, (
    f"Missing variables: {missing_columns}"
)

# ------------------------------------------------------------
# 3. Create a reusable missingness cross-tab function
# ------------------------------------------------------------

def missingness_by_group(data, group_var, target_var):

    # Preserve actual survey codes without interpreting them
    group_codes = (
        data[group_var]
        .astype("string")
        .fillna("<MISSING>")
    )

    target_missing = data[target_var].isna()

    # Number observed/missing within each group
    result = pd.crosstab(
        group_codes,
        target_missing
    )

    result = result.reindex(
        columns=[False, True],
        fill_value=0
    )

    result.columns = [
        "Observed_N",
        "Missing_N"
    ]

    result["Total_N"] = (
        result["Observed_N"] + result["Missing_N"]
    )

    result["Missing_Percent"] = (
        100 * result["Missing_N"] / result["Total_N"]
    )

    result.index.name = group_var

    assert result["Total_N"].sum() == len(data)

    return result.reset_index()

# ------------------------------------------------------------
# 4. Run all routing checks
# ------------------------------------------------------------

routing_results = []

print("========== STEP 1.6 RESULTS ==========")

for group_var, target_var in routing_checks:

    result = missingness_by_group(
        women,
        group_var,
        target_var
    )

    result.insert(0, "Target_Variable", target_var)
    result.insert(1, "Routing_Variable", group_var)

    routing_results.append(result)

    print("\n" + "=" * 60)
    print("TARGET:", target_var)
    print("GROUPED BY:", group_var)
    print("=" * 60)

    print(
        result[
            [
                group_var,
                "Observed_N",
                "Missing_N",
                "Total_N",
                "Missing_Percent"
            ]
        ].round(2).to_string(index=False)
    )

# ------------------------------------------------------------
# 5. Inspect actual ownership and attendance codes
# ------------------------------------------------------------

print("\n========== ROUTING CODE FREQUENCIES ==========")

routing_variables = [
    "earn_sector",
    "ever_attended_education",
    "transport_stop_halfkm",
    "owns_land_property_nonoperated",
    "owns_other_assets"
]

for variable in routing_variables:

    print(f"\n{variable}:")

    print(
        women[variable]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

# ------------------------------------------------------------
# 6. Check observed zero and negative asset values
# ------------------------------------------------------------

print("\n========== ASSET VALUE DIAGNOSTICS ==========")

for variable in [
    "land_property_value",
    "other_asset_value"
]:

    values = pd.to_numeric(
        women[variable],
        errors="coerce"
    )

    print(f"\n{variable}:")
    print("Observed:", int(values.notna().sum()))
    print("Missing:", int(values.isna().sum()))
    print("Observed zero:", int((values == 0).sum()))
    print("Observed negative:", int((values < 0).sum()))
    print("Observed positive:", int((values > 0).sum()))

# ------------------------------------------------------------
# 7. Save results
# ------------------------------------------------------------

# Each cross-tab has a different grouping column.
# Export them separately to retain clear column names.

for i, result in enumerate(routing_results, start=1):

    result.to_csv(
        os.path.join(
            output_dir,
            f"Step1_6_Routing_Check_{i:02d}.csv"
        ),
        index=False
    )

# ------------------------------------------------------------
# 8. Final verification
# ------------------------------------------------------------

assert len(routing_results) == len(routing_checks)
assert len(women) == 2110

print("\nSTEP 1.6 VERIFICATION SUCCESSFUL.")








# %% STEP 1.7: STRUCTURAL MISSING-VALUE RECODING

# ============================================================
# HIES 2022 - Verified structural missingness rules
#
# IMPORTANT:
# - Preserve original "women"
# - Do not fit imputation or scaling on full data
# - No PCA, Boruta, or model training here
# ============================================================

import numpy as np
import pandas as pd

# 1. Create a separate cleaned dataset
women_clean = women.copy(deep=True)

# ------------------------------------------------------------
# 2. Educational attainment
# Never attended education = 0 completed classes
# ------------------------------------------------------------

education_zero_mask = (
    women_clean["highest_class_completed"].isna()
    & (women_clean["ever_attended_education"] == 0)
)

education_recoded = int(education_zero_mask.sum())

women_clean.loc[
    education_zero_mask,
    "highest_class_completed"
] = 0

# ------------------------------------------------------------
# 3. Land/property value
# No ownership = structural zero
# Unknown ownership remains missing
# ------------------------------------------------------------

land_zero_mask = (
    women_clean["land_property_value"].isna()
    & (women_clean["owns_land_property_nonoperated"] == 0)
)

land_recoded = int(land_zero_mask.sum())

women_clean.loc[
    land_zero_mask,
    "land_property_value"
] = 0

# ------------------------------------------------------------
# 4. Other asset value
# No ownership = structural zero
# Unknown ownership remains missing
# ------------------------------------------------------------

other_zero_mask = (
    women_clean["other_asset_value"].isna()
    & (women_clean["owns_other_assets"] == 0)
)

other_recoded = int(other_zero_mask.sum())

women_clean.loc[
    other_zero_mask,
    "other_asset_value"
] = 0

# ------------------------------------------------------------
# 5. Preserve transport travel-time missingness
# Create indicator BEFORE later imputation
# ------------------------------------------------------------

women_clean["transport_stop_time_min_missing"] = (
    women_clean["transport_stop_time_min"]
    .isna()
    .astype(int)
)

# ------------------------------------------------------------
# 6. Explicit not-applicable/missing category
# -99 is a category, NOT a numeric measurement
# ------------------------------------------------------------

categorical_missing_vars = [
    "earn_emp_status_ag",
    "earn_emp_status_nonag",
    "earn_org_type",
    "transport_stop_halfkm"
]

categorical_recode_counts = {}

for var in categorical_missing_vars:

    categorical_recode_counts[var] = int(
        women_clean[var].isna().sum()
    )

    women_clean[var] = (
        women_clean[var].fillna(-99)
    )

# ------------------------------------------------------------
# 7. Verify outcome and sampling weights unchanged
# ------------------------------------------------------------

assert women_clean.index.equals(women.index)

assert (
    women_clean["low_work_intensity"]
    .equals(women["low_work_intensity"])
)

assert np.allclose(
    women_clean["population_weight"].to_numpy(),
    women["population_weight"].to_numpy()
)

# ------------------------------------------------------------
# 8. Report recoding results
# ------------------------------------------------------------

print("========== STEP 1.7 RESULTS ==========")

print("\nSTRUCTURAL ZEROS:")
print("Education recoded:", education_recoded)
print("Land/property recoded:", land_recoded)
print("Other assets recoded:", other_recoded)

print("\nREMAINING MISSING VALUES:")
for var in [
    "highest_class_completed",
    "land_property_value",
    "other_asset_value",
    "transport_stop_time_min",
    "fies_raw_score",
    "electricity_hours_24h"
]:
    print(
        var + ":",
        int(women_clean[var].isna().sum())
    )

print("\nCATEGORICAL MISSING VALUES RECODED:")
for var, count in categorical_recode_counts.items():
    print(f"{var}: {count}")

print("\nTRANSPORT MISSING INDICATOR:")
print(
    women_clean[
        "transport_stop_time_min_missing"
    ].value_counts().sort_index()
)

# ------------------------------------------------------------
# 9. Final verification
# ------------------------------------------------------------

assert len(women_clean) == 2110

assert education_recoded == 594
assert land_recoded == 2004
assert other_recoded == 1058

assert women_clean[
    "highest_class_completed"
].isna().sum() == 0

assert women_clean[
    "land_property_value"
].isna().sum() == 10

assert women_clean[
    "other_asset_value"
].isna().sum() == 10

assert women_clean[
    "transport_stop_time_min_missing"
].sum() == 833

assert all(
    women_clean[v].notna().all()
    for v in categorical_missing_vars
)

print("\nSTEP 1.7 VERIFICATION SUCCESSFUL.")




# %% STEP 1.8: FINAL DATA CONSISTENCY AUDIT

# ============================================================
# HIES 2022 - Data consistency and plausibility checks
#
# IMPORTANT:
# This step does NOT modify any data.
# It only identifies potential data quality issues.
# ============================================================

import numpy as np
import pandas as pd
import os

print("========== STEP 1.8 RESULTS ==========")

# ------------------------------------------------------------
# 1. Check analytical sample and outcome
# ------------------------------------------------------------

print("\n1. SAMPLE AND OUTCOME VERIFICATION")

print("Analytical observations:", len(women_clean))
print("Unique dataframe indices:", women_clean.index.nunique())

print("Low work intensity cases:",
      int(women_clean["low_work_intensity"].sum()))

print("Not low work intensity cases:",
      int((women_clean["low_work_intensity"] == 0).sum()))

assert len(women_clean) == 2110
assert women_clean.index.is_unique
assert women_clean["low_work_intensity"].sum() == 553

# ------------------------------------------------------------
# 2. Plausibility checks
# ------------------------------------------------------------

print("\n2. NUMERIC PLAUSIBILITY CHECKS")

age = pd.to_numeric(
    women_clean["age"], errors="coerce"
)

hours = pd.to_numeric(
    women_clean["earn_annual_hours"], errors="coerce"
)

education = pd.to_numeric(
    women_clean["highest_class_completed"],
    errors="coerce"
)

plausibility = {
    "Age below 15": int((age < 15).sum()),
    "Age above 110": int((age > 110).sum()),
    "Negative annual hours": int((hours < 0).sum()),
    "Annual hours above 8760": int((hours > 8760).sum()),
    "Negative completed classes": int((education < 0).sum())
}

for check, count in plausibility.items():
    print(f"{check}: {count}")

print("\nObserved numeric ranges:")

for name, values in [
    ("Age", age),
    ("Annual working hours", hours),
    ("Completed classes", education)
]:
    print(
        f"{name}: min={values.min()}, "
        f"median={values.median()}, "
        f"max={values.max()}"
    )

# ------------------------------------------------------------
# 3. Verify structural recoding
# ------------------------------------------------------------

print("\n3. STRUCTURAL RECODING VERIFICATION")

never_school = (
    women_clean["ever_attended_education"] == 0
)

no_land = (
    women_clean["owns_land_property_nonoperated"] == 0
)

no_other_assets = (
    women_clean["owns_other_assets"] == 0
)

checks = {
    "Never attended school with nonzero classes":
        int((
            women_clean.loc[
                never_school,
                "highest_class_completed"
            ] != 0
        ).sum()),

    "Nonowners with nonzero land value":
        int((
            women_clean.loc[
                no_land,
                "land_property_value"
            ] != 0
        ).sum()),

    "Nonowners with nonzero other asset value":
        int((
            women_clean.loc[
                no_other_assets,
                "other_asset_value"
            ] != 0
        ).sum()),

    "Transport missing-indicator mismatches":
        int((
            women_clean[
                "transport_stop_time_min_missing"
            ].to_numpy()
            != women_clean[
                "transport_stop_time_min"
            ].isna().astype(int).to_numpy()
        ).sum())
}

for name, count in checks.items():
    print(f"{name}: {count}")

assert all(count == 0 for count in checks.values())

# ------------------------------------------------------------
# 4. Employment-sector consistency
# ------------------------------------------------------------

print("\n4. EMPLOYMENT SECTOR DIAGNOSTICS")

ag_sector = women_clean["earn_sector"] == 1
nonag_sector = women_clean["earn_sector"] == 2

ag_status_observed = (
    women_clean["earn_emp_status_ag"] != -99
)

nonag_status_observed = (
    women_clean["earn_emp_status_nonag"] != -99
)

print("Agricultural-sector observations:",
      int(ag_sector.sum()))

print("Non-agricultural-sector observations:",
      int(nonag_sector.sum()))

print("Agricultural workers with non-ag status:",
      int((ag_sector & nonag_status_observed).sum()))

print("Non-agricultural workers with ag status:",
      int((nonag_sector & ag_status_observed).sum()))

print("Missing organization type (-99):",
      int((women_clean["earn_org_type"] == -99).sum()))

# These cross-sector observations are diagnostics only.
# Do not delete or recode them without survey documentation.

# ------------------------------------------------------------
# 5. Wealth PCA input audit
# ------------------------------------------------------------

print("\n5. WEALTH PCA INPUT DIAGNOSTICS")

wealth_rows = []

for variable in wealth_vars:

    values = pd.to_numeric(
        women_clean[variable],
        errors="coerce"
    )

    wealth_rows.append({
        "Variable": variable,
        "Missing_N": int(values.isna().sum()),
        "Negative_N": int((values < 0).sum()),
        "Zero_N": int((values == 0).sum()),
        "Minimum": values.min(),
        "Maximum": values.max()
    })

wealth_consistency = pd.DataFrame(wealth_rows)

print(
    wealth_consistency.round(2).to_string(index=False)
)

# ------------------------------------------------------------
# 6. Remaining missing values
# ------------------------------------------------------------

print("\n6. REMAINING MISSINGNESS")

remaining_vars = [
    "fies_raw_score",
    "electricity_hours_24h",
    "transport_stop_time_min",
    "land_property_value",
    "other_asset_value"
]

for variable in remaining_vars:

    print(
        variable + ":",
        int(women_clean[variable].isna().sum())
    )

# ------------------------------------------------------------
# 7. Preserve original outcome and survey weights
# ------------------------------------------------------------

print("\n7. DATA INTEGRITY CHECK")

outcome_unchanged = (
    women_clean["low_work_intensity"]
    .equals(women["low_work_intensity"])
)

weights_unchanged = np.array_equal(
    women_clean["population_weight"].to_numpy(),
    women["population_weight"].to_numpy()
)

print("Outcome unchanged:", outcome_unchanged)
print("Survey weights unchanged:", weights_unchanged)

assert outcome_unchanged
assert weights_unchanged

# ------------------------------------------------------------
# 8. Save audit for research reproducibility
# ------------------------------------------------------------

wealth_consistency.to_csv(
    os.path.join(
        output_dir,
        "Step1_8_Wealth_Consistency_Audit.csv"
    ),
    index=False
)

pd.DataFrame(
    list(plausibility.items()),
    columns=["Check", "Count"]
).to_csv(
    os.path.join(
        output_dir,
        "Step1_8_Plausibility_Audit.csv"
    ),
    index=False
)

# ------------------------------------------------------------
# 9. Final verification
# ------------------------------------------------------------

assert len(wealth_consistency) == 14

print("\nSTEP 1.8 AUDIT COMPLETED.")





# %% STEP 1.9: EXTREME VALUE AND SPECIAL-CODE AUDIT

# ============================================================
# HIES 2022 - Data quality investigation
#
# IMPORTANT:
# This is an audit only.
# No observations or values will be modified.
# ============================================================

import numpy as np
import pandas as pd

print("========== STEP 1.9 RESULTS ==========")

# ------------------------------------------------------------
# 1. Educational attainment distribution
# ------------------------------------------------------------

print("\n1. HIGHEST CLASS COMPLETED")

education = pd.to_numeric(
    women_clean["highest_class_completed"],
    errors="coerce"
)

print("\nAll education codes and frequencies:")

print(
    education.value_counts(dropna=False)
    .sort_index()
    .to_string()
)

print("\nObservations with education > 20:",
      int((education > 20).sum()))

print("Observations with education = 99:",
      int((education == 99).sum()))

print("\nEducation = 99 by school attendance:")

print(
    pd.crosstab(
        women_clean["ever_attended_education"],
        education == 99,
        dropna=False
    ).to_string()
)

# ------------------------------------------------------------
# 2. Household rooms
# ------------------------------------------------------------

print("\n2. HOUSEHOLD ROOMS")

rooms = pd.to_numeric(
    women_clean["hh_rooms"],
    errors="coerce"
)

print("Rooms > 20:", int((rooms > 20).sum()))

print("\nLargest room values:")
print(
    rooms.sort_values(ascending=False)
    .head(10).to_string(index=False)
)

# ------------------------------------------------------------
# 3. Durable goods
# ------------------------------------------------------------

print("\n3. DURABLE GOODS")

durables = pd.to_numeric(
    women_clean["durable_total_number"],
    errors="coerce"
)

print("Total durable items > 100:",
      int((durables > 100).sum()))

print("\nLargest durable-item counts:")
print(
    durables.sort_values(ascending=False)
    .head(10).to_string(index=False)
)

# ------------------------------------------------------------
# 4. Very high age and annual working hours
# ------------------------------------------------------------

print("\n4. AGE AND ANNUAL WORKING HOURS")

age = pd.to_numeric(
    women_clean["age"],
    errors="coerce"
)

hours = pd.to_numeric(
    women_clean["earn_annual_hours"],
    errors="coerce"
)

print("Age >= 90:", int((age >= 90).sum()))

print("Age = 99:", int((age == 99).sum()))

print("Annual working hours > 5000:",
      int((hours > 5000).sum()))

print("\nAnnual working hours quantiles:")

print(
    hours.quantile(
        [0.50, 0.90, 0.95, 0.99, 1.00]
    ).to_string()
)

# ------------------------------------------------------------
# 5. Show records with potentially unusual values
# No personal identifiers printed
# ------------------------------------------------------------

flagged = (
    (education > 20)
    | (rooms > 20)
    | (durables > 100)
    | (age >= 90)
    | (hours > 5000)
)

review_columns = [
    "age",
    "ever_attended_education",
    "highest_class_completed",
    "hh_rooms",
    "covered_area",
    "durable_owned_count",
    "durable_total_number",
    "earn_annual_hours"
]

print("\n5. FLAGGED OBSERVATIONS")

print("Total flagged observations:",
      int(flagged.sum()))

print("\nFirst 25 flagged records:")

print(
    women_clean.loc[
        flagged, review_columns
    ].head(25).to_string(index=False)
)

# ------------------------------------------------------------
# 6. Verify no data have been modified
# ------------------------------------------------------------

assert len(women_clean) == 2110

assert (
    women_clean["low_work_intensity"]
    .equals(women["low_work_intensity"])
)

print("\nSTEP 1.9 AUDIT COMPLETED.")
print("No values were modified.")







# %% STEP 1.10: VARIABLE LABELS AND EXTREME-VALUE VERIFICATION

# ============================================================
# HIES 2022 - Investigate potential special codes
# and extreme household asset values
#
# IMPORTANT: AUDIT ONLY - NO DATA MODIFICATION
# ============================================================

import numpy as np
import pandas as pd

print("========== STEP 1.10 RESULTS ==========")

# ------------------------------------------------------------
# 1. Read Stata variable labels
# ------------------------------------------------------------

variables_to_check = [
    "highest_class_completed",
    "ever_attended_education",
    "age",
    "hh_rooms",
    "durable_owned_count",
    "durable_total_number",
    "durable_total_value",
    "earn_annual_hours"
]

with pd.io.stata.StataReader(file_path) as reader:
    stata_labels = reader.variable_labels()
    stata_value_labels = reader.value_labels()

print("\n1. ORIGINAL STATA VARIABLE LABELS")

for variable in variables_to_check:
    print(
        variable, ":",
        stata_labels.get(variable, "No label available")
    )

# Print potentially relevant value-label definitions
print("\nPOTENTIALLY RELEVANT VALUE-LABEL SETS")

matched = 0

for label_name, mapping in stata_value_labels.items():

    name = str(label_name).lower()

    if any(
        keyword in name
        for keyword in ["educ", "class", "grade", "age"]
    ):
        print("\n", label_name, ":", mapping)
        matched += 1

print("Matching label sets found:", matched)

# ------------------------------------------------------------
# 2. Investigate education code 99
# ------------------------------------------------------------

print("\n2. EDUCATION CODE 99")

education_99 = women_clean.loc[
    women_clean["highest_class_completed"] == 99,
    [
        "age",
        "ever_attended_education",
        "literate_basic",
        "highest_class_completed",
        "earn_annual_hours"
    ]
]

print("Education = 99 cases:", len(education_99))
print(education_99.to_string(index=False))

print(
    "Education = 99 in full dataset:",
    int((df["highest_class_completed"] == 99).sum())
)

# ------------------------------------------------------------
# 3. Investigate age 99
# ------------------------------------------------------------

print("\n3. AGE 99")

age_99 = women_clean.loc[
    women_clean["age"] == 99,
    [
        "age",
        "highest_class_completed",
        "earn_annual_hours",
        "earn_sector"
    ]
]

print(age_99.to_string(index=False))

# ------------------------------------------------------------
# 4. Investigate extreme room counts
# ------------------------------------------------------------

print("\n4. HOUSEHOLD ROOMS")

extreme_rooms = women_clean.loc[
    women_clean["hh_rooms"] > 20,
    [
        "hh_rooms",
        "hh_MEMBER",
        "covered_area",
        "house_value",
        "durable_owned_count",
        "durable_total_number"
    ]
]

print(extreme_rooms.to_string(index=False))

# ------------------------------------------------------------
# 5. Examine durable-count distribution
# ------------------------------------------------------------

print("\n5. DURABLE GOODS DISTRIBUTION")

durable_columns = [
    "durable_owned_count",
    "durable_total_number",
    "durable_total_value"
]

print(
    women_clean[durable_columns]
    .describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )
    .round(2)
    .to_string()
)

# Compare high total numbers with number of asset types
high_durables = women_clean.loc[
    women_clean["durable_total_number"] > 100
]

print("\nHouseholds with total number > 100:",
      len(high_durables))

print(
    "Median owned asset types in that group:",
    high_durables["durable_owned_count"].median()
)

print(
    "Median total number in that group:",
    high_durables["durable_total_number"].median()
)

# ------------------------------------------------------------
# 6. Look for related variables in original dataset
# ------------------------------------------------------------

print("\n6. RELATED DATASET COLUMNS")

related_columns = [
    col for col in df.columns
    if any(
        key in col.lower()
        for key in ["durable", "asset", "hh_rooms"]
    )
]

print("Related columns found:", len(related_columns))

for col in related_columns[:50]:
    print(col)

# ------------------------------------------------------------
# 7. Verify no data changes
# ------------------------------------------------------------

assert len(women_clean) == 2110
assert women_clean["low_work_intensity"].sum() == 553
assert len(education_99) == 5

print("\nSTEP 1.10 AUDIT COMPLETED.")
print("No data values were modified.")





# %% STEP 1.11: EDUCATION CODING AND PCA INPUT PREPARATION

# ============================================================
# HIES 2022 - Questionnaire-informed coding decisions
#
# - Preserve original women_clean
# - Education codes treated as categories
# - Age 99 retained as 99-or-more
# - Isolated room value 83 flagged for PCA only
# - No statistical imputation or model fitting
# ============================================================

import numpy as np
import pandas as pd
import os

# 1. Create the next-stage analytical dataset
women_analysis = women_clean.copy(deep=True)

# 2. Map questionnaire education codes to broad groups
education_map = {
    0: "No class completed",
    **{i: "Primary (1-5)" for i in range(1, 6)},
    **{i: "Lower secondary (6-9)" for i in range(6, 10)},
    10: "SSC or equivalent",
    11: "HSC or equivalent",
    **{i: "Vocational/technical" for i in range(12, 15)},
    **{i: "Tertiary/professional" for i in range(15, 19)},
    99: "Other specified"
}

education_codes = pd.to_numeric(
    women_analysis["highest_class_completed"],
    errors="coerce"
).astype("Int64")

unknown_codes = sorted(
    set(education_codes.dropna().astype(int))
    - set(education_map)
)

assert not unknown_codes, (
    f"Unmapped education codes: {unknown_codes}"
)

group_order = [
    "No class completed",
    "Primary (1-5)",
    "Lower secondary (6-9)",
    "SSC or equivalent",
    "HSC or equivalent",
    "Vocational/technical",
    "Tertiary/professional",
    "Other specified"
]

women_analysis["education_attainment_group"] = (
    pd.Categorical(
        education_codes.map(education_map),
        categories=group_order
    )
)

# 3. Prepare rooms specifically for future PCA
# Do NOT overwrite the original hh_rooms column
women_analysis["hh_rooms_pca"] = pd.to_numeric(
    women_analysis["hh_rooms"],
    errors="coerce"
)

room_flag = women_analysis["hh_rooms_pca"] == 83

women_analysis.loc[
    room_flag, "hh_rooms_pca"
] = np.nan

# 4. Preserve the age 99 top-code
age_topcoded = (
    pd.to_numeric(
        women_analysis["age"],
        errors="coerce"
    ) == 99
)

# 5. Education descriptive distribution
education_summary = (
    women_analysis.groupby(
        "education_attainment_group",
        observed=True
    )
    .agg(
        N=("low_work_intensity", "size"),
        Weighted_Total=("population_weight", "sum")
    )
    .reindex(group_order, fill_value=0)
)

education_summary["Unweighted_Percent"] = (
    education_summary["N"]
    / len(women_analysis) * 100
)

education_summary["Weighted_Percent"] = (
    education_summary["Weighted_Total"]
    / education_summary["Weighted_Total"].sum()
    * 100
)

# 6. Print results
print("========== STEP 1.11 RESULTS ==========")

print("\nEDUCATIONAL ATTAINMENT:")
print(
    education_summary[
        ["N", "Unweighted_Percent", "Weighted_Percent"]
    ].round(2).to_string()
)

print("\nSPECIAL CODE VERIFICATION:")
print("Education code 99 retained as Other:",
      int((education_codes == 99).sum()))

print("Age 99-or-more cases retained:",
      int(age_topcoded.sum()))

print("Original room value 83 cases:",
      int(room_flag.sum()))

print("Rooms missing in PCA input:",
      int(women_analysis["hh_rooms_pca"].isna().sum()))

print("Original room value still present:",
      int((women_analysis["hh_rooms"] == 83).sum()))

print("\nDATA INTEGRITY:")
print("Analytical observations:", len(women_analysis))
print("Outcome cases:",
      int(women_analysis["low_work_intensity"].sum()))

# 7. Verify all coding changes
assert len(women_analysis) == 2110
assert len(education_map) == 20
assert women_analysis[
    "education_attainment_group"
].notna().all()

assert int((education_codes == 99).sum()) == 5
assert int(age_topcoded.sum()) == 2
assert int(room_flag.sum()) == 1

assert education_summary["N"].sum() == 2110

assert women_analysis[
    "low_work_intensity"
].equals(women_clean["low_work_intensity"])

assert women_analysis[
    "population_weight"
].equals(women_clean["population_weight"])

# 8. Save coding audit
education_summary.to_csv(
    os.path.join(
        output_dir,
        "Step1_11_Education_Group_Distribution.csv"
    )
)

print("\nSTEP 1.11 VERIFICATION SUCCESSFUL.")




# %% STEP 1.12: EDUCATION AND WORK-INTENSITY DESCRIPTIVES

# ============================================================
# HIES 2022 - Educational attainment by work intensity
#
# Produces:
# - Unweighted frequency and prevalence
# - Survey-weighted group share
# - Survey-weighted low-work-intensity prevalence
#
# No modelling, imputation or data modification
# ============================================================

import os
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 1. Prepare descriptive analysis data
# ------------------------------------------------------------

education_data = women_analysis[
    [
        "education_attainment_group",
        "low_work_intensity",
        "population_weight"
    ]
].copy()

assert education_data.notna().all().all()
assert len(education_data) == 2110

education_data["population_weight"] = (
    education_data["population_weight"].astype(float)
)

education_data["weighted_low"] = (
    education_data["population_weight"]
    * education_data["low_work_intensity"]
)

# ------------------------------------------------------------
# 2. Summarize by education category
# ------------------------------------------------------------

education_table = (
    education_data
    .groupby(
        "education_attainment_group",
        observed=True
    )
    .agg(
        Total_N=("low_work_intensity", "size"),
        Low_N=("low_work_intensity", "sum"),
        Weighted_Total=("population_weight", "sum"),
        Weighted_Low=("weighted_low", "sum")
    )
    .reindex(group_order, fill_value=0)
)

education_table["Not_Low_N"] = (
    education_table["Total_N"]
    - education_table["Low_N"]
)

# ------------------------------------------------------------
# 3. Calculate unweighted and survey-weighted rates
# ------------------------------------------------------------

education_table["Unweighted_Low_Percent"] = (
    100
    * education_table["Low_N"]
    / education_table["Total_N"]
)

education_table["Weighted_Low_Percent"] = (
    100
    * education_table["Weighted_Low"]
    / education_table["Weighted_Total"]
)

# Weighted distribution of educational groups
education_table["Weighted_Group_Percent"] = (
    100
    * education_table["Weighted_Total"]
    / education_table["Weighted_Total"].sum()
)

# ------------------------------------------------------------
# 4. Flag small groups
# ------------------------------------------------------------

education_table["Small_Group_N_lt_30"] = (
    education_table["Total_N"] < 30
)

# ------------------------------------------------------------
# 5. Print results
# ------------------------------------------------------------

print("========== STEP 1.12 RESULTS ==========")

print("\nEDUCATION AND LOW WORK INTENSITY:")

display_columns = [
    "Total_N",
    "Not_Low_N",
    "Low_N",
    "Unweighted_Low_Percent",
    "Weighted_Low_Percent",
    "Weighted_Group_Percent"
]

print(
    education_table[display_columns]
    .round(2)
    .to_string()
)

# ------------------------------------------------------------
# 6. Verify totals and weighted prevalence
# ------------------------------------------------------------

overall_weighted_low = (
    100
    * education_table["Weighted_Low"].sum()
    / education_table["Weighted_Total"].sum()
)

print("\nVERIFICATION:")

print("Total observations:",
      int(education_table["Total_N"].sum()))

print("Low-work-intensity cases:",
      int(education_table["Low_N"].sum()))

print("Weighted group percentages sum:",
      round(
          education_table["Weighted_Group_Percent"].sum(),
          4
      ))

print("Overall weighted low prevalence (%):",
      round(overall_weighted_low, 4))

print("\nSmall education groups (N < 30):")

print(
    education_table.loc[
        education_table["Small_Group_N_lt_30"],
        ["Total_N", "Low_N"]
    ].to_string()
)

# ------------------------------------------------------------
# 7. Validation checks
# ------------------------------------------------------------

assert education_table["Total_N"].sum() == 2110
assert education_table["Low_N"].sum() == 553

assert np.isclose(
    overall_weighted_low,
    weighted_prevalence_64 * 100,
    atol=1e-8
)

assert np.isclose(
    education_table["Weighted_Group_Percent"].sum(),
    100,
    atol=1e-8
)

# ------------------------------------------------------------
# 8. Save reproducible descriptive table
# ------------------------------------------------------------

education_table.to_csv(
    os.path.join(
        output_dir,
        "Step1_12_Education_Work_Intensity.csv"
    )
)

print("\nSTEP 1.12 VERIFICATION SUCCESSFUL.")





# %% STEP 1.13: CONTINUOUS DESCRIPTIVE STATISTICS

# ============================================================
# HIES 2022 - Continuous descriptive characteristics
#
# Report:
# - Overall, Not Low, and Low work-intensity groups
# - Unweighted mean, SD, median and IQR
# - Survey-weighted descriptive mean and SD
# - Missing observations
#
# No imputation, hypothesis testing or modelling
# ============================================================

import os
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 1. Define continuous descriptive variables
# ------------------------------------------------------------

continuous_descriptive_vars = {
    "Age (years)": "age",
    "Household size": "hh_MEMBER",
    "FIES food insecurity score": "fies_raw_score",
    "Electricity hours (last 24h)": "electricity_hours_24h",
    "Transport stop time (minutes)": "transport_stop_time_min",
    "Annual working hours": "earn_annual_hours"
}

# ------------------------------------------------------------
# 2. Verify required columns
# ------------------------------------------------------------

required = (
    list(continuous_descriptive_vars.values())
    + ["low_work_intensity", "population_weight"]
)

assert all(
    col in women_analysis.columns
    for col in required
)

# ------------------------------------------------------------
# 3. Define survey weights and outcome
# ------------------------------------------------------------

weights = pd.to_numeric(
    women_analysis["population_weight"],
    errors="coerce"
).astype(float)

outcome = women_analysis[
    "low_work_intensity"
].astype(int)

assert np.isfinite(weights).all()
assert (weights > 0).all()
assert outcome.isin([0, 1]).all()

# ------------------------------------------------------------
# 4. Calculate statistics for each variable and group
# ------------------------------------------------------------

results = []

groups = {
    "Overall": outcome.isin([0, 1]),
    "Not low": outcome == 0,
    "Low": outcome == 1
}

for variable_label, variable_name in (
    continuous_descriptive_vars.items()
):

    values = pd.to_numeric(
        women_analysis[variable_name],
        errors="coerce"
    ).astype(float)

    valid_values = (
        values.notna()
        & np.isfinite(values)
    )

    for group_name, group_mask in groups.items():

        # Number of people in the group
        total_n = int(group_mask.sum())

        # Observed numeric values only
        observed_mask = group_mask & valid_values

        x = values.loc[
            observed_mask
        ].to_numpy(dtype=float)

        w = weights.loc[
            observed_mask
        ].to_numpy(dtype=float)

        observed_n = len(x)
        missing_n = total_n - observed_n

        assert observed_n > 1

        # Unweighted descriptive statistics
        unweighted_mean = np.mean(x)
        unweighted_sd = np.std(x, ddof=1)

        median = np.median(x)
        q1 = np.quantile(x, 0.25)
        q3 = np.quantile(x, 0.75)

        # Survey-weighted descriptive mean
        weighted_mean = np.average(
            x,
            weights=w
        )

        # Weighted descriptive SD
        # Not a survey-design-adjusted standard error
        weighted_sd = np.sqrt(
            np.average(
                (x - weighted_mean) ** 2,
                weights=w
            )
        )

        results.append({
            "Variable": variable_label,
            "Group": group_name,
            "Total_N": total_n,
            "Observed_N": observed_n,
            "Missing_N": missing_n,
            "Unweighted_Mean": unweighted_mean,
            "Unweighted_SD": unweighted_sd,
            "Median": median,
            "Q1": q1,
            "Q3": q3,
            "Weighted_Mean": weighted_mean,
            "Weighted_SD": weighted_sd
        })

# ------------------------------------------------------------
# 5. Create final descriptive table
# ------------------------------------------------------------

continuous_table = pd.DataFrame(results)

# ------------------------------------------------------------
# 6. Print results
# ------------------------------------------------------------

print("========== STEP 1.13 RESULTS ==========")

display_cols = [
    "Variable",
    "Group",
    "Observed_N",
    "Missing_N",
    "Unweighted_Mean",
    "Unweighted_SD",
    "Median",
    "Q1",
    "Q3",
    "Weighted_Mean",
    "Weighted_SD"
]

print("\nCONTINUOUS DESCRIPTIVE STATISTICS:")

print(
    continuous_table[display_cols]
    .round(2)
    .to_string(index=False)
)

# ------------------------------------------------------------
# 7. Verify completeness
# ------------------------------------------------------------

print("\n========== VERIFICATION ==========")

print("Analytical observations:",
      len(women_analysis))

print("Low-work-intensity cases:",
      int(outcome.sum()))

print("Continuous variables:",
      len(continuous_descriptive_vars))

print("Total summary rows:",
      len(continuous_table))

assert len(women_analysis) == 2110
assert outcome.sum() == 553

assert len(continuous_table) == (
    len(continuous_descriptive_vars) * 3
)

# Check group totals for every variable
for variable in continuous_descriptive_vars:

    subset = continuous_table.loc[
        continuous_table["Variable"] == variable
    ]

    assert subset.loc[
        subset["Group"] == "Overall",
        "Total_N"
    ].iloc[0] == 2110

    assert (
        subset["Observed_N"]
        + subset["Missing_N"]
        == subset["Total_N"]
    ).all()

# ------------------------------------------------------------
# 8. Save results
# ------------------------------------------------------------

continuous_table.to_csv(
    os.path.join(
        output_dir,
        "Step1_13_Continuous_Descriptive_Statistics.csv"
    ),
    index=False
)

print("\nSTEP 1.13 VERIFICATION SUCCESSFUL.")




# %% STEP 1.14: SURVEY-WEIGHTED CATEGORICAL DESCRIPTIVES

# ============================================================
# HIES 2022 - Categorical characteristics by work intensity
#
# Outputs:
# 1. Total N, Not Low N, Low N
# 2. Unweighted low-work-intensity percentage
# 3. Survey-weighted low-work-intensity percentage
# 4. Survey-weighted category share
# 5. Sparse-category identification
#
# NO imputation or hypothesis testing in this step
# ============================================================

import numpy as np
import pandas as pd
import os

# ------------------------------------------------------------
# 1. Define descriptive categorical variables
# ------------------------------------------------------------

categorical_descriptive_vars = {
    "Residence": "rural_urban",
    "Administrative division": "division_code",
    "Marital status": "marital_status",
    "Basic literacy": "literate_basic",
    "Educational attainment": "education_attainment_group",
    "Bank account": "has_bank_account",
    "MFS account": "has_mfs_account",
    "Internet use": "internet_used",
    "Functional difficulty": "any_functional_difficulty",
    "Employment sector": "earn_sector",
    "Non-agricultural employment status":
        "earn_emp_status_nonag",
    "Organization/workplace type": "earn_org_type"
}

# ------------------------------------------------------------
# 2. Verify variables
# ------------------------------------------------------------

assert all(
    variable in women_analysis.columns
    for variable in categorical_descriptive_vars.values()
)

assert len(women_analysis) == 2110

# ------------------------------------------------------------
# 3. Create descriptive statistics
# ------------------------------------------------------------

all_results = []

for label, variable in categorical_descriptive_vars.items():

    temp = women_analysis[
        [
            variable,
            "low_work_intensity",
            "population_weight"
        ]
    ].copy()

    # Represent categories consistently.
    # The original numeric codes are preserved.
    temp["Category"] = (
        temp[variable]
        .astype("string")
        .fillna("<MISSING>")
    )

    temp["Weight"] = pd.to_numeric(
        temp["population_weight"],
        errors="coerce"
    ).astype(float)

    temp["Weighted_Low"] = (
        temp["Weight"]
        * temp["low_work_intensity"]
    )

    # Category-level totals
    summary = (
        temp.groupby("Category", observed=True)
        .agg(
            Total_N=("low_work_intensity", "size"),
            Low_N=("low_work_intensity", "sum"),
            Weighted_Total=("Weight", "sum"),
            Weighted_Low=("Weighted_Low", "sum")
        )
        .reset_index()
    )

    summary["Not_Low_N"] = (
        summary["Total_N"] - summary["Low_N"]
    )

    # Unweighted prevalence within each category
    summary["Unweighted_Low_Percent"] = (
        100 * summary["Low_N"] / summary["Total_N"]
    )

    # Weighted prevalence within each category
    summary["Weighted_Low_Percent"] = (
        100
        * summary["Weighted_Low"]
        / summary["Weighted_Total"]
    )

    # Weighted share of entire analytical sample
    summary["Weighted_Group_Percent"] = (
        100
        * summary["Weighted_Total"]
        / temp["Weight"].sum()
    )

    # Sparse-category and zero-cell flags
    summary["Sparse_N_lt_30"] = (
        summary["Total_N"] < 30
    )

    summary["Zero_Outcome_Cell"] = (
        (summary["Low_N"] == 0)
        | (summary["Not_Low_N"] == 0)
    )

    summary.insert(0, "Variable", label)

    # Check complete category coverage
    assert summary["Total_N"].sum() == 2110
    assert summary["Low_N"].sum() == 553

    assert np.isclose(
        summary["Weighted_Group_Percent"].sum(),
        100
    )

    all_results.append(summary)

# ------------------------------------------------------------
# 4. Combine results
# ------------------------------------------------------------

categorical_table = pd.concat(
    all_results,
    ignore_index=True
)

# ------------------------------------------------------------
# 5. Print tables
# ------------------------------------------------------------

print("========== STEP 1.14 RESULTS ==========")

for label in categorical_descriptive_vars:

    subset = categorical_table.loc[
        categorical_table["Variable"] == label
    ]

    print("\n" + "=" * 65)
    print(label.upper())
    print("=" * 65)

    print(
        subset[
            [
                "Category",
                "Total_N",
                "Not_Low_N",
                "Low_N",
                "Unweighted_Low_Percent",
                "Weighted_Low_Percent"
            ]
        ].round(2).to_string(index=False)
    )

# ------------------------------------------------------------
# 6. Identify sparse categories
# ------------------------------------------------------------

sparse_categories = categorical_table.loc[
    categorical_table["Sparse_N_lt_30"]
].copy()

zero_cell_categories = categorical_table.loc[
    categorical_table["Zero_Outcome_Cell"]
].copy()

print("\n========== SPARSE CATEGORY AUDIT ==========")

print("Number of sparse category rows:",
      len(sparse_categories))

print("Number of zero-outcome-cell rows:",
      len(zero_cell_categories))

print("\nSparse categories:")

print(
    sparse_categories[
        [
            "Variable",
            "Category",
            "Total_N",
            "Low_N"
        ]
    ].to_string(index=False)
)

# ------------------------------------------------------------
# 7. Verify weighted prevalence by each variable
# ------------------------------------------------------------

overall_weighted_prevalence = np.average(
    women_analysis["low_work_intensity"],
    weights=women_analysis["population_weight"]
) * 100

print("\n========== VERIFICATION ==========")

for label in categorical_descriptive_vars:

    subset = categorical_table.loc[
        categorical_table["Variable"] == label
    ]

    recovered_prevalence = (
        100
        * subset["Weighted_Low"].sum()
        / subset["Weighted_Total"].sum()
    )

    assert np.isclose(
        recovered_prevalence,
        overall_weighted_prevalence
    )

print("Analytical sample:", len(women_analysis))
print("Low cases:",
      int(women_analysis["low_work_intensity"].sum()))

print("Categorical variables:",
      len(categorical_descriptive_vars))

print("Overall weighted low prevalence (%):",
      round(overall_weighted_prevalence, 4))

# ------------------------------------------------------------
# 8. Save outputs
# ------------------------------------------------------------

categorical_table.to_csv(
    os.path.join(
        output_dir,
        "Step1_14_Categorical_Descriptive_Table.csv"
    ),
    index=False
)

sparse_categories.to_csv(
    os.path.join(
        output_dir,
        "Step1_14_Sparse_Category_Audit.csv"
    ),
    index=False
)

print("\nSTEP 1.14 VERIFICATION SUCCESSFUL.")




# %% STEP 1.15: VERIFY ORIGINAL CATEGORICAL VALUE LABELS

# ============================================================
# HIES 2022 - Categorical coding verification
#
# Read attached Stata value labels where available.
# No recoding, imputation, or modelling.
# ============================================================

import pandas as pd
import os

variables_to_verify = [
    "highest_class_completed",
    "earn_org_type",
    "earn_emp_status_nonag",
    "earn_sector",
    "marital_status",
    "division_code",
    "rural_urban"
]

# 1. Read the original Stata labels
labeled_data = pd.read_stata(
    file_path,
    columns=variables_to_verify,
    convert_categoricals=True
)

# 2. Restrict to the same analytical observations
labeled_women = labeled_data.loc[
    women.index
].copy()

label_audit_results = []

print("========== STEP 1.15 RESULTS ==========")

for variable in variables_to_verify:

    has_stata_labels = isinstance(
        labeled_women[variable].dtype,
        pd.CategoricalDtype
    )

    original_codes = (
        women[variable]
        .astype("string")
        .fillna("<MISSING>")
    )

    displayed_labels = (
        labeled_women[variable]
        .astype("string")
        .fillna("<MISSING>")
    )

    mapping = pd.DataFrame({
        "Code": original_codes,
        "Stata_Label": displayed_labels
    })

    mapping = (
        mapping.groupby(
            ["Code", "Stata_Label"],
            dropna=False
        )
        .size()
        .reset_index(name="N")
    )

    mapping.insert(0, "Variable", variable)

    label_audit_results.append(mapping)

    print("\n" + "=" * 55)
    print("VARIABLE:", variable)
    print("Attached Stata value labels:",
          has_stata_labels)
    print("=" * 55)

    print(mapping[
        ["Code", "Stata_Label", "N"]
    ].to_string(index=False))

# 3. Save all discovered mappings
label_audit = pd.concat(
    label_audit_results,
    ignore_index=True
)

label_audit.to_csv(
    os.path.join(
        output_dir,
        "Step1_15_Category_Label_Audit.csv"
    ),
    index=False
)

# 4. Validate analytical sample coverage
assert len(labeled_women) == 2110
assert len(label_audit_results) == 7

for variable in variables_to_verify:

    assert (
        label_audit.loc[
            label_audit["Variable"] == variable,
            "N"
        ].sum() == 2110
    )

print("\nSTEP 1.15 VERIFICATION SUCCESSFUL.")



















# %% STEP 1.16: VERIFIED EMPLOYMENT CATEGORY LABELS

# ============================================================
# HIES 2022 - Questionnaire-based category labels
#
# Purpose:
# Create publication-ready descriptive labels.
#
# IMPORTANT:
# - Do not modify modelling predictors
# - Preserve original codes and outcome
# - Treat -99 as missing/not applicable
# ============================================================

import os
import numpy as np
import pandas as pd

# 1. Define verified questionnaire code mappings
category_maps = {

    "earn_sector": {
        1: "Agriculture",
        2: "Non-agriculture"
    },

    "earn_emp_status_ag": {
        -99: "Missing/not applicable",
        1: "Day labour",
        2: "Self-employed",
        3: "Employer",
        4: "Employee"
    },

    "earn_emp_status_nonag": {
        -99: "Missing/not applicable",
        1: "Day labour",
        2: "Self-employed",
        3: "Employer",
        4: "Employee"
    },

    "earn_org_type": {
        -99: "Missing/not applicable",
        1: "Government organization",
        2: "Autonomous body",
        3: "Private office/institution",
        4: "Public mill/factory",
        5: "Private mill/factory",
        6: "Local government",
        7: "NGO",
        8: "Household",
        9: "Other specified"
    }
}

# 2. Prepare descriptive data only
all_labeled_tables = []

for variable, mapping in category_maps.items():

    codes = pd.to_numeric(
        women_analysis[variable],
        errors="coerce"
    ).astype("Int64")

    labels = codes.map(mapping)

    # Stop if an unexpected code appears
    unmapped_codes = sorted(
        set(codes[labels.isna()].dropna().astype(int))
    )

    assert not unmapped_codes, (
        f"{variable}: Unmapped codes {unmapped_codes}"
    )

    assert labels.notna().all(), (
        f"{variable}: Missing category labels"
    )

    temp = pd.DataFrame({
        "Category_Code": codes.astype(str),
        "Category_Label": labels,
        "Outcome": women_analysis[
            "low_work_intensity"
        ].to_numpy(),
        "Weight": women_analysis[
            "population_weight"
        ].to_numpy(dtype=float)
    })

    temp["Weighted_Low"] = (
        temp["Outcome"] * temp["Weight"]
    )

    # 3. Calculate category-level results
    summary = (
        temp.groupby(
            ["Category_Code", "Category_Label"]
        )
        .agg(
            Total_N=("Outcome", "size"),
            Low_N=("Outcome", "sum"),
            Weighted_Total=("Weight", "sum"),
            Weighted_Low=("Weighted_Low", "sum")
        )
        .reset_index()
    )

    summary["Not_Low_N"] = (
        summary["Total_N"] - summary["Low_N"]
    )

    summary["Unweighted_Low_Percent"] = (
        100 * summary["Low_N"] / summary["Total_N"]
    )

    summary["Weighted_Low_Percent"] = (
        100 * summary["Weighted_Low"]
        / summary["Weighted_Total"]
    )

    summary["Sparse_N_lt_30"] = (
        summary["Total_N"] < 30
    )

    summary.insert(0, "Variable", variable)

    # 4. Verify category totals
    assert summary["Total_N"].sum() == 2110
    assert summary["Low_N"].sum() == 553

    all_labeled_tables.append(summary)

# 5. Combine descriptive results
verified_label_table = pd.concat(
    all_labeled_tables,
    ignore_index=True
)

# 6. Print results
print("========== STEP 1.16 RESULTS ==========")

for variable in category_maps:

    subset = verified_label_table.loc[
        verified_label_table["Variable"] == variable
    ]

    print("\n" + "=" * 60)
    print(variable.upper())
    print("=" * 60)

    print(
        subset[
            [
                "Category_Code",
                "Category_Label",
                "Total_N",
                "Low_N",
                "Weighted_Low_Percent"
            ]
        ].round(2).to_string(index=False)
    )

# 7. Overall verification
print("\n========== VERIFICATION ==========")

print("Variables with verified labels:",
      len(category_maps))

print("Total category rows:",
      len(verified_label_table))

print("Sparse category rows:",
      int(
          verified_label_table[
              "Sparse_N_lt_30"
          ].sum()
      ))

print("Analytical sample:", len(women_analysis))

print("Outcome cases:",
      int(
          women_analysis[
              "low_work_intensity"
          ].sum()
      ))

# 8. Save descriptive label table
verified_label_table.to_csv(
    os.path.join(
        output_dir,
        "Step1_16_Verified_Employment_Labels.csv"
    ),
    index=False
)

assert len(women_analysis) == 2110

print("\nSTEP 1.16 VERIFICATION SUCCESSFUL.")





# %% STEP 1.17: SECTOR-SPECIFIC EMPLOYMENT DESCRIPTIVES

# ============================================================
# HIES 2022 - Employment status within relevant sectors
#
# Agriculture: earn_sector == 1
# Non-agriculture: earn_sector == 2
#
# No changes to the analytical or modelling dataset.
# ============================================================

import numpy as np
import pandas as pd
import os

sector_definitions = [
    (
        "Agricultural employment status",
        1,
        "earn_emp_status_ag"
    ),
    (
        "Non-agricultural employment status",
        2,
        "earn_emp_status_nonag"
    )
]

sector_results = []

print("========== STEP 1.17 RESULTS ==========")

for section_name, sector_code, variable in sector_definitions:

    # 1. Select observations within the relevant sector
    subset = women_analysis.loc[
        women_analysis["earn_sector"] == sector_code,
        [
            variable,
            "low_work_intensity",
            "population_weight"
        ]
    ].copy()

    # 2. Preserve questionnaire codes
    subset["Category_Code"] = pd.to_numeric(
        subset[variable],
        errors="coerce"
    ).fillna(-99).astype(int)

    # Use category labels defined in Step 1.16
    subset["Category_Label"] = (
        subset["Category_Code"]
        .map(category_maps[variable])
    )

    assert subset["Category_Label"].notna().all()

    # 3. Survey-weighted outcome contribution
    subset["Weighted_Low"] = (
        subset["population_weight"]
        * subset["low_work_intensity"]
    )

    # 4. Category-level summary
    summary = (
        subset.groupby(
            ["Category_Code", "Category_Label"]
        )
        .agg(
            Total_N=("low_work_intensity", "size"),
            Low_N=("low_work_intensity", "sum"),
            Weighted_Total=("population_weight", "sum"),
            Weighted_Low=("Weighted_Low", "sum")
        )
        .reset_index()
    )

    summary["Not_Low_N"] = (
        summary["Total_N"] - summary["Low_N"]
    )

    summary["Unweighted_Low_Percent"] = (
        100 * summary["Low_N"]
        / summary["Total_N"]
    )

    summary["Weighted_Low_Percent"] = (
        100 * summary["Weighted_Low"]
        / summary["Weighted_Total"]
    )

    summary["Sparse_N_lt_30"] = (
        summary["Total_N"] < 30
    )

    summary.insert(0, "Sector", section_name)

    # 5. Verification
    expected_n = 227 if sector_code == 1 else 1883

    assert len(subset) == expected_n
    assert summary["Total_N"].sum() == expected_n

    sector_prevalence = np.average(
        subset["low_work_intensity"],
        weights=subset["population_weight"]
    ) * 100

    recovered_prevalence = (
        summary["Weighted_Low"].sum()
        / summary["Weighted_Total"].sum()
        * 100
    )

    assert np.isclose(
        sector_prevalence,
        recovered_prevalence
    )

    # 6. Display results
    print("\n" + "=" * 60)
    print(section_name.upper())
    print("=" * 60)

    print("Sector observations:", len(subset))
    print("Low-work-intensity cases:",
          int(subset["low_work_intensity"].sum()))

    print("Sector weighted prevalence (%):",
          round(sector_prevalence, 2))

    print("\nEmployment status distribution:")

    print(
        summary[
            [
                "Category_Code",
                "Category_Label",
                "Total_N",
                "Low_N",
                "Unweighted_Low_Percent",
                "Weighted_Low_Percent",
                "Sparse_N_lt_30"
            ]
        ].round(2).to_string(index=False)
    )

    sector_results.append(summary)

# 7. Combine and save results
sector_status_table = pd.concat(
    sector_results,
    ignore_index=True
)

sector_status_table.to_csv(
    os.path.join(
        output_dir,
        "Step1_17_Sector_Specific_Employment_Status.csv"
    ),
    index=False
)

# 8. Verify original data are unchanged
assert len(women_analysis) == 2110
assert women_analysis["low_work_intensity"].sum() == 553

print("\nSTEP 1.17 VERIFICATION SUCCESSFUL.")






# %% STEP 1.18: SURVEY-ADJUSTED BINARY GROUP COMPARISONS

# ============================================================
# HIES 2022 - Design-aware group comparisons
#
# Compares:
# 1. Rural vs Urban
# 2. Agriculture vs Non-agriculture
#
# Method:
# Stratified PSU-level Taylor linearization
# using the full available survey design structure.
#
# No observations modified.
# No model fitting.
# ============================================================

import numpy as np
import pandas as pd
import os
from scipy.stats import t as student_t

# ------------------------------------------------------------
# 1. Preserve complete PSU/stratum information
# ------------------------------------------------------------

full_design = df[
    ["stratum16", "psu"]
].copy()

assert full_design.notna().all().all()
assert df.index.is_unique
assert women_analysis.index.is_unique

n_full_psu = (
    full_design[["stratum16", "psu"]]
    .drop_duplicates()
    .shape[0]
)

n_strata = full_design["stratum16"].nunique()

design_df = n_full_psu - n_strata

print("========== STEP 1.18 RESULTS ==========")
print("Full survey PSU groups:", n_full_psu)
print("Survey strata:", n_strata)
print("Design degrees of freedom:", design_df)

# ------------------------------------------------------------
# 2. Define a survey-adjusted comparison function
# ------------------------------------------------------------

def survey_prevalence_difference(
    variable,
    group_1,
    group_2,
    comparison_name
):

    # Use only the analytical sample to estimate prevalence
    data = women_analysis[
        [
            variable,
            "low_work_intensity",
            "population_weight"
        ]
    ].copy()

    g = data[variable].to_numpy()
    y = data["low_work_intensity"].to_numpy(dtype=float)
    w = data["population_weight"].to_numpy(dtype=float)

    mask_1 = (g == group_1)
    mask_2 = (g == group_2)

    assert mask_1.sum() > 0
    assert mask_2.sum() > 0
    assert not np.any(mask_1 & mask_2)

    # Weighted totals in each group
    total_w1 = w[mask_1].sum()
    total_w2 = w[mask_2].sum()

    # Survey-weighted group prevalences
    p1 = np.sum(w[mask_1] * y[mask_1]) / total_w1
    p2 = np.sum(w[mask_2] * y[mask_2]) / total_w2

    # Group difference: first group minus second group
    difference = p1 - p2

    # --------------------------------------------------------
    # 3. Linearized observation-level contributions
    # --------------------------------------------------------

    contribution = np.zeros(len(data), dtype=float)

    contribution[mask_1] = (
        w[mask_1] * (y[mask_1] - p1) / total_w1
    )

    contribution[mask_2] = (
        -w[mask_2] * (y[mask_2] - p2) / total_w2
    )

    # --------------------------------------------------------
    # 4. Include all available survey PSUs
    # Non-domain observations contribute zero.
    # --------------------------------------------------------

    psu_data = full_design.copy()
    psu_data["Linearized"] = 0.0

    psu_data.loc[
        data.index,
        "Linearized"
    ] = contribution

    psu_totals = (
        psu_data
        .groupby(
            ["stratum16", "psu"],
            observed=True
        )["Linearized"]
        .sum()
        .reset_index()
    )

    # --------------------------------------------------------
    # 5. Stratified PSU-level variance
    # --------------------------------------------------------

    variance = 0.0

    for stratum, part in psu_totals.groupby(
        "stratum16",
        observed=True
    ):

        values = part["Linearized"].to_numpy()

        m = len(values)

        assert m >= 2

        variance += (
            m / (m - 1)
        ) * np.sum(
            (values - values.mean()) ** 2
        )

    standard_error = np.sqrt(variance)

    assert standard_error > 0

    # --------------------------------------------------------
    # 6. Design-adjusted Wald test and 95% CI
    # --------------------------------------------------------

    t_stat = difference / standard_error

    p_value = 2 * student_t.sf(
        abs(t_stat),
        df=design_df
    )

    critical = student_t.ppf(
        0.975,
        df=design_df
    )

    ci_low = difference - critical * standard_error
    ci_high = difference + critical * standard_error

    return {
        "Comparison": comparison_name,
        "Group1_N": int(mask_1.sum()),
        "Group2_N": int(mask_2.sum()),
        "Group1_Weighted_Prevalence": p1 * 100,
        "Group2_Weighted_Prevalence": p2 * 100,
        "Difference_pp": difference * 100,
        "SE_pp": standard_error * 100,
        "CI_Lower_pp": ci_low * 100,
        "CI_Upper_pp": ci_high * 100,
        "t_statistic": t_stat,
        "p_value": p_value
    }


# ------------------------------------------------------------
# 7. Run the two planned comparisons
# ------------------------------------------------------------

comparison_results = [
    survey_prevalence_difference(
        variable="rural_urban",
        group_1=1,
        group_2=2,
        comparison_name="Rural minus Urban"
    ),

    survey_prevalence_difference(
        variable="earn_sector",
        group_1=1,
        group_2=2,
        comparison_name="Agriculture minus Non-agriculture"
    )
]

comparison_table = pd.DataFrame(comparison_results)

# ------------------------------------------------------------
# 8. Print full results
# ------------------------------------------------------------

print("\nSURVEY-ADJUSTED GROUP COMPARISONS:")
print(
    comparison_table.round(5).to_string(index=False)
)

# ------------------------------------------------------------
# 9. Verify against Step 1.14
# ------------------------------------------------------------

assert n_full_psu == 720
assert n_strata == 16
assert design_df == 704

assert comparison_table.iloc[0]["Group1_N"] == 734
assert comparison_table.iloc[0]["Group2_N"] == 1376

assert comparison_table.iloc[1]["Group1_N"] == 227
assert comparison_table.iloc[1]["Group2_N"] == 1883

assert np.isclose(
    comparison_table.iloc[0][
        "Group1_Weighted_Prevalence"
    ],
    32.16,
    atol=0.02
)

assert np.isclose(
    comparison_table.iloc[1][
        "Group1_Weighted_Prevalence"
    ],
    54.46,
    atol=0.02
)

# ------------------------------------------------------------
# 10. Save reproducible results
# ------------------------------------------------------------

comparison_table.to_csv(
    os.path.join(
        output_dir,
        "Step1_18_Survey_Adjusted_Comparisons.csv"
    ),
    index=False
)

print("\nSTEP 1.18 VERIFICATION SUCCESSFUL.")















# %% STEP 1.19: SURVEY-ADJUSTED MULTI-CATEGORY WALD TEST

# ============================================================
# HIES 2022 - Design-aware multi-category comparisons
#
# Tests:
# 1. Administrative division (8 categories)
# 2. Marital status (5 categories)
#
# Method:
# Stratified PSU Taylor linearization + joint Wald F-test
#
# No recoding, imputation, or modelling
# ============================================================

import numpy as np
import pandas as pd
import os
from scipy.stats import f as f_distribution

# 1. Full survey design
design = df[["stratum16", "psu"]].copy()

assert design.notna().all().all()
assert design.index.is_unique
assert women_analysis.index.isin(design.index).all()

n_psu_full = design.drop_duplicates().shape[0]
n_strata_full = design["stratum16"].nunique()
design_df = n_psu_full - n_strata_full

assert n_psu_full == 720
assert n_strata_full == 16
assert design_df == 704


# 2. Reusable multi-category survey test
def survey_multicategory_test(variable):

    data = women_analysis[
        [variable, "low_work_intensity", "population_weight"]
    ].copy()

    g = pd.to_numeric(
        data[variable], errors="coerce"
    ).to_numpy()

    y = data["low_work_intensity"].to_numpy(
        dtype=float
    )

    w = data["population_weight"].to_numpy(
        dtype=float
    )

    assert np.isfinite(g).all()
    assert np.isfinite(w).all()
    assert (w > 0).all()

    categories = sorted(np.unique(g))
    k = len(categories)

    assert k >= 3

    prevalence = np.zeros(k)
    contributions = np.zeros((len(data), k))

    rows = []

    # 3. Weighted prevalence and linearized
    # contributions for each category
    for j, category in enumerate(categories):

        mask = g == category

        weight_total = w[mask].sum()

        prevalence[j] = (
            np.sum(w[mask] * y[mask]) / weight_total
        )

        contributions[mask, j] = (
            w[mask]
            * (y[mask] - prevalence[j])
            / weight_total
        )

        rows.append({
            "Variable": variable,
            "Category_Code": category,
            "Total_N": int(mask.sum()),
            "Low_N": int(y[mask].sum()),
            "Weighted_Low_Percent": prevalence[j] * 100
        })

    # 4. Include zero contributions from all
    # non-domain survey observations
    linearized_cols = [
        f"linearized_{j}" for j in range(k)
    ]

    linearized_full = pd.DataFrame(
        0.0,
        index=design.index,
        columns=linearized_cols
    )

    linearized_full.loc[
        data.index, linearized_cols
    ] = contributions

    full_contributions = pd.concat(
        [design, linearized_full],
        axis=1
    )

    # 5. Aggregate at PSU level within strata
    psu_totals = (
        full_contributions
        .groupby(
            ["stratum16", "psu"],
            observed=True
        )[linearized_cols]
        .sum()
    )

    # 6. Covariance matrix
    covariance = np.zeros((k, k))

    for stratum, part in psu_totals.groupby(
        level=0, sort=True
    ):

        values = part.to_numpy(dtype=float)

        m = len(values)

        assert m >= 2

        centered = (
            values - values.mean(axis=0, keepdims=True)
        )

        covariance += (
            m / (m - 1)
        ) * (centered.T @ centered)

    # 7. Test equality of all category prevalences
    # Reference is the first sorted category
    contrast = np.zeros((k - 1, k))

    for j in range(1, k):
        contrast[j - 1, 0] = -1
        contrast[j - 1, j] = 1

    differences = contrast @ prevalence

    contrast_covariance = (
        contrast @ covariance @ contrast.T
    )

    assert np.linalg.matrix_rank(
        contrast_covariance
    ) == k - 1, "Singular contrast covariance matrix."

    # Joint Wald F-statistic
    f_statistic = (
        differences
        @ np.linalg.solve(
            contrast_covariance, differences
        )
        / (k - 1)
    )

    df1 = k - 1
    df2 = design_df

    p_value = f_distribution.sf(
        f_statistic, df1, df2
    )

    category_table = pd.DataFrame(rows)

    test_table = pd.DataFrame([{
        "Variable": variable,
        "Categories": k,
        "Wald_F": f_statistic,
        "DF1": df1,
        "DF2": df2,
        "P_Value": p_value
    }])

    return category_table, test_table


# 8. Run planned tests
variables_to_test = [
    "division_code",
    "marital_status"
]

category_results = []
test_results = []

print("========== STEP 1.19 RESULTS ==========")

for variable in variables_to_test:

    category_table, test_table = (
        survey_multicategory_test(variable)
    )

    category_results.append(category_table)
    test_results.append(test_table)

    print("\n" + "=" * 60)
    print("VARIABLE:", variable)
    print("=" * 60)

    print("\nCategory prevalences:")
    print(
        category_table.round(3).to_string(index=False)
    )

    print("\nSurvey-adjusted Wald F-test:")
    print(
        "F =", round(test_table["Wald_F"].iloc[0], 4)
    )
    print(
        "Degrees of freedom:",
        int(test_table["DF1"].iloc[0]),
        ",",
        int(test_table["DF2"].iloc[0])
    )
    print(
        "p-value:",
        f"{test_table['P_Value'].iloc[0]:.6e}"
    )


# 9. Combine results
multicategory_prevalence = pd.concat(
    category_results,
    ignore_index=True
)

multicategory_tests = pd.concat(
    test_results,
    ignore_index=True
)

# 10. Verify
assert len(multicategory_prevalence) == 13
assert len(multicategory_tests) == 2

for variable in variables_to_test:

    subset = multicategory_prevalence.loc[
        multicategory_prevalence["Variable"] == variable
    ]

    assert subset["Total_N"].sum() == 2110
    assert subset["Low_N"].sum() == 553

assert np.isfinite(
    multicategory_tests["P_Value"]
).all()

assert (
    multicategory_tests["P_Value"].between(0, 1)
).all()

# 11. Save results
multicategory_prevalence.to_csv(
    os.path.join(
        output_dir,
        "Step1_19_Multicategory_Prevalence.csv"
    ),
    index=False
)

multicategory_tests.to_csv(
    os.path.join(
        output_dir,
        "Step1_19_Survey_Adjusted_Wald_Tests.csv"
    ),
    index=False
)

print("\nSTEP 1.19 VERIFICATION SUCCESSFUL.")






# %% STEP 1.20: REMAINING BINARY SURVEY-ADJUSTED TESTS

# ============================================================
# HIES 2022 - Survey-adjusted binary group comparisons
#
# Uses the verified function from Step 1.18:
# survey_prevalence_difference()
#
# Also applies Holm correction to the nine tests
# completed across Steps 1.18, 1.19 and 1.20.
#
# No data modification or model fitting.
# ============================================================

import numpy as np
import pandas as pd
import os

# 1. Define remaining binary comparisons
binary_comparisons = [
    ("literate_basic", 0, 1, "Not literate minus literate"),
    ("has_bank_account", 0, 1, "No bank account minus yes"),
    ("has_mfs_account", 0, 1, "No MFS account minus yes"),
    ("internet_used", 0, 1, "No internet use minus yes"),
    ("any_functional_difficulty", 1, 0,
     "Functional difficulty yes minus no")
]

# 2. Run survey-adjusted comparisons
binary_results = []

for variable, group1, group2, label in binary_comparisons:

    result = survey_prevalence_difference(
        variable=variable,
        group_1=group1,
        group_2=group2,
        comparison_name=label
    )

    binary_results.append(result)

binary_comparison_table = pd.DataFrame(binary_results)

# 3. Print results
print("========== STEP 1.20 RESULTS ==========")

print("\nSURVEY-ADJUSTED BINARY COMPARISONS:")

for _, row in binary_comparison_table.iterrows():

    print("\nComparison:", row["Comparison"])

    print(
        "Weighted prevalence (%):",
        f"{row['Group1_Weighted_Prevalence']:.2f}",
        "vs",
        f"{row['Group2_Weighted_Prevalence']:.2f}"
    )

    print(
        "Difference (percentage points):",
        f"{row['Difference_pp']:.3f}"
    )

    print(
        "95% CI:",
        f"[{row['CI_Lower_pp']:.3f}, "
        f"{row['CI_Upper_pp']:.3f}]"
    )

    print(
        "Unadjusted p-value:",
        f"{row['p_value']:.6e}"
    )

# ------------------------------------------------------------
# 4. Combine all nine tests conducted so far
# ------------------------------------------------------------

previous_binary = comparison_table[
    ["Comparison", "p_value"]
].rename(
    columns={"p_value": "P_Value"}
)

new_binary = binary_comparison_table[
    ["Comparison", "p_value"]
].rename(
    columns={"p_value": "P_Value"}
)

previous_multi = multicategory_tests[
    ["Variable", "P_Value"]
].rename(
    columns={"Variable": "Comparison"}
)

all_tests = pd.concat(
    [
        previous_binary,
        previous_multi,
        new_binary
    ],
    ignore_index=True
)

assert len(all_tests) == 9

# ------------------------------------------------------------
# 5. Holm multiple-testing correction
# ------------------------------------------------------------

p = all_tests["P_Value"].to_numpy(dtype=float)

assert np.isfinite(p).all()
assert np.all((p >= 0) & (p <= 1))

m = len(p)

order = np.argsort(p)

sorted_p = p[order]

holm_sorted = np.maximum.accumulate(
    sorted_p * (m - np.arange(m))
)

holm_sorted = np.minimum(
    holm_sorted, 1.0
)

holm_adjusted = np.empty(m, dtype=float)

holm_adjusted[order] = holm_sorted

all_tests["Holm_Adjusted_P"] = holm_adjusted

all_tests["Significant_After_Holm"] = (
    all_tests["Holm_Adjusted_P"] < 0.05
)

# 6. Print adjusted results
print("\n========== NINE-TEST HOLM CORRECTION ==========")

print(
    all_tests.sort_values("P_Value")
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.6e}"
    )
)

# ------------------------------------------------------------
# 7. Save reproducible outputs
# ------------------------------------------------------------

binary_comparison_table.to_csv(
    os.path.join(
        output_dir,
        "Step1_20_Binary_Survey_Adjusted_Tests.csv"
    ),
    index=False
)

all_tests.to_csv(
    os.path.join(
        output_dir,
        "Step1_20_Nine_Test_Holm_Correction.csv"
    ),
    index=False
)

# ------------------------------------------------------------
# 8. Final verification
# ------------------------------------------------------------

assert len(binary_comparison_table) == 5
assert len(all_tests) == 9

assert all_tests[
    "Holm_Adjusted_P"
].between(0, 1).all()

assert (
    all_tests["Holm_Adjusted_P"]
    >= all_tests["P_Value"] - 1e-12
).all()

print("\nSTEP 1.20 VERIFICATION SUCCESSFUL.")

















# %% STEP 1.21: SURVEY-ADJUSTED CONTINUOUS GROUP COMPARISONS

# ============================================================
# HIES 2022 - Design-aware weighted mean comparisons
#
# Method:
# Stratified PSU-level Taylor linearization
#
# Mean difference:
# Low work intensity MINUS Not low work intensity
#
# Uses the full available PSU-stratum structure.
# No imputation, feature selection, or data modification.
# ============================================================

import os
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

# ------------------------------------------------------------
# 1. Define the continuous variables to compare
# ------------------------------------------------------------

continuous_tests = {
    "Age (years)": "age",
    "Household size": "hh_MEMBER",
    "FIES food insecurity score": "fies_raw_score",
    "Electricity hours (last 24h)": "electricity_hours_24h",
    "Transport stop time (minutes)": "transport_stop_time_min"
}

# ------------------------------------------------------------
# 2. Preserve the complete survey design
# ------------------------------------------------------------

full_survey_design = df[
    ["stratum16", "psu"]
].copy()

assert full_survey_design.notna().all().all()
assert full_survey_design.index.is_unique
assert women_analysis.index.isin(
    full_survey_design.index
).all()

n_full_psu = (
    full_survey_design
    .drop_duplicates()
    .shape[0]
)

n_strata = (
    full_survey_design["stratum16"].nunique()
)

survey_df = n_full_psu - n_strata

assert n_full_psu == 720
assert n_strata == 16
assert survey_df == 704

# ------------------------------------------------------------
# 3. Function for survey-weighted mean differences
# ------------------------------------------------------------

def survey_weighted_mean_comparison(
    variable_name,
    variable_label
):

    data = women_analysis[
        [
            variable_name,
            "low_work_intensity",
            "population_weight"
        ]
    ].copy()

    # Convert numeric variables safely
    data["X"] = pd.to_numeric(
        data[variable_name],
        errors="coerce"
    )

    data["Weight"] = pd.to_numeric(
        data["population_weight"],
        errors="coerce"
    )

    # Complete-case domain for this variable
    valid = (
        np.isfinite(data["X"])
        & np.isfinite(data["Weight"])
        & (data["Weight"] > 0)
    )

    observed = data.loc[valid].copy()

    x = observed["X"].to_numpy(dtype=float)

    y = observed[
        "low_work_intensity"
    ].to_numpy(dtype=int)

    w = observed["Weight"].to_numpy(dtype=float)

    low_mask = y == 1
    not_low_mask = y == 0

    assert low_mask.sum() > 1
    assert not_low_mask.sum() > 1

    # --------------------------------------------------------
    # 4. Survey-weighted group means
    # --------------------------------------------------------

    W_low = w[low_mask].sum()
    W_not_low = w[not_low_mask].sum()

    mean_low = np.average(
        x[low_mask],
        weights=w[low_mask]
    )

    mean_not_low = np.average(
        x[not_low_mask],
        weights=w[not_low_mask]
    )

    difference = mean_low - mean_not_low

    # --------------------------------------------------------
    # 5. Linearized contributions for the difference
    # --------------------------------------------------------

    linearized = np.zeros(len(observed), dtype=float)

    linearized[low_mask] = (
        w[low_mask]
        * (x[low_mask] - mean_low)
        / W_low
    )

    linearized[not_low_mask] = (
        -w[not_low_mask]
        * (x[not_low_mask] - mean_not_low)
        / W_not_low
    )

    # --------------------------------------------------------
    # 6. Include every available PSU in the survey
    # Observations outside the observed-data domain
    # contribute zero.
    # --------------------------------------------------------

    full = full_survey_design.copy()

    full["Linearized"] = 0.0

    full.loc[
        observed.index,
        "Linearized"
    ] = linearized

    psu_totals = (
        full.groupby(
            ["stratum16", "psu"],
            observed=True
        )["Linearized"]
        .sum()
        .reset_index()
    )

    # --------------------------------------------------------
    # 7. Stratified PSU-level variance
    # --------------------------------------------------------

    variance = 0.0

    for stratum, part in psu_totals.groupby(
        "stratum16",
        observed=True
    ):

        values = part[
            "Linearized"
        ].to_numpy(dtype=float)

        m = len(values)

        assert m >= 2

        variance += (
            m / (m - 1)
        ) * np.sum(
            (values - values.mean()) ** 2
        )

    standard_error = np.sqrt(variance)

    assert np.isfinite(standard_error)
    assert standard_error > 0

    # --------------------------------------------------------
    # 8. Confidence interval and Wald t-test
    # --------------------------------------------------------

    t_statistic = difference / standard_error

    p_value = 2 * student_t.sf(
        abs(t_statistic),
        df=survey_df
    )

    t_critical = student_t.ppf(
        0.975,
        df=survey_df
    )

    ci_lower = (
        difference - t_critical * standard_error
    )

    ci_upper = (
        difference + t_critical * standard_error
    )

    # --------------------------------------------------------
    # 9. Return all statistics
    # --------------------------------------------------------

    return {
        "Variable": variable_label,
        "Observed_N": len(observed),
        "Missing_N": len(data) - len(observed),
        "Not_Low_N": int(not_low_mask.sum()),
        "Low_N": int(low_mask.sum()),
        "Weighted_Mean_Not_Low": mean_not_low,
        "Weighted_Mean_Low": mean_low,
        "Difference_Low_minus_Not_Low": difference,
        "SE_Difference": standard_error,
        "CI_Lower": ci_lower,
        "CI_Upper": ci_upper,
        "t_statistic": t_statistic,
        "DF": survey_df,
        "P_Value": p_value
    }


# ------------------------------------------------------------
# 10. Run all five continuous comparisons
# ------------------------------------------------------------

continuous_test_results = []

for label, variable in continuous_tests.items():

    result = survey_weighted_mean_comparison(
        variable_name=variable,
        variable_label=label
    )

    continuous_test_results.append(result)

continuous_inference_table = pd.DataFrame(
    continuous_test_results
)

# ------------------------------------------------------------
# 11. Print results
# ------------------------------------------------------------

print("========== STEP 1.21 RESULTS ==========")

print("\nSURVEY-ADJUSTED CONTINUOUS COMPARISONS:")

for _, row in continuous_inference_table.iterrows():

    print("\nVariable:", row["Variable"])

    print(
        "Observed N:",
        int(row["Observed_N"])
    )

    print(
        "Missing N:",
        int(row["Missing_N"])
    )

    print(
        "Weighted mean (Not low):",
        f"{row['Weighted_Mean_Not_Low']:.4f}"
    )

    print(
        "Weighted mean (Low):",
        f"{row['Weighted_Mean_Low']:.4f}"
    )

    print(
        "Difference (Low - Not low):",
        f"{row['Difference_Low_minus_Not_Low']:.4f}"
    )

    print(
        "95% CI:",
        f"[{row['CI_Lower']:.4f}, "
        f"{row['CI_Upper']:.4f}]"
    )

    print(
        "p-value:",
        f"{row['P_Value']:.6e}"
    )

# ------------------------------------------------------------
# 12. Validation
# ------------------------------------------------------------

assert len(continuous_inference_table) == 5

assert continuous_inference_table[
    "P_Value"
].between(0, 1).all()

assert np.isfinite(
    continuous_inference_table[
        "SE_Difference"
    ]
).all()

assert (
    continuous_inference_table["Observed_N"]
    + continuous_inference_table["Missing_N"]
    == 2110
).all()

# Verify means match Step 1.13
for label in continuous_tests:

    row = continuous_inference_table.loc[
        continuous_inference_table["Variable"] == label
    ].iloc[0]

    old = continuous_table.loc[
        continuous_table["Variable"] == label
    ]

    old_not_low = old.loc[
        old["Group"] == "Not low",
        "Weighted_Mean"
    ].iloc[0]

    old_low = old.loc[
        old["Group"] == "Low",
        "Weighted_Mean"
    ].iloc[0]

    assert np.isclose(
        row["Weighted_Mean_Not_Low"],
        old_not_low
    )

    assert np.isclose(
        row["Weighted_Mean_Low"],
        old_low
    )

# ------------------------------------------------------------
# 13. Save reproducible results
# ------------------------------------------------------------

continuous_inference_table.to_csv(
    os.path.join(
        output_dir,
        "Step1_21_Survey_Adjusted_Continuous_Tests.csv"
    ),
    index=False
)

print("\nSTEP 1.21 VERIFICATION SUCCESSFUL.")




# %% STEP 1.22: SURVEY-ADJUSTED EDUCATION OMNIBUS TEST

# ============================================================
# HIES 2022 - Educational attainment and low work intensity
#
# Primary inferential domain:
# Six ordered, sufficiently populated education groups
#
# Excluded ONLY from this specific test:
# - Vocational/technical (N=20)
# - Other specified (N=5)
#
# Method:
# Stratified PSU Taylor linearization
# Joint Wald F-test of equal weighted prevalences
#
# Original analytical data remain unchanged.
# ============================================================

import os
import numpy as np
import pandas as pd
from scipy.stats import f as f_distribution

# ------------------------------------------------------------
# 1. Specify six educational groups
# ------------------------------------------------------------

education_groups_test = [
    "No class completed",
    "Primary (1-5)",
    "Lower secondary (6-9)",
    "SSC or equivalent",
    "HSC or equivalent",
    "Tertiary/professional"
]

# ------------------------------------------------------------
# 2. Select the education-specific inferential domain
# ------------------------------------------------------------

education_test_data = women_analysis.loc[
    women_analysis[
        "education_attainment_group"
    ].isin(education_groups_test),
    [
        "education_attainment_group",
        "low_work_intensity",
        "population_weight"
    ]
].copy()

print("========== STEP 1.22 RESULTS ==========")

print("\nFULL ANALYTICAL SAMPLE:")
print("N =", len(women_analysis))

print("\nEDUCATION TEST DOMAIN:")
print("Included N =", len(education_test_data))

print(
    "Excluded from this test only =",
    len(women_analysis) - len(education_test_data)
)

print(
    "Included low cases =",
    int(
        education_test_data[
            "low_work_intensity"
        ].sum()
    )
)

assert len(women_analysis) == 2110
assert len(education_test_data) == 2085

assert (
    education_test_data[
        "low_work_intensity"
    ].sum() == 550
)

# ------------------------------------------------------------
# 3. Extract outcome and survey weights
# ------------------------------------------------------------

g = (
    education_test_data[
        "education_attainment_group"
    ].astype(str).to_numpy()
)

y = education_test_data[
    "low_work_intensity"
].to_numpy(dtype=float)

w = education_test_data[
    "population_weight"
].to_numpy(dtype=float)

assert np.isfinite(w).all()
assert (w > 0).all()

k = len(education_groups_test)

prevalences = np.zeros(k)

linearized = np.zeros(
    (len(education_test_data), k),
    dtype=float
)

category_results = []

# ------------------------------------------------------------
# 4. Compute prevalence by education category
# ------------------------------------------------------------

for j, category in enumerate(education_groups_test):

    mask = g == category

    category_n = int(mask.sum())
    category_low = int(y[mask].sum())

    assert category_n >= 30

    total_weight = w[mask].sum()

    prevalence = (
        np.sum(w[mask] * y[mask])
        / total_weight
    )

    prevalences[j] = prevalence

    # Linearized contribution to category prevalence
    linearized[mask, j] = (
        w[mask]
        * (y[mask] - prevalence)
        / total_weight
    )

    category_results.append({
        "Education_Group": category,
        "Total_N": category_n,
        "Low_N": category_low,
        "Weighted_Low_Percent": prevalence * 100
    })

education_group_table = pd.DataFrame(
    category_results
)

print("\nEDUCATION CATEGORY PREVALENCE:")

print(
    education_group_table.round(3)
    .to_string(index=False)
)

# ------------------------------------------------------------
# 5. Full survey PSU and stratum structure
# ------------------------------------------------------------

full_education_design = df[
    ["stratum16", "psu"]
].copy()

assert full_education_design.index.is_unique
assert education_test_data.index.isin(
    full_education_design.index
).all()

n_psu = (
    full_education_design.drop_duplicates().shape[0]
)

n_strata = (
    full_education_design["stratum16"].nunique()
)

design_df = n_psu - n_strata

assert n_psu == 720
assert n_strata == 16
assert design_df == 704

# ------------------------------------------------------------
# 6. Assign zero contributions outside the domain
# ------------------------------------------------------------

linearized_columns = [
    f"edu_linearized_{j}"
    for j in range(k)
]

for column in linearized_columns:
    full_education_design[column] = 0.0

full_education_design.loc[
    education_test_data.index,
    linearized_columns
] = linearized

# ------------------------------------------------------------
# 7. Aggregate contributions within survey PSUs
# ------------------------------------------------------------

education_psu_totals = (
    full_education_design
    .groupby(
        ["stratum16", "psu"],
        observed=True
    )[linearized_columns]
    .sum()
)

# ------------------------------------------------------------
# 8. Calculate survey-design covariance matrix
# ------------------------------------------------------------

covariance = np.zeros((k, k))

for stratum, part in education_psu_totals.groupby(
    level=0,
    sort=True
):

    values = part.to_numpy(dtype=float)

    m = len(values)

    assert m >= 2

    centered = (
        values
        - values.mean(axis=0, keepdims=True)
    )

    covariance += (
        m / (m - 1)
    ) * (centered.T @ centered)

# ------------------------------------------------------------
# 9. Joint Wald test of equal category prevalences
# ------------------------------------------------------------

# Compare all other categories against the first group

contrast = np.zeros((k - 1, k))

for j in range(1, k):

    contrast[j - 1, 0] = -1

    contrast[j - 1, j] = 1

prevalence_differences = (
    contrast @ prevalences
)

contrast_covariance = (
    contrast @ covariance @ contrast.T
)

matrix_rank = np.linalg.matrix_rank(
    contrast_covariance
)

assert matrix_rank == k - 1, (
    "Contrast covariance is singular."
)

wald_f = (
    prevalence_differences
    @ np.linalg.solve(
        contrast_covariance,
        prevalence_differences
    )
) / (k - 1)

df1 = k - 1
df2 = design_df

p_value = f_distribution.sf(
    wald_f,
    df1,
    df2
)

# ------------------------------------------------------------
# 10. Print formal test result
# ------------------------------------------------------------

print("\nSURVEY-ADJUSTED EDUCATION OMNIBUS TEST:")

print("Number of groups:", k)

print("Wald F-statistic:",
      round(wald_f, 6))

print("Numerator df:", df1)

print("Denominator df:", df2)

print("p-value:", f"{p_value:.8e}")

print(
    "Significant at 5%:",
    bool(p_value < 0.05)
)

# ------------------------------------------------------------
# 11. Save results
# ------------------------------------------------------------

education_test_summary = pd.DataFrame([{
    "Variable": "Educational attainment",
    "Included_N": len(education_test_data),
    "Excluded_N": (
        len(women_analysis)
        - len(education_test_data)
    ),
    "Categories": k,
    "Wald_F": wald_f,
    "DF1": df1,
    "DF2": df2,
    "P_Value": p_value
}])

education_group_table.to_csv(
    os.path.join(
        output_dir,
        "Step1_22_Education_Group_Prevalence.csv"
    ),
    index=False
)

education_test_summary.to_csv(
    os.path.join(
        output_dir,
        "Step1_22_Education_Omnibus_Test.csv"
    ),
    index=False
)

# ------------------------------------------------------------
# 12. Final verification
# ------------------------------------------------------------

assert education_group_table["Total_N"].sum() == 2085

assert education_group_table["Low_N"].sum() == 550

assert np.isfinite(wald_f)

assert 0 <= p_value <= 1

assert len(women_analysis) == 2110

print("\nSTEP 1.22 VERIFICATION SUCCESSFUL.")


# %% STEP 1.23: SECTOR-SPECIFIC SURVEY-ADJUSTED TESTS

# ============================================================
# HIES 2022 - Employment status within each sector
#
# Contrast:
# Day labour (code 1) minus other employment statuses
#
# Separate tests for:
# 1. Agriculture
# 2. Non-agriculture
#
# Method:
# Stratified PSU-level Taylor linearization
#
# Original analytical dataset remains unchanged.
# ============================================================

import os
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

# ------------------------------------------------------------
# 1. Preserve full survey design
# ------------------------------------------------------------

design_123 = df[["stratum16", "psu"]].copy()

assert design_123.index.is_unique
assert design_123.notna().all().all()

n_psu_123 = design_123.drop_duplicates().shape[0]
n_strata_123 = design_123["stratum16"].nunique()

design_df_123 = n_psu_123 - n_strata_123

assert n_psu_123 == 720
assert n_strata_123 == 16
assert design_df_123 == 704


# ------------------------------------------------------------
# 2. Define sector-specific comparison function
# ------------------------------------------------------------

def employment_status_contrast(
    sector_code,
    status_variable,
    sector_label
):

    # Select the relevant employment sector only
    data = women_analysis.loc[
        women_analysis["earn_sector"] == sector_code,
        [
            status_variable,
            "low_work_intensity",
            "population_weight"
        ]
    ].copy()

    status = pd.to_numeric(
        data[status_variable],
        errors="coerce"
    )

    # Only questionnaire employment-status codes
    # are expected within the relevant sector.
    assert status.notna().all()
    assert status.isin([1, 2, 3, 4]).all()

    # Day labour versus all other recorded statuses
    day_mask = (status == 1).to_numpy()

    other_mask = (status != 1).to_numpy()

    assert np.all(day_mask | other_mask)
    assert not np.any(day_mask & other_mask)

    y = data["low_work_intensity"].to_numpy(
        dtype=float
    )

    w = data["population_weight"].to_numpy(
        dtype=float
    )

    assert np.isfinite(w).all()
    assert (w > 0).all()

    # --------------------------------------------------------
    # 3. Survey-weighted prevalence in each group
    # --------------------------------------------------------

    W_day = w[day_mask].sum()
    W_other = w[other_mask].sum()

    p_day = np.average(
        y[day_mask],
        weights=w[day_mask]
    )

    p_other = np.average(
        y[other_mask],
        weights=w[other_mask]
    )

    difference = p_day - p_other

    # --------------------------------------------------------
    # 4. Linearized observation contributions
    # --------------------------------------------------------

    linearized = np.zeros(
        len(data),
        dtype=float
    )

    linearized[day_mask] = (
        w[day_mask]
        * (y[day_mask] - p_day)
        / W_day
    )

    linearized[other_mask] = (
        -w[other_mask]
        * (y[other_mask] - p_other)
        / W_other
    )

    # --------------------------------------------------------
    # 5. Retain complete survey PSU structure
    # Observations outside this sector contribute zero
    # --------------------------------------------------------

    full = design_123.copy()

    full["Linearized"] = 0.0

    full.loc[
        data.index,
        "Linearized"
    ] = linearized

    psu_totals = (
        full.groupby(
            ["stratum16", "psu"],
            observed=True
        )["Linearized"]
        .sum()
        .reset_index()
    )

    # --------------------------------------------------------
    # 6. Stratified Taylor variance estimate
    # --------------------------------------------------------

    variance = 0.0

    for stratum, part in psu_totals.groupby(
        "stratum16",
        observed=True
    ):

        values = part[
            "Linearized"
        ].to_numpy(dtype=float)

        m = len(values)

        assert m >= 2

        variance += (
            m / (m - 1)
        ) * np.sum(
            (values - values.mean()) ** 2
        )

    se = np.sqrt(variance)

    assert np.isfinite(se)
    assert se > 0

    # --------------------------------------------------------
    # 7. Wald t-test and 95% confidence interval
    # --------------------------------------------------------

    t_stat = difference / se

    p_value = 2 * student_t.sf(
        abs(t_stat),
        df=design_df_123
    )

    critical = student_t.ppf(
        0.975,
        df=design_df_123
    )

    ci_lower = difference - critical * se
    ci_upper = difference + critical * se

    return {
        "Sector": sector_label,
        "Sector_N": len(data),
        "Day_Labour_N": int(day_mask.sum()),
        "Other_Status_N": int(other_mask.sum()),
        "Day_Labour_Low_N": int(y[day_mask].sum()),
        "Other_Status_Low_N": int(y[other_mask].sum()),
        "Day_Labour_Prevalence_Percent": p_day * 100,
        "Other_Status_Prevalence_Percent": p_other * 100,
        "Difference_pp": difference * 100,
        "SE_pp": se * 100,
        "CI_Lower_pp": ci_lower * 100,
        "CI_Upper_pp": ci_upper * 100,
        "t_statistic": t_stat,
        "DF": design_df_123,
        "P_Value": p_value
    }


# ------------------------------------------------------------
# 8. Run two sector-specific comparisons
# ------------------------------------------------------------

employment_results = [
    employment_status_contrast(
        sector_code=1,
        status_variable="earn_emp_status_ag",
        sector_label="Agriculture"
    ),
    employment_status_contrast(
        sector_code=2,
        status_variable="earn_emp_status_nonag",
        sector_label="Non-agriculture"
    )
]

employment_contrast_table = pd.DataFrame(
    employment_results
)

# ------------------------------------------------------------
# 9. Display results
# ------------------------------------------------------------

print("========== STEP 1.23 RESULTS ==========")

for _, row in employment_contrast_table.iterrows():

    print("\n" + "=" * 60)
    print("SECTOR:", row["Sector"])
    print("=" * 60)

    print("Sector N:", int(row["Sector_N"]))

    print(
        "Day labour N:",
        int(row["Day_Labour_N"])
    )

    print(
        "Other employment status N:",
        int(row["Other_Status_N"])
    )

    print(
        "Day labour low cases:",
        int(row["Day_Labour_Low_N"])
    )

    print(
        "Other status low cases:",
        int(row["Other_Status_Low_N"])
    )

    print(
        "Day labour weighted prevalence (%):",
        f"{row['Day_Labour_Prevalence_Percent']:.3f}"
    )

    print(
        "Other weighted prevalence (%):",
        f"{row['Other_Status_Prevalence_Percent']:.3f}"
    )

    print(
        "Difference in percentage points:",
        f"{row['Difference_pp']:.3f}"
    )

    print(
        "95% CI:",
        f"[{row['CI_Lower_pp']:.3f}, "
        f"{row['CI_Upper_pp']:.3f}]"
    )

    print(
        "p-value:",
        f"{row['P_Value']:.8e}"
    )

# ------------------------------------------------------------
# 10. Verify observed category totals
# ------------------------------------------------------------

ag = employment_contrast_table.iloc[0]
nonag = employment_contrast_table.iloc[1]

assert ag["Sector_N"] == 227
assert ag["Day_Labour_N"] == 169
assert ag["Other_Status_N"] == 58

assert nonag["Sector_N"] == 1883
assert nonag["Day_Labour_N"] == 389
assert nonag["Other_Status_N"] == 1494

assert (
    ag["Day_Labour_Low_N"]
    + ag["Other_Status_Low_N"]
) == 126

assert (
    nonag["Day_Labour_Low_N"]
    + nonag["Other_Status_Low_N"]
) == 427

assert employment_contrast_table[
    "P_Value"
].between(0, 1).all()

assert len(women_analysis) == 2110

# ------------------------------------------------------------
# 11. Save reproducible results
# ------------------------------------------------------------

employment_contrast_table.to_csv(
    os.path.join(
        output_dir,
        "Step1_23_Employment_Sector_Contrasts.csv"
    ),
    index=False
)

print("\nSTEP 1.23 VERIFICATION SUCCESSFUL.")






# %% STEP 1.24: HOLM CORRECTION FOR ALL 17 TESTS

# ============================================================
# HIES 2022 - Multiple-testing correction
#
# Combine 17 exploratory survey-adjusted tests:
# - 9 categorical tests
# - 5 continuous tests
# - 1 education omnibus test
# - 2 employment status contrasts
#
# Method: Holm step-down adjustment
#
# No data modification or new hypothesis tests.
# ============================================================

import os
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 1. Collect previously calculated p-values
# ------------------------------------------------------------

# Nine categorical tests from Step 1.20
part_1 = all_tests[
    ["Comparison", "P_Value"]
].copy()

part_1["Test_Family"] = "Categorical"

# Five continuous tests from Step 1.21
part_2 = continuous_inference_table[
    ["Variable", "P_Value"]
].copy()

part_2 = part_2.rename(
    columns={"Variable": "Comparison"}
)

part_2["Comparison"] = (
    "Continuous: " + part_2["Comparison"]
)

part_2["Test_Family"] = "Continuous"

# Education omnibus test from Step 1.22
part_3 = education_test_summary[
    ["Variable", "P_Value"]
].copy()

part_3 = part_3.rename(
    columns={"Variable": "Comparison"}
)

part_3["Comparison"] = (
    "Education omnibus (six groups)"
)

part_3["Test_Family"] = "Education"

# Two sector-specific employment tests from Step 1.23
part_4 = employment_contrast_table[
    ["Sector", "P_Value"]
].copy()

part_4 = part_4.rename(
    columns={"Sector": "Comparison"}
)

part_4["Comparison"] = (
    "Day labour vs others: " + part_4["Comparison"]
)

part_4["Test_Family"] = "Employment"

# ------------------------------------------------------------
# 2. Combine all tests
# ------------------------------------------------------------

final_tests = pd.concat(
    [part_1, part_2, part_3, part_4],
    ignore_index=True
)

assert len(part_1) == 9
assert len(part_2) == 5
assert len(part_3) == 1
assert len(part_4) == 2
assert len(final_tests) == 17

assert final_tests["Comparison"].is_unique

p_values = final_tests[
    "P_Value"
].to_numpy(dtype=float)

assert np.isfinite(p_values).all()
assert np.all(
    (p_values >= 0) & (p_values <= 1)
)

# ------------------------------------------------------------
# 3. Apply Holm step-down correction
# ------------------------------------------------------------

m = len(p_values)

# Stable ascending order of original p-values
order = np.argsort(
    p_values,
    kind="mergesort"
)

sorted_p = p_values[order]

# Multipliers: m, m-1, ..., 1
multipliers = (
    m - np.arange(m)
)

adjusted_sorted = np.maximum.accumulate(
    sorted_p * multipliers
)

adjusted_sorted = np.clip(
    adjusted_sorted,
    0.0,
    1.0
)

holm_adjusted = np.empty(
    m,
    dtype=float
)

holm_adjusted[order] = adjusted_sorted

final_tests["Holm_Adjusted_P"] = (
    holm_adjusted
)

# ------------------------------------------------------------
# 4. Mark statistical significance
# ------------------------------------------------------------

alpha = 0.05

final_tests["Raw_P_Significant"] = (
    final_tests["P_Value"] < alpha
)

final_tests["Holm_Significant"] = (
    final_tests["Holm_Adjusted_P"] < alpha
)

final_tests["Significance_Changed"] = (
    final_tests["Raw_P_Significant"]
    != final_tests["Holm_Significant"]
)

# ------------------------------------------------------------
# 5. Sort results for publication reporting
# ------------------------------------------------------------

final_tests = final_tests.sort_values(
    "P_Value",
    kind="mergesort"
).reset_index(drop=True)

print("========== STEP 1.24 RESULTS ==========")

print("\nALL 17 EXPLORATORY TESTS:")
print(
    final_tests[
        [
            "Test_Family",
            "Comparison",
            "P_Value",
            "Holm_Adjusted_P",
            "Raw_P_Significant",
            "Holm_Significant"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6e}"
    )
)

# ------------------------------------------------------------
# 6. Summarize significance changes
# ------------------------------------------------------------

print("\n========== SIGNIFICANCE SUMMARY ==========")

print("Total tests:", len(final_tests))

print(
    "Significant before correction:",
    int(final_tests["Raw_P_Significant"].sum())
)

print(
    "Significant after Holm correction:",
    int(final_tests["Holm_Significant"].sum())
)

print(
    "Tests with changed significance:",
    int(final_tests["Significance_Changed"].sum())
)

print("\nTESTS THAT LOST SIGNIFICANCE:")

lost_significance = final_tests.loc[
    final_tests["Raw_P_Significant"]
    & ~final_tests["Holm_Significant"],
    [
        "Comparison",
        "P_Value",
        "Holm_Adjusted_P"
    ]
]

if lost_significance.empty:
    print("None")
else:
    print(
        lost_significance.to_string(
            index=False,
            float_format=lambda x: f"{x:.6e}"
        )
    )

# ------------------------------------------------------------
# 7. Verification
# ------------------------------------------------------------

assert len(final_tests) == 17

assert final_tests[
    "Holm_Adjusted_P"
].between(0, 1).all()

assert (
    final_tests["Holm_Adjusted_P"].to_numpy()
    >= final_tests["P_Value"].to_numpy() - 1e-12
).all()

assert not (
    final_tests["Holm_Significant"]
    & ~final_tests["Raw_P_Significant"]
).any()

assert len(women_analysis) == 2110

assert (
    women_analysis["low_work_intensity"].sum()
    == 553
)

# ------------------------------------------------------------
# 8. Save results
# ------------------------------------------------------------

final_tests.to_csv(
    os.path.join(
        output_dir,
        "Step1_24_Final_17_Test_Holm_Correction.csv"
    ),
    index=False
)

print("\nSTEP 1.24 VERIFICATION SUCCESSFUL.")













# %% STEP 1.25: PSU-STRATIFIED OUTER CV FOLD AUDIT

# ============================================================
# HIES 2022 - Nested CV preparation
#
# Create 5 outer validation folds at PSU level,
# distributing PSUs within each survey stratum.
#
# IMPORTANT:
# - No model is fitted
# - No PCA, Boruta or imputation is fitted
# - No original variables are modified
# - Inner CV will be created later within training folds
# ============================================================

import os
import numpy as np
import pandas as pd

RANDOM_SEED = 20261030
N_OUTER_FOLDS = 5

rng = np.random.default_rng(RANDOM_SEED)

# ------------------------------------------------------------
# 1. Prepare PSU-level information
# ------------------------------------------------------------

cv_source = women_analysis[
    [
        "stratum16",
        "psu",
        "low_work_intensity",
        "population_weight"
    ]
].copy()

assert cv_source.index.is_unique
assert cv_source.notna().all().all()

psu_info = (
    cv_source
    .groupby(["stratum16", "psu"], observed=True)
    .agg(
        N=("low_work_intensity", "size"),
        Low_N=("low_work_intensity", "sum")
    )
    .reset_index()
)

assert len(cv_source) == 2110
assert int(cv_source["low_work_intensity"].sum()) == 553
assert len(psu_info) == 589
assert psu_info["stratum16"].nunique() == 16

# Ensure every stratum contains enough PSUs
psus_per_stratum = (
    psu_info.groupby("stratum16")["psu"]
    .size()
)

assert (
    psus_per_stratum >= N_OUTER_FOLDS
).all(), "At least one stratum has too few PSUs."

# ------------------------------------------------------------
# 2. Assign whole PSUs to outer folds within strata
# ------------------------------------------------------------

assignments = []

global_fold_n = np.zeros(
    N_OUTER_FOLDS, dtype=int
)

overall_event_rate = (
    cv_source["low_work_intensity"].mean()
)

for stratum, block in psu_info.groupby(
    "stratum16", sort=True
):

    work = block.copy()

    # Random tie-breaker, then distribute large PSUs first
    work["Random_Tie"] = rng.random(len(work))

    work = work.sort_values(
        ["N", "Random_Tie"],
        ascending=[False, True],
        kind="mergesort"
    )

    stratum_fold_groups = np.zeros(
        N_OUTER_FOLDS, dtype=int
    )

    stratum_fold_n = np.zeros(
        N_OUTER_FOLDS, dtype=int
    )

    for row in work.itertuples(index=False):

        # First balance number of PSUs within stratum,
        # then number of people and global fold sizes.
        scores = [
            (
                stratum_fold_groups[f],
                stratum_fold_n[f] + row.N,
                global_fold_n[f] + row.N,
                f
            )
            for f in range(N_OUTER_FOLDS)
        ]

        chosen_fold = min(
            range(N_OUTER_FOLDS),
            key=lambda f: scores[f]
        )

        assignments.append({
            "stratum16": stratum,
            "psu": row.psu,
            "outer_fold": chosen_fold
        })

        stratum_fold_groups[chosen_fold] += 1
        stratum_fold_n[chosen_fold] += row.N
        global_fold_n[chosen_fold] += row.N

psu_fold_assignment = pd.DataFrame(
    assignments
)

# ------------------------------------------------------------
# 3. Attach folds to individual observations
# ------------------------------------------------------------

cv_manifest = cv_source.copy()

cv_manifest["original_row_index"] = (
    cv_manifest.index
)

cv_manifest = cv_manifest.merge(
    psu_fold_assignment,
    on=["stratum16", "psu"],
    how="left",
    validate="many_to_one",
    sort=False
)

cv_manifest = cv_manifest.set_index(
    "original_row_index"
).sort_index()

assert cv_manifest["outer_fold"].notna().all()

cv_manifest["outer_fold"] = (
    cv_manifest["outer_fold"].astype(int)
)

# ------------------------------------------------------------
# 4. Audit each outer validation fold
# ------------------------------------------------------------

fold_summary_rows = []

for fold in range(N_OUTER_FOLDS):

    test = cv_manifest.loc[
        cv_manifest["outer_fold"] == fold
    ]

    train = cv_manifest.loc[
        cv_manifest["outer_fold"] != fold
    ]

    test_groups = set(
        zip(test["stratum16"], test["psu"])
    )

    train_groups = set(
        zip(train["stratum16"], train["psu"])
    )

    overlap = len(
        test_groups.intersection(train_groups)
    )

    assert overlap == 0

    assert test["low_work_intensity"].nunique() == 2
    assert train["low_work_intensity"].nunique() == 2

    weighted_prevalence = (
        np.average(
            test["low_work_intensity"],
            weights=test["population_weight"]
        ) * 100
    )

    fold_summary_rows.append({
        "Fold": fold + 1,
        "Train_N": len(train),
        "Validation_N": len(test),
        "Validation_Low_N": int(
            test["low_work_intensity"].sum()
        ),
        "Validation_Low_Percent": (
            test["low_work_intensity"].mean() * 100
        ),
        "Validation_Weighted_Low_Percent":
            weighted_prevalence,
        "Validation_PSU_Count": len(test_groups),
        "Validation_Strata_Count":
            test["stratum16"].nunique(),
        "Train_Validation_PSU_Overlap": overlap
    })

outer_fold_summary = pd.DataFrame(
    fold_summary_rows
)

# ------------------------------------------------------------
# 5. Verify stratum coverage across all five folds
# ------------------------------------------------------------

stratum_fold_psus = pd.crosstab(
    psu_fold_assignment["stratum16"],
    psu_fold_assignment["outer_fold"]
)

assert (stratum_fold_psus >= 1).all().all()

# Each PSU must occur in exactly one outer fold
assert (
    psu_fold_assignment
    .groupby(["stratum16", "psu"])["outer_fold"]
    .nunique() == 1
).all()

assert outer_fold_summary["Validation_N"].sum() == 2110

assert (
    outer_fold_summary["Validation_Low_N"].sum()
    == 553
)

assert (
    outer_fold_summary["Validation_PSU_Count"].sum()
    == 589
)

# ------------------------------------------------------------
# 6. Print diagnostic results
# ------------------------------------------------------------

print("========== STEP 1.25 RESULTS ==========")

print("\nFULL SAMPLE:")
print("Women:", len(cv_manifest))
print("Low cases:", int(cv_manifest["low_work_intensity"].sum()))
print("Survey strata:", psu_info["stratum16"].nunique())
print("Analytical PSU groups:", len(psu_info))
print("Outer folds:", N_OUTER_FOLDS)
print("Random seed:", RANDOM_SEED)

print("\nOUTER FOLD AUDIT:")

print(
    outer_fold_summary.round(2).to_string(index=False)
)

print("\nPSU COUNTS BY STRATUM AND FOLD:")

print(stratum_fold_psus.to_string())

print("\nFINAL VERIFICATION:")
print(
    "Maximum PSU overlap:",
    outer_fold_summary[
        "Train_Validation_PSU_Overlap"
    ].max()
)

print(
    "Minimum strata per validation fold:",
    outer_fold_summary[
        "Validation_Strata_Count"
    ].min()
)

print(
    "Total validation observations:",
    outer_fold_summary["Validation_N"].sum()
)

print(
    "Total validation low cases:",
    outer_fold_summary["Validation_Low_N"].sum()
)

# ------------------------------------------------------------
# 7. Save reproducible fold assignments
# ------------------------------------------------------------

cv_manifest.to_csv(
    os.path.join(
        output_dir,
        "Step1_25_Outer_CV_Manifest.csv"
    ),
    index_label="original_row_index"
)

outer_fold_summary.to_csv(
    os.path.join(
        output_dir,
        "Step1_25_Outer_CV_Audit.csv"
    ),
    index=False
)

stratum_fold_psus.to_csv(
    os.path.join(
        output_dir,
        "Step1_25_PSU_Stratum_Fold_Distribution.csv"
    )
)

print("\nSTEP 1.25 VERIFICATION SUCCESSFUL.")







# %% STEP 1.26: ML PREDICTOR SCHEMA AND ENVIRONMENT AUDIT

# ============================================================
# HIES 2022 - Leakage-free ML preparation
#
# Audit:
# 1. Final 35 candidate predictors
# 2. Categorical versus numerical treatment
# 3. Missing values and -99 placeholders
# 4. Raw inputs for fold-specific wealth PCA
# 5. Python ML package availability
#
# No transformations are fitted in this step.
# ============================================================

import os
import numpy as np
import pandas as pd
import importlib.util
import importlib.metadata

# ------------------------------------------------------------
# 1. Define categorical predictors
# ------------------------------------------------------------

model_categorical_features = [
    "marital_status",
    "relation_head_grp",
    "literate_basic",
    "education_attainment_group",
    "has_mobile",
    "has_bank_account",
    "has_mfs_account",
    "internet_used",
    "has_computer",
    "any_functional_difficulty",
    "rural_urban",
    "division_code",
    "rmo_code",
    "domain16_grp",
    "toilet_type_grp",
    "drinking_water_source_grp",
    "light_source",
    "cooking_stove_type",
    "cooking_fuel_grp",
    "transport_stop_halfkm",
    "transport_stop_time_min_missing",
    "regular_waste_collection",
    "slum_household",
    "earn_sector",
    "earn_emp_status_ag",
    "earn_emp_status_nonag",
    "earn_org_type_grp",
    "earn_ind_code_grp",
    "earn_occ_code_grp"
]

# ------------------------------------------------------------
# 2. Define numerical predictors
# ------------------------------------------------------------

model_numeric_features = [
    "age",
    "hh_MEMBER",
    "fies_raw_score",
    "wealth_living_index",
    "electricity_hours_24h",
    "transport_stop_time_min"
]

model_feature_names = (
    model_categorical_features
    + model_numeric_features
)

assert len(model_categorical_features) == 29
assert len(model_numeric_features) == 6
assert len(model_feature_names) == 35
assert len(set(model_feature_names)) == 35

# ------------------------------------------------------------
# 3. Map planned grouped predictors to their raw sources
#
# Rare-category grouping must be learned inside
# the appropriate training fold later.
# ------------------------------------------------------------

feature_source_map = {
    "relation_head_grp": "relation_head",
    "domain16_grp": "domain16",
    "toilet_type_grp": "toilet_type",
    "drinking_water_source_grp":
        "drinking_water_source",
    "cooking_fuel_grp": "cooking_fuel",
    "earn_org_type_grp": "earn_org_type",
    "earn_ind_code_grp": "earn_ind_code",
    "earn_occ_code_grp": "earn_occ_code"
}

# Wealth index is the only predictor requiring
# a separate PCA construction.
pca_derived_feature = "wealth_living_index"

# ------------------------------------------------------------
# 4. Audit all non-PCA feature sources
# ------------------------------------------------------------

schema_rows = []

for feature in model_feature_names:

    feature_type = (
        "Categorical"
        if feature in model_categorical_features
        else "Numerical"
    )

    if feature == pca_derived_feature:

        schema_rows.append({
            "Model_Feature": feature,
            "Feature_Type": feature_type,
            "Source_Column": "Fold-specific PCA",
            "Missing_N": np.nan,
            "Negative99_N": np.nan,
            "Unique_Values": np.nan,
            "Construction": "Fit within training folds"
        })

        continue

    source = feature_source_map.get(
        feature, feature
    )

    assert source in women_analysis.columns, (
        f"Missing source column: {source}"
    )

    values = women_analysis[source]

    missing_n = int(values.isna().sum())

    numeric_check = pd.to_numeric(
        values,
        errors="coerce"
    )

    negative99_n = int(
        (numeric_check == -99).sum()
    )

    schema_rows.append({
        "Model_Feature": feature,
        "Feature_Type": feature_type,
        "Source_Column": source,
        "Missing_N": missing_n,
        "Negative99_N": negative99_n,
        "Unique_Values": int(values.nunique(dropna=True)),
        "Construction": (
            "Training-fold category grouping"
            if feature in feature_source_map
            else "Existing source variable"
        )
    })

model_schema_audit = pd.DataFrame(schema_rows)

# ------------------------------------------------------------
# 5. Confirm prohibited leakage variables are excluded
# ------------------------------------------------------------

prohibited_predictors = {
    "earn_annual_hours",
    "low_work_intensity",
    "population_weight",
    "psu",
    "stratum16"
}

assert not (
    prohibited_predictors
    & set(model_feature_names)
)

assert not (
    prohibited_predictors
    & set(
        model_schema_audit["Source_Column"]
    )
)

# ------------------------------------------------------------
# 6. Audit the 14 raw wealth PCA inputs
#
# hh_rooms_pca includes the earlier treatment
# of the single implausible room count.
#
# These data are audited now, not transformed.
# ------------------------------------------------------------

wealth_pca_inputs = [
    "hh_rooms_pca",
    "covered_area",
    "house_value",
    "land_property_value",
    "other_asset_value",
    "sec9a1_food_value",
    "sec9b2_value_total",
    "sec9c_total_value",
    "sec9d1_value_total",
    "sec9d2_value_total",
    "durable_owned_count",
    "durable_total_number",
    "durable_total_value",
    "sec9_total_observed_value"
]

assert len(wealth_pca_inputs) == 14

wealth_pca_rows = []

for variable in wealth_pca_inputs:

    assert variable in women_analysis.columns

    values = pd.to_numeric(
        women_analysis[variable],
        errors="coerce"
    ).astype(float)

    finite = values[np.isfinite(values)]

    wealth_pca_rows.append({
        "PCA_Input": variable,
        "Observed_N": int(len(finite)),
        "Missing_or_Nonfinite_N": int(
            len(values) - len(finite)
        ),
        "Negative_N": int((finite < 0).sum()),
        "Zero_N": int((finite == 0).sum()),
        "Minimum": float(finite.min()),
        "Median": float(finite.median()),
        "Maximum": float(finite.max())
    })

wealth_pca_audit = pd.DataFrame(
    wealth_pca_rows
)

# ------------------------------------------------------------
# 7. Audit ML libraries available in Spyder
# ------------------------------------------------------------

package_specs = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("scikit-learn", "sklearn"),
    ("Boruta", "boruta"),
    ("xgboost", "xgboost"),
    ("lightgbm", "lightgbm"),
    ("catboost", "catboost"),
    ("shap", "shap"),
    ("imbalanced-learn", "imblearn")
]

package_rows = []

for package_name, module_name in package_specs:

    installed = (
        importlib.util.find_spec(module_name)
        is not None
    )

    if installed:
        try:
            version = importlib.metadata.version(
                package_name
            )
        except importlib.metadata.PackageNotFoundError:
            version = "Installed; version unavailable"
    else:
        version = "NOT INSTALLED"

    package_rows.append({
        "Package": package_name,
        "Available": installed,
        "Version": version
    })

package_audit = pd.DataFrame(package_rows)

# ------------------------------------------------------------
# 8. Print results
# ------------------------------------------------------------

print("========== STEP 1.26 RESULTS ==========")

print("\nPREDICTOR SCHEMA SUMMARY:")

print("Total candidate predictors:",
      len(model_feature_names))

print("Categorical predictors:",
      len(model_categorical_features))

print("Numerical predictors:",
      len(model_numeric_features))

print("Fold-specific PCA predictor:",
      pca_derived_feature)

print("\nPREDICTOR SOURCE AND MISSINGNESS AUDIT:")

print(
    model_schema_audit.to_string(index=False)
)

print("\nWEALTH PCA INPUT AUDIT:")

print(
    wealth_pca_audit.round(2).to_string(index=False)
)

print("\nPYTHON PACKAGE AUDIT:")

print(
    package_audit.to_string(index=False)
)

# ------------------------------------------------------------
# 9. Additional structural checks
# ------------------------------------------------------------

assert (
    women_analysis[
        "transport_stop_time_min_missing"
    ].sum() == 833
)

assert (
    women_analysis["education_attainment_group"]
    .notna().all()
)

assert len(cv_manifest) == 2110

assert cv_manifest["outer_fold"].nunique() == 5

assert len(women_analysis) == 2110

assert (
    women_analysis["low_work_intensity"].sum()
    == 553
)

# ------------------------------------------------------------
# 10. Save audit outputs
# ------------------------------------------------------------

model_schema_audit.to_csv(
    os.path.join(
        output_dir,
        "Step1_26_ML_Predictor_Schema.csv"
    ),
    index=False
)

wealth_pca_audit.to_csv(
    os.path.join(
        output_dir,
        "Step1_26_Wealth_PCA_Input_Audit.csv"
    ),
    index=False
)

package_audit.to_csv(
    os.path.join(
        output_dir,
        "Step1_26_Python_Package_Audit.csv"
    ),
    index=False
)

print("\nSTEP 1.26 VERIFICATION SUCCESSFUL.")























# %% STEP 1.27: NESTED CV INNER-FOLD CONSTRUCTION

# ============================================================
# HIES 2022 - PSU-stratified inner cross-validation
#
# For every outer-training sample:
#   - Create three inner validation folds
#   - Assign complete PSU groups to folds
#   - Distribute PSU groups within each survey stratum
#   - Check outcome and survey-weight balance
#
# NO models, PCA, Boruta or imputation are fitted.
# ============================================================

import os
import numpy as np
import pandas as pd

N_INNER_FOLDS = 3
INNER_SEED = 20261030

assert len(cv_manifest) == 2110
assert cv_manifest.index.is_unique
assert cv_manifest.index.name == "original_row_index"
assert cv_manifest["outer_fold"].nunique() == 5

inner_manifest_parts = []
inner_audit_rows = []

print("========== STEP 1.27 RESULTS ==========")

# ------------------------------------------------------------
# 1. Process each outer-training sample separately
# ------------------------------------------------------------

for outer_fold in range(5):

    outer_train = cv_manifest.loc[
        cv_manifest["outer_fold"] != outer_fold
    ].copy()

    outer_validation = cv_manifest.loc[
        cv_manifest["outer_fold"] == outer_fold
    ].copy()

    # Create one row per PSU in the outer-training data
    psu_summary = (
        outer_train
        .groupby(["stratum16", "psu"], observed=True)
        .agg(
            N=("low_work_intensity", "size"),
            Low_N=("low_work_intensity", "sum")
        )
        .reset_index()
    )

    # Every stratum must support three inner folds
    stratum_psu_counts = (
        psu_summary.groupby("stratum16")["psu"]
        .size()
    )

    assert (
        stratum_psu_counts >= N_INNER_FOLDS
    ).all()

    rng = np.random.default_rng(
        INNER_SEED + 1000 + outer_fold
    )

    target_rate = (
        outer_train["low_work_intensity"].mean()
    )

    global_fold_n = np.zeros(
        N_INNER_FOLDS, dtype=int
    )

    global_fold_low = np.zeros(
        N_INNER_FOLDS, dtype=int
    )

    psu_assignments = []

    # --------------------------------------------------------
    # 2. Assign PSUs to inner folds within each stratum
    # --------------------------------------------------------

    for stratum, block in psu_summary.groupby(
        "stratum16", sort=True
    ):

        block = block.copy()

        # Randomized tie-breaking for reproducibility
        block["Random_Tie"] = rng.random(
            len(block)
        )

        # Place larger PSUs first
        block = block.sort_values(
            ["N", "Random_Tie"],
            ascending=[False, True],
            kind="mergesort"
        )

        stratum_fold_psus = np.zeros(
            N_INNER_FOLDS, dtype=int
        )

        for row in block.itertuples(index=False):

            # Primary: balance PSU counts in the stratum
            # Secondary: balance overall sample sizes
            # Tertiary: balance outcome counts
            chosen_fold = min(
                range(N_INNER_FOLDS),
                key=lambda f: (
                    stratum_fold_psus[f],
                    global_fold_n[f],
                    abs(
                        (global_fold_low[f] + row.Low_N)
                        - target_rate
                        * (global_fold_n[f] + row.N)
                    ),
                    f
                )
            )

            psu_assignments.append({
                "stratum16": stratum,
                "psu": row.psu,
                "inner_fold": chosen_fold
            })

            stratum_fold_psus[chosen_fold] += 1
            global_fold_n[chosen_fold] += row.N
            global_fold_low[chosen_fold] += row.Low_N

    assignment_table = pd.DataFrame(
        psu_assignments
    )

    assert len(assignment_table) == len(psu_summary)

    # Every stratum represented in every inner fold
    coverage = pd.crosstab(
        assignment_table["stratum16"],
        assignment_table["inner_fold"]
    )

    assert coverage.shape == (16, 3)
    assert (coverage >= 1).all().all()

    # --------------------------------------------------------
    # 3. Attach inner-fold assignments to individual women
    # --------------------------------------------------------

    assigned = (
        outer_train
        .reset_index()
        .merge(
            assignment_table,
            on=["stratum16", "psu"],
            how="left",
            validate="many_to_one",
            sort=False
        )
    )

    assert len(assigned) == len(outer_train)
    assert assigned["inner_fold"].notna().all()

    assigned["inner_fold"] = (
        assigned["inner_fold"].astype(int)
    )

    assigned["outer_fold"] = outer_fold

    # Check outer-validation PSU isolation
    outer_val_groups = set(
        zip(
            outer_validation["stratum16"],
            outer_validation["psu"]
        )
    )

    outer_train_groups = set(
        zip(
            outer_train["stratum16"],
            outer_train["psu"]
        )
    )

    assert not (
        outer_val_groups & outer_train_groups
    )

    # --------------------------------------------------------
    # 4. Audit all three inner validation folds
    # --------------------------------------------------------

    for inner_fold in range(N_INNER_FOLDS):

        inner_val = assigned.loc[
            assigned["inner_fold"] == inner_fold
        ]

        inner_train = assigned.loc[
            assigned["inner_fold"] != inner_fold
        ]

        val_groups = set(
            zip(inner_val["stratum16"], inner_val["psu"])
        )

        train_groups = set(
            zip(inner_train["stratum16"], inner_train["psu"])
        )

        psu_overlap = len(
            val_groups & train_groups
        )

        outer_psu_overlap = len(
            val_groups & outer_val_groups
        )

        assert psu_overlap == 0
        assert outer_psu_overlap == 0

        assert inner_val["stratum16"].nunique() == 16

        # Each split must contain both outcome classes
        assert (
            inner_val["low_work_intensity"].nunique()
            == 2
        )

        assert (
            inner_train["low_work_intensity"].nunique()
            == 2
        )

        val_weighted_low = (
            np.average(
                inner_val["low_work_intensity"],
                weights=inner_val["population_weight"]
            ) * 100
        )

        inner_audit_rows.append({
            "Outer_Fold": outer_fold + 1,
            "Inner_Fold": inner_fold + 1,
            "Inner_Train_N": len(inner_train),
            "Inner_Validation_N": len(inner_val),
            "Inner_Validation_Low_N": int(
                inner_val["low_work_intensity"].sum()
            ),
            "Inner_Validation_Low_Percent": (
                inner_val["low_work_intensity"].mean()
                * 100
            ),
            "Inner_Validation_Weighted_Low_Percent":
                val_weighted_low,
            "Validation_PSU_Count": len(val_groups),
            "Validation_Strata_Count":
                inner_val["stratum16"].nunique(),
            "Train_Validation_PSU_Overlap":
                psu_overlap,
            "Outer_Validation_PSU_Overlap":
                outer_psu_overlap
        })

    # Every outer-training observation assigned once
    assert (
        assigned["original_row_index"].nunique()
        == len(outer_train)
    )

    inner_manifest_parts.append(
        assigned[
            [
                "original_row_index",
                "outer_fold",
                "inner_fold",
                "stratum16",
                "psu",
                "low_work_intensity",
                "population_weight"
            ]
        ].copy()
    )

# ------------------------------------------------------------
# 5. Combine manifests and audit results
# ------------------------------------------------------------

inner_cv_manifest = pd.concat(
    inner_manifest_parts,
    ignore_index=True
)

inner_cv_audit = pd.DataFrame(
    inner_audit_rows
)

# ------------------------------------------------------------
# 6. Global checks
# ------------------------------------------------------------

assert len(inner_cv_audit) == 15

# Each woman belongs to four of five outer-training sets
assert len(inner_cv_manifest) == 2110 * 4

observations_per_outer_training = (
    inner_cv_manifest.groupby(
        "original_row_index"
    )["outer_fold"].nunique()
)

assert (
    observations_per_outer_training == 4
).all()

assert (
    inner_cv_audit[
        "Train_Validation_PSU_Overlap"
    ] == 0
).all()

assert (
    inner_cv_audit[
        "Outer_Validation_PSU_Overlap"
    ] == 0
).all()

assert (
    inner_cv_audit[
        "Validation_Strata_Count"
    ] == 16
).all()

for outer_fold in range(5):

    relevant = inner_cv_audit.loc[
        inner_cv_audit["Outer_Fold"]
        == outer_fold + 1
    ]

    expected_n = int(
        (cv_manifest["outer_fold"] != outer_fold).sum()
    )

    assert (
        relevant["Inner_Validation_N"].sum()
        == expected_n
    )

# ------------------------------------------------------------
# 7. Print results
# ------------------------------------------------------------

print("\nINNER CROSS-VALIDATION AUDIT:")

print(
    inner_cv_audit.round(2).to_string(index=False)
)

print("\n========== FINAL VERIFICATION ==========")

print("Outer folds:", 5)

print("Inner folds per outer fold:", N_INNER_FOLDS)

print("Total inner validation splits:",
      len(inner_cv_audit))

print("Minimum inner validation N:",
      inner_cv_audit["Inner_Validation_N"].min())

print("Minimum low cases in inner validation:",
      inner_cv_audit["Inner_Validation_Low_N"].min())

print("Minimum strata per inner fold:",
      inner_cv_audit["Validation_Strata_Count"].min())

print("Maximum train/validation PSU overlap:",
      inner_cv_audit[
          "Train_Validation_PSU_Overlap"
      ].max())

print("Maximum outer-validation PSU overlap:",
      inner_cv_audit[
          "Outer_Validation_PSU_Overlap"
      ].max())

print("Manifest records:",
      len(inner_cv_manifest))

# ------------------------------------------------------------
# 8. Save reproducible outputs
# ------------------------------------------------------------

inner_cv_manifest.to_csv(
    os.path.join(
        output_dir,
        "Step1_27_Nested_Inner_CV_Manifest.csv"
    ),
    index=False
)

inner_cv_audit.to_csv(
    os.path.join(
        output_dir,
        "Step1_27_Nested_Inner_CV_Audit.csv"
    ),
    index=False
)

print("\nSTEP 1.27 VERIFICATION SUCCESSFUL.")

























# %% STEP 1.28: TRAINING-FOLD PREPROCESSING AND PCA

import os
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# ------------------------------------------------------------
# 1. Construct raw predictor frame
# No learned transformations at this stage
# ------------------------------------------------------------

X_raw = pd.DataFrame(index=women_analysis.index)

for feature in model_categorical_features:

    source = feature_source_map.get(feature, feature)

    X_raw[feature] = (
        women_analysis[source]
        .astype("string")
        .fillna("__MISSING__")
        .astype(str)
    )

numeric_features_without_pca = [
    feature for feature in model_numeric_features
    if feature != "wealth_living_index"
]

for feature in numeric_features_without_pca:

    X_raw[feature] = pd.to_numeric(
        women_analysis[feature],
        errors="coerce"
    ).astype(float)

for feature in wealth_pca_inputs:

    X_raw[feature] = pd.to_numeric(
        women_analysis[feature],
        errors="coerce"
    ).astype(float)

assert X_raw.shape == (2110, 48)
assert X_raw.index.equals(women_analysis.index)

# Prevent outcome/design variables entering the model
for forbidden in [
    "low_work_intensity",
    "earn_annual_hours",
    "population_weight",
    "psu",
    "stratum16"
]:
    assert forbidden not in X_raw.columns


# ------------------------------------------------------------
# 2. Training-only winsorization and log transform
# ------------------------------------------------------------

class WealthLogWinsorizer(BaseEstimator, TransformerMixin):

    def __init__(self, lower_q=0.01, upper_q=0.99):
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, X, y=None):

        arr = np.asarray(X, dtype=float)

        arr = np.where(
            np.isfinite(arr), arr, np.nan
        )

        if np.any(arr[np.isfinite(arr)] < 0):
            raise ValueError(
                "Negative wealth input detected."
            )

        if np.any(np.isnan(arr).all(axis=0)):
            raise ValueError(
                "Entire wealth column missing in training."
            )

        self.lower_bounds_ = np.nanquantile(
            arr, self.lower_q, axis=0
        )

        self.upper_bounds_ = np.nanquantile(
            arr, self.upper_q, axis=0
        )

        self.n_features_in_ = arr.shape[1]

        return self

    def transform(self, X):

        arr = np.asarray(X, dtype=float)

        arr = np.where(
            np.isfinite(arr), arr, np.nan
        )

        if np.any(arr[np.isfinite(arr)] < 0):
            raise ValueError(
                "Negative wealth input detected."
            )

        clipped = np.clip(
            arr,
            self.lower_bounds_,
            self.upper_bounds_
        )

        return np.log1p(clipped)

    def get_feature_names_out(self, input_features=None):

        if input_features is None:
            return np.array([
                f"x{i}" for i in range(self.n_features_in_)
            ], dtype=object)

        return np.asarray(input_features, dtype=object)


# ------------------------------------------------------------
# 3. PCA with consistent index orientation
#
# Anchor PCA direction to the house_value input.
# Higher house_value loading is defined as positive.
# ------------------------------------------------------------

class OrientedPCA1(BaseEstimator, TransformerMixin):

    def __init__(self, anchor_feature_index=2):
        self.anchor_feature_index = anchor_feature_index

    def fit(self, X, y=None):

        self.pca_ = PCA(
            n_components=1,
            svd_solver="full"
        )

        self.pca_.fit(X)

        loading = self.pca_.components_[
            0, self.anchor_feature_index
        ]

        self.orientation_ = (
            1.0 if loading >= 0 else -1.0
        )

        self.explained_variance_ratio_ = (
            self.pca_.explained_variance_ratio_[0]
        )

        self.anchor_loading_ = (
            loading * self.orientation_
        )

        return self

    def transform(self, X):

        return (
            self.orientation_
            * self.pca_.transform(
                np.asarray(X, dtype=float)
            )
        )

    def get_feature_names_out(self, input_features=None):

        return np.array(
            ["wealth_living_index"],
            dtype=object
        )


# ------------------------------------------------------------
# 4. Build reusable, initially unfitted preprocessing
# ------------------------------------------------------------

preprocessor_template = ColumnTransformer(
    transformers=[

        (
            "categorical",
            OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=15,
                sparse_output=False,
                dtype=np.float64
            ),
            model_categorical_features
        ),

        (
            "numerical",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]),
            numeric_features_without_pca
        ),

        (
            "wealth",
            Pipeline([
                (
                    "winsor_log",
                    WealthLogWinsorizer(
                        lower_q=0.01,
                        upper_q=0.99
                    )
                ),
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                ),
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "pca",
                    OrientedPCA1(
                        anchor_feature_index=2
                    )
                )
            ]),
            wealth_pca_inputs
        )
    ],
    remainder="drop",
    sparse_threshold=0,
    verbose_feature_names_out=True
)


# ------------------------------------------------------------
# 5. Select one inner split for an initial leakage audit
# ------------------------------------------------------------

demo_outer_fold = 0
demo_inner_fold = 0

demo_manifest = inner_cv_manifest.loc[
    inner_cv_manifest["outer_fold"] == demo_outer_fold
].copy()

train_indices = demo_manifest.loc[
    demo_manifest["inner_fold"] != demo_inner_fold,
    "original_row_index"
].to_numpy()

validation_indices = demo_manifest.loc[
    demo_manifest["inner_fold"] == demo_inner_fold,
    "original_row_index"
].to_numpy()

assert len(train_indices) == 1128
assert len(validation_indices) == 561

assert len(
    set(train_indices) & set(validation_indices)
) == 0

X_train_raw = X_raw.loc[train_indices]
X_validation_raw = X_raw.loc[validation_indices]

assert X_train_raw.shape == (1128, 48)
assert X_validation_raw.shape == (561, 48)


# ------------------------------------------------------------
# 6. Fit using TRAINING observations ONLY
# ------------------------------------------------------------

preprocessor_demo = clone(preprocessor_template)

X_train_processed = (
    preprocessor_demo.fit_transform(X_train_raw)
)

# Validation receives only learned transformations
X_validation_processed = (
    preprocessor_demo.transform(X_validation_raw)
)

feature_names_processed = (
    preprocessor_demo.get_feature_names_out()
)

assert X_train_processed.shape[0] == 1128
assert X_validation_processed.shape[0] == 561

assert (
    X_train_processed.shape[1]
    == X_validation_processed.shape[1]
)

assert (
    X_train_processed.shape[1]
    == len(feature_names_processed)
)

assert np.isfinite(X_train_processed).all()
assert np.isfinite(X_validation_processed).all()


# ------------------------------------------------------------
# 7. Inspect fitted wealth PCA
# ------------------------------------------------------------

wealth_pipeline = (
    preprocessor_demo.named_transformers_["wealth"]
)

wealth_winsor = (
    wealth_pipeline.named_steps["winsor_log"]
)

wealth_pca = (
    wealth_pipeline.named_steps["pca"]
)

# Verify that clipping bounds come from TRAINING only
train_wealth = X_train_raw[
    wealth_pca_inputs
].to_numpy(dtype=float)

expected_lower = np.nanquantile(
    train_wealth, 0.01, axis=0
)

expected_upper = np.nanquantile(
    train_wealth, 0.99, axis=0
)

assert np.allclose(
    wealth_winsor.lower_bounds_,
    expected_lower
)

assert np.allclose(
    wealth_winsor.upper_bounds_,
    expected_upper
)

assert wealth_pca.anchor_loading_ >= 0


# ------------------------------------------------------------
# 8. Audit rare-category handling
# ------------------------------------------------------------

encoder = (
    preprocessor_demo.named_transformers_["categorical"]
)

rare_summary = []

for feature, rare_values in zip(
    model_categorical_features,
    encoder.infrequent_categories_
):

    n_rare = (
        0 if rare_values is None
        else len(rare_values)
    )

    rare_summary.append({
        "Feature": feature,
        "Rare_Training_Categories_Grouped": n_rare
    })

rare_audit = pd.DataFrame(rare_summary)

# ------------------------------------------------------------
# 9. Print diagnostics
# ------------------------------------------------------------

print("========== STEP 1.28 RESULTS ==========")

print("\nRAW INPUT:")
print("Observations:", len(X_raw))
print("Raw columns:", X_raw.shape[1])
print("Categorical predictors:",
      len(model_categorical_features))
print("Regular numerical predictors:",
      len(numeric_features_without_pca))
print("Wealth PCA raw inputs:",
      len(wealth_pca_inputs))

print("\nINNER SPLIT:")
print("Outer fold:", demo_outer_fold + 1)
print("Inner fold:", demo_inner_fold + 1)
print("Training observations:", len(train_indices))
print("Validation observations:", len(validation_indices))

print("\nPROCESSED MATRICES:")
print("Training shape:", X_train_processed.shape)
print("Validation shape:", X_validation_processed.shape)
print("Finite training values:",
      bool(np.isfinite(X_train_processed).all()))
print("Finite validation values:",
      bool(np.isfinite(X_validation_processed).all()))

print("\nWEALTH PCA:")
print(
    "First-component explained variance (%):",
    round(
        wealth_pca.explained_variance_ratio_ * 100,
        3
    )
)
print(
    "House-value anchor loading:",
    round(wealth_pca.anchor_loading_, 5)
)
print(
    "Wealth feature name:",
    [
        name for name in feature_names_processed
        if "wealth_living_index" in name
    ]
)

print("\nRARE CATEGORY GROUPING:")
print(
    rare_audit.loc[
        rare_audit[
            "Rare_Training_Categories_Grouped"
        ] > 0
    ].to_string(index=False)
)

education_index = model_categorical_features.index(
    "education_attainment_group"
)

print("\nEDUCATION TRAINING CATEGORIES:")
print(encoder.categories_[education_index])

print("\nEDUCATION INFREQUENT CATEGORIES:")
print(
    encoder.infrequent_categories_[education_index]
)

print("\nTOTAL ENCODED FEATURES:",
      len(feature_names_processed))


# ------------------------------------------------------------
# 10. Save audit information
# ------------------------------------------------------------

rare_audit.to_csv(
    os.path.join(
        output_dir,
        "Step1_28_Rare_Category_Audit.csv"
    ),
    index=False
)

pd.DataFrame({
    "Processed_Feature": feature_names_processed
}).to_csv(
    os.path.join(
        output_dir,
        "Step1_28_Processed_Feature_Names.csv"
    ),
    index=False
)

print("\nSTEP 1.28 VERIFICATION SUCCESSFUL.")







# %% STEP 1.29: SURVEY-WEIGHTED BORUTA - ONE INNER FOLD

# ============================================================
# HIES 2022 - Boruta feature-selection pilot
#
# Data:
# Outer Fold 1, Inner Fold 1 - TRAINING ONLY
#
# Boruta:
# - Random Forest importance
# - Survey sampling weights
# - Shadow-feature comparison
# - Original predictor-group mapping
#
# No validation observations are used for selection.
# No hyperparameter tuning in this step.
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

BORUTA_SEED = 20261030

# ------------------------------------------------------------
# 1. Verify existing training-only processed data
# ------------------------------------------------------------

assert X_train_processed.shape == (1128, 156)
assert X_validation_processed.shape == (561, 156)

assert np.isfinite(X_train_processed).all()

assert len(feature_names_processed) == 156

assert len(set(train_indices) & set(validation_indices)) == 0

y_boruta = women_analysis.loc[
    train_indices, "low_work_intensity"
].to_numpy(dtype=int)

w_boruta = women_analysis.loc[
    train_indices, "population_weight"
].to_numpy(dtype=float)

assert len(y_boruta) == 1128
assert len(w_boruta) == 1128
assert set(np.unique(y_boruta)) == {0, 1}
assert np.isfinite(w_boruta).all()
assert (w_boruta > 0).all()

# Normalizing weights preserves their relative magnitudes
w_boruta = w_boruta / w_boruta.mean()


# ------------------------------------------------------------
# 2. Weight-aware Random Forest for BorutaPy
#
# BorutaPy calls estimator.fit(X, y) without passing
# sample_weight. This wrapper supplies the training
# weights on every internal forest fit.
# ------------------------------------------------------------

class SurveyWeightedBorutaForest(RandomForestClassifier):

    def fit(self, X, y, sample_weight=None):

        if sample_weight is None:
            sample_weight = getattr(
                self,
                "_boruta_training_weights",
                None
            )

        if sample_weight is None:
            raise ValueError(
                "Survey weights are required for Boruta."
            )

        sample_weight = np.asarray(
            sample_weight,
            dtype=float
        )

        if len(sample_weight) != len(y):
            raise ValueError(
                "Boruta sample-weight length mismatch."
            )

        return super().fit(
            X,
            y,
            sample_weight=sample_weight
        )


# ------------------------------------------------------------
# 3. Configure Boruta's base Random Forest
# ------------------------------------------------------------

boruta_rf_demo = SurveyWeightedBorutaForest(
    n_estimators=160,
    max_depth=5,
    min_samples_leaf=3,
    max_features="sqrt",
    bootstrap=True,
    class_weight=None,
    n_jobs=-1,
    random_state=BORUTA_SEED
)

# Attach weights corresponding exactly to training rows
boruta_rf_demo._boruta_training_weights = (
    w_boruta.copy()
)

# ------------------------------------------------------------
# 4. Configure Boruta feature selector
# ------------------------------------------------------------

boruta_demo = BorutaPy(
    estimator=boruta_rf_demo,
    n_estimators=160,
    perc=100,
    alpha=0.05,
    two_step=True,
    max_iter=40,
    random_state=BORUTA_SEED,
    verbose=0
)

# ------------------------------------------------------------
# 5. Fit Boruta ONLY on inner-training observations
# ------------------------------------------------------------

print("========== STEP 1.29 RESULTS ==========")

print("\nStarting training-only Boruta selection...")

boruta_demo.fit(
    np.asarray(
        X_train_processed,
        dtype=np.float64
    ),
    y_boruta
)

# ------------------------------------------------------------
# 6. Extract Boruta column-level decisions
# ------------------------------------------------------------

confirmed_mask = np.asarray(
    boruta_demo.support_,
    dtype=bool
)

tentative_mask = np.asarray(
    boruta_demo.support_weak_,
    dtype=bool
)

ranking = np.asarray(
    boruta_demo.ranking_,
    dtype=int
)

assert len(confirmed_mask) == 156
assert len(tentative_mask) == 156
assert len(ranking) == 156

assert not np.any(
    confirmed_mask & tentative_mask
)

boruta_column_table = pd.DataFrame({
    "Encoded_Feature": feature_names_processed,
    "Confirmed": confirmed_mask,
    "Tentative": tentative_mask,
    "Ranking": ranking
})

boruta_column_table["Boruta_Status"] = np.select(
    [
        boruta_column_table["Confirmed"],
        boruta_column_table["Tentative"]
    ],
    [
        "Confirmed",
        "Tentative"
    ],
    default="Rejected"
)


# ------------------------------------------------------------
# 7. Map encoded columns back to the 35 predictors
# ------------------------------------------------------------

def identify_original_predictor(encoded_name):

    # One-hot encoded categorical features
    if encoded_name.startswith("categorical__"):

        remainder = encoded_name[
            len("categorical__"):
        ]

        matches = [
            feature
            for feature in model_categorical_features
            if remainder.startswith(feature + "_")
        ]

        if len(matches) != 1:
            raise ValueError(
                f"Ambiguous categorical mapping: {encoded_name}"
            )

        return matches[0]

    # Standard numerical features
    if encoded_name.startswith("numerical__"):

        feature = encoded_name[
            len("numerical__"):
        ]

        if feature not in numeric_features_without_pca:
            raise ValueError(
                f"Unexpected numerical feature: {encoded_name}"
            )

        return feature

    # Wealth PCA index
    if encoded_name == "wealth__wealth_living_index":
        return "wealth_living_index"

    raise ValueError(
        f"Unrecognized encoded feature: {encoded_name}"
    )


boruta_column_table["Original_Predictor"] = [
    identify_original_predictor(name)
    for name in boruta_column_table["Encoded_Feature"]
]

assert (
    boruta_column_table["Original_Predictor"].nunique()
    == 35
)

assert set(
    boruta_column_table["Original_Predictor"]
) == set(model_feature_names)


# ------------------------------------------------------------
# 8. Summarize decisions by original predictor
# ------------------------------------------------------------

boruta_group_summary = (
    boruta_column_table
    .groupby("Original_Predictor")
    .agg(
        Encoded_Columns=("Encoded_Feature", "size"),
        Confirmed_Columns=("Confirmed", "sum"),
        Tentative_Columns=("Tentative", "sum"),
        Best_Boruta_Rank=("Ranking", "min")
    )
    .reset_index()
)

boruta_group_summary["Group_Status"] = np.select(
    [
        boruta_group_summary["Confirmed_Columns"] > 0,
        boruta_group_summary["Tentative_Columns"] > 0
    ],
    [
        "At least one confirmed dummy",
        "Tentative dummy only"
    ],
    default="No selected dummy"
)

status_order = {
    "At least one confirmed dummy": 0,
    "Tentative dummy only": 1,
    "No selected dummy": 2
}

boruta_group_summary["_sort"] = (
    boruta_group_summary["Group_Status"].map(
        status_order
    )
)

boruta_group_summary = (
    boruta_group_summary
    .sort_values(
        ["_sort", "Best_Boruta_Rank", "Original_Predictor"]
    )
    .drop(columns="_sort")
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# 9. Print Boruta results
# ------------------------------------------------------------

print("\nTRAINING DATA:")
print("Training N:", len(y_boruta))
print("Low-work-intensity cases:", int(y_boruta.sum()))
print("Encoded candidate columns:", len(confirmed_mask))
print("Original candidate predictors:",
      len(model_feature_names))

print("\nBORUTA COLUMN-LEVEL SUMMARY:")

print(
    "Confirmed:",
    int(confirmed_mask.sum())
)

print(
    "Tentative:",
    int(tentative_mask.sum())
)

print(
    "Rejected:",
    int((~confirmed_mask & ~tentative_mask).sum())
)

print("\nORIGINAL PREDICTOR-LEVEL SUMMARY:")

print(
    boruta_group_summary.to_string(index=False)
)

print("\nCONFIRMED ENCODED FEATURES:")

selected_columns = boruta_column_table.loc[
    boruta_column_table["Confirmed"],
    ["Encoded_Feature", "Original_Predictor"]
]

if selected_columns.empty:
    print("No confirmed encoded features.")
else:
    print(selected_columns.to_string(index=False))

print("\nTENTATIVE ENCODED FEATURES:")

tentative_columns = boruta_column_table.loc[
    boruta_column_table["Tentative"],
    ["Encoded_Feature", "Original_Predictor"]
]

if tentative_columns.empty:
    print("No tentative encoded features.")
else:
    print(tentative_columns.to_string(index=False))

# ------------------------------------------------------------
# 10. Verify final counts and weight preservation
# ------------------------------------------------------------

assert (
    confirmed_mask.sum()
    + tentative_mask.sum()
    + (~confirmed_mask & ~tentative_mask).sum()
    == 156
)

assert len(boruta_group_summary) == 35

assert np.isclose(
    w_boruta.mean(),
    1.0
)

assert len(women_analysis) == 2110

assert (
    women_analysis["low_work_intensity"].sum()
    == 553
)

# ------------------------------------------------------------
# 11. Save reproducible outputs
# ------------------------------------------------------------

boruta_column_table.to_csv(
    os.path.join(
        output_dir,
        "Step1_29_Boruta_Encoded_Features.csv"
    ),
    index=False
)

boruta_group_summary.to_csv(
    os.path.join(
        output_dir,
        "Step1_29_Boruta_Original_Predictor_Summary.csv"
    ),
    index=False
)

print("\nSTEP 1.29 VERIFICATION SUCCESSFUL.")







# %% STEP 1.30: REUSABLE BORUTA GROUP-SELECTION PIPELINE

# ============================================================
# HIES 2022
#
# Integrates:
# 1. Training-only preprocessing
# 2. Training-only wealth PCA
# 3. Survey-weighted Boruta
# 4. Complete encoded-group retention
#
# Rule:
# If any dummy of an original predictor is confirmed,
# retain ALL encoded columns of that predictor.
#
# This is a grouped retention rule after dummy-level
# Boruta, NOT formal group-wise Boruta testing.
#
# Pilot: Outer Fold 1, Inner Fold 1
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

class LeakageSafeBorutaGroupSelector(
    BaseEstimator, TransformerMixin
):

    def __init__(
        self,
        preprocessor,
        categorical_features,
        numeric_features,
        original_features,
        n_estimators=160,
        max_iter=40,
        max_depth=5,
        min_samples_leaf=3,
        perc=100,
        alpha=0.05,
        random_state=20261030,
        n_jobs=-1
    ):

        self.preprocessor = preprocessor
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        self.original_features = original_features

        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.perc = perc
        self.alpha = alpha
        self.random_state = random_state
        self.n_jobs = n_jobs

    # --------------------------------------------------------
    # Map each encoded feature to its original predictor
    # --------------------------------------------------------

    def _identify_parent(self, name):

        if name.startswith("categorical__"):

            remainder = name[len("categorical__"):]

            matches = [
                f for f in self.categorical_features
                if remainder.startswith(f + "_")
            ]

            if len(matches) != 1:
                raise ValueError(
                    f"Ambiguous categorical feature: {name}"
                )

            return matches[0]

        if name.startswith("numerical__"):

            parent = name[len("numerical__"):]

            if parent not in self.numeric_features:
                raise ValueError(
                    f"Unexpected numeric feature: {name}"
                )

            return parent

        if name == "wealth__wealth_living_index":
            return "wealth_living_index"

        raise ValueError(
            f"Unrecognized processed feature: {name}"
        )

    # --------------------------------------------------------
    # Fit preprocessing and Boruta on training data only
    # --------------------------------------------------------

    def fit(self, X, y, sample_weight=None):

        if sample_weight is None:
            raise ValueError(
                "Survey weights must be supplied."
            )

        y_array = np.asarray(y, dtype=int)

        weights = np.asarray(
            sample_weight, dtype=float
        )

        assert len(X) == len(y_array)
        assert len(weights) == len(y_array)
        assert np.isfinite(weights).all()
        assert (weights > 0).all()

        weights = weights / weights.mean()

        # Fit ALL preprocessing on training rows only
        self.preprocessor_ = clone(
            self.preprocessor
        )

        X_processed = (
            self.preprocessor_.fit_transform(X)
        )

        X_processed = np.asarray(
            X_processed, dtype=np.float64
        )

        assert np.isfinite(X_processed).all()

        self.processed_feature_names_ = np.asarray(
            self.preprocessor_.get_feature_names_out(),
            dtype=object
        )

        assert (
            len(self.processed_feature_names_)
            == X_processed.shape[1]
        )

        self.parent_names_ = np.asarray(
            [
                self._identify_parent(name)
                for name in self.processed_feature_names_
            ],
            dtype=object
        )

        assert set(self.parent_names_) == set(
            self.original_features
        )

        # Weighted RF: wrapper created in Step 1.29
        forest = SurveyWeightedBorutaForest(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features="sqrt",
            bootstrap=True,
            class_weight=None,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )

        forest._boruta_training_weights = (
            weights.copy()
        )

        self.boruta_ = BorutaPy(
            estimator=forest,
            n_estimators=self.n_estimators,
            perc=self.perc,
            alpha=self.alpha,
            two_step=True,
            max_iter=self.max_iter,
            random_state=self.random_state,
            verbose=0
        )

        self.boruta_.fit(
            X_processed,
            y_array
        )

        self.confirmed_mask_ = np.asarray(
            self.boruta_.support_,
            dtype=bool
        )

        self.tentative_mask_ = np.asarray(
            self.boruta_.support_weak_,
            dtype=bool
        )

        self.ranking_ = np.asarray(
            self.boruta_.ranking_,
            dtype=int
        )

        # Never silently substitute rejected predictors
        if not self.confirmed_mask_.any():
            raise RuntimeError(
                "Boruta confirmed no features. "
                "Inspect this fold before modelling."
            )

        confirmed_parents = set(
            self.parent_names_[self.confirmed_mask_]
        )

        self.selected_groups_ = [
            f for f in self.original_features
            if f in confirmed_parents
        ]

        # Retain ALL encoded columns of selected parents
        self.retain_mask_ = np.isin(
            self.parent_names_,
            self.selected_groups_
        )

        self.selected_feature_names_ = (
            self.processed_feature_names_[
                self.retain_mask_
            ]
        )

        self.n_features_in_processed_ = (
            X_processed.shape[1]
        )

        return self

    # --------------------------------------------------------
    # Validation/test: transform only, never refit
    # --------------------------------------------------------

    def transform(self, X):

        if not hasattr(self, "retain_mask_"):
            raise RuntimeError(
                "Selector must be fitted first."
            )

        processed = self.preprocessor_.transform(X)

        processed = np.asarray(
            processed, dtype=np.float64
        )

        assert (
            processed.shape[1]
            == self.n_features_in_processed_
        )

        selected = processed[:, self.retain_mask_]

        if not np.isfinite(selected).all():
            raise ValueError(
                "Non-finite values after preprocessing."
            )

        return selected

    def get_feature_names_out(self, input_features=None):

        if not hasattr(self, "selected_feature_names_"):
            raise RuntimeError(
                "Selector must be fitted first."
            )

        return self.selected_feature_names_.copy()


# ------------------------------------------------------------
# 1. Create an unfitted reusable selector
# ------------------------------------------------------------

boruta_group_template = (
    LeakageSafeBorutaGroupSelector(
        preprocessor=preprocessor_template,
        categorical_features=model_categorical_features,
        numeric_features=numeric_features_without_pca,
        original_features=model_feature_names,
        n_estimators=160,
        max_iter=40,
        random_state=20261030
    )
)

# ------------------------------------------------------------
# 2. Reproduce the original pilot split
# ------------------------------------------------------------

boruta_group_demo = clone(
    boruta_group_template
)

X_train_selected = (
    boruta_group_demo.fit_transform(
        X_train_raw,
        y_boruta,
        sample_weight=w_boruta
    )
)

X_validation_selected = (
    boruta_group_demo.transform(
        X_validation_raw
    )
)

# ------------------------------------------------------------
# 3. Compare with the Step 1.29 pilot
# ------------------------------------------------------------

assert np.array_equal(
    boruta_group_demo.processed_feature_names_,
    feature_names_processed
)

assert np.array_equal(
    boruta_group_demo.confirmed_mask_,
    confirmed_mask
), "Boruta pilot was not reproduced."

expected_selected_parents = set(
    boruta_column_table.loc[
        boruta_column_table["Confirmed"],
        "Original_Predictor"
    ]
)

assert set(
    boruta_group_demo.selected_groups_
) == expected_selected_parents

# Matrices must match the previously fitted preprocessor
assert np.allclose(
    X_train_selected,
    X_train_processed[
        :, boruta_group_demo.retain_mask_
    ]
)

assert np.allclose(
    X_validation_selected,
    X_validation_processed[
        :, boruta_group_demo.retain_mask_
    ]
)

# ------------------------------------------------------------
# 4. Create a group-level audit
# ------------------------------------------------------------

group_audit_rows = []

for parent in model_feature_names:

    mask = (
        boruta_group_demo.parent_names_ == parent
    )

    group_audit_rows.append({
        "Original_Predictor": parent,
        "Encoded_Columns": int(mask.sum()),
        "Boruta_Confirmed_Columns": int(
            boruta_group_demo.confirmed_mask_[mask].sum()
        ),
        "Boruta_Tentative_Columns": int(
            boruta_group_demo.tentative_mask_[mask].sum()
        ),
        "Retained_In_Model": bool(
            boruta_group_demo.retain_mask_[mask].any()
        )
    })

boruta_group_pipeline_audit = pd.DataFrame(
    group_audit_rows
)

# ------------------------------------------------------------
# 5. Print results
# ------------------------------------------------------------

print("========== STEP 1.30 RESULTS ==========")

print("\nPILOT REPRODUCTION:")
print(
    "Encoded features before selection:",
    len(boruta_group_demo.processed_feature_names_)
)

print(
    "Boruta-confirmed columns:",
    int(boruta_group_demo.confirmed_mask_.sum())
)

print(
    "Boruta-tentative columns:",
    int(boruta_group_demo.tentative_mask_.sum())
)

print(
    "Original predictors selected:",
    len(boruta_group_demo.selected_groups_)
)

print("\nSELECTED ORIGINAL PREDICTORS:")
print(
    boruta_group_demo.selected_groups_
)

print("\nGROUP-LEVEL MODEL RETENTION:")

print(
    boruta_group_pipeline_audit.loc[
        boruta_group_pipeline_audit[
            "Retained_In_Model"
        ]
    ].to_string(index=False)
)

print("\nFINAL SELECTED MATRICES:")
print("Training:", X_train_selected.shape)
print("Validation:", X_validation_selected.shape)

print("\nSELECTED ENCODED MODEL FEATURES:")
print(
    pd.Series(
        boruta_group_demo.get_feature_names_out()
    ).to_string(index=False)
)

print("\nLEAKAGE AUDIT:")
print(
    "Training-only Boruta reproduction:",
    "PASSED"
)
print(
    "Validation preprocessing equivalence:",
    "PASSED"
)

# ------------------------------------------------------------
# 6. Save selected-feature metadata
# ------------------------------------------------------------

boruta_group_pipeline_audit.to_csv(
    os.path.join(
        output_dir,
        "Step1_30_Boruta_Group_Retention_Audit.csv"
    ),
    index=False
)

pd.DataFrame({
    "Selected_Encoded_Model_Feature":
        boruta_group_demo.get_feature_names_out()
}).to_csv(
    os.path.join(
        output_dir,
        "Step1_30_Selected_Encoded_Model_Features.csv"
    ),
    index=False
)

# ------------------------------------------------------------
# 7. Final verification
# ------------------------------------------------------------

assert X_train_selected.shape[0] == 1128
assert X_validation_selected.shape[0] == 561

assert (
    X_train_selected.shape[1]
    == X_validation_selected.shape[1]
)

assert np.isfinite(X_train_selected).all()
assert np.isfinite(X_validation_selected).all()

assert len(women_analysis) == 2110

print("\nSTEP 1.30 VERIFICATION SUCCESSFUL.")














# %% STEP 1.31: BORUTA STABILITY ACROSS INNER FOLDS

# ============================================================
# HIES 2022 - Boruta stability audit
#
# Outer Fold 1:
# - Inner Fold 1: reuse existing fitted selector
# - Inner Fold 2: fit Boruta on training only
# - Inner Fold 3: fit Boruta on training only
#
# Audit:
# - Original predictor selection frequency
# - Confirmed / tentative encoded features
# - Retained encoded columns
# - Wealth PCA explained variance
# - PSU leakage
#
# IMPORTANT:
# No hyperparameter tuning or performance evaluation.
# ============================================================

import os
import numpy as np
import pandas as pd
from sklearn.base import clone

# ------------------------------------------------------------
# 1. Define the outer fold to audit
# ------------------------------------------------------------

STABILITY_OUTER_FOLD = 0

outer1_manifest = inner_cv_manifest.loc[
    inner_cv_manifest["outer_fold"]
    == STABILITY_OUTER_FOLD
].copy()

assert len(outer1_manifest) == 1689

assert (
    outer1_manifest["inner_fold"].nunique()
    == 3
)

fold_results = []
predictor_results = []

# ------------------------------------------------------------
# 2. Process all three inner folds
# ------------------------------------------------------------

print("========== STEP 1.31 RESULTS ==========")

for inner_fold in range(3):

    print("\n" + "=" * 65)
    print(
        "OUTER FOLD 1 - INNER FOLD",
        inner_fold + 1
    )
    print("=" * 65)

    train_rows = outer1_manifest.loc[
        outer1_manifest["inner_fold"] != inner_fold
    ].copy()

    validation_rows = outer1_manifest.loc[
        outer1_manifest["inner_fold"] == inner_fold
    ].copy()

    train_idx = train_rows[
        "original_row_index"
    ].to_numpy()

    validation_idx = validation_rows[
        "original_row_index"
    ].to_numpy()

    # --------------------------------------------------------
    # 3. Verify PSU separation
    # --------------------------------------------------------

    train_groups = set(
        zip(
            train_rows["stratum16"],
            train_rows["psu"]
        )
    )

    validation_groups = set(
        zip(
            validation_rows["stratum16"],
            validation_rows["psu"]
        )
    )

    psu_overlap = len(
        train_groups.intersection(validation_groups)
    )

    assert psu_overlap == 0

    assert len(
        set(train_idx).intersection(validation_idx)
    ) == 0

    assert (
        validation_rows["stratum16"].nunique()
        == 16
    )

    # --------------------------------------------------------
    # 4. Prepare training and validation data
    # --------------------------------------------------------

    X_tr = X_raw.loc[train_idx]

    X_val = X_raw.loc[validation_idx]

    y_tr = women_analysis.loc[
        train_idx,
        "low_work_intensity"
    ].to_numpy(dtype=int)

    w_tr = women_analysis.loc[
        train_idx,
        "population_weight"
    ].to_numpy(dtype=float)

    assert len(X_tr) == len(y_tr)
    assert len(X_tr) == len(w_tr)

    assert np.isfinite(w_tr).all()
    assert (w_tr > 0).all()

    # --------------------------------------------------------
    # 5. Fit or reuse training-only Boruta selector
    # --------------------------------------------------------

    if inner_fold == 0:

        # Reuse verified Step 1.30 pilot
        selector = boruta_group_demo

        assert len(train_idx) == 1128
        assert len(validation_idx) == 561

        assert np.array_equal(
            selector.confirmed_mask_,
            confirmed_mask
        )

        print("Selector: Reused Step 1.30")

    else:

        # New selector; fit only on this inner training set
        selector = clone(
            boruta_group_template
        )

        selector.fit(
            X_tr,
            y_tr,
            sample_weight=w_tr
        )

        print("Selector: Newly fitted on training only")

    # Validation is transformed but never used in fit
    X_val_selected = selector.transform(
        X_val
    )

    assert np.isfinite(
        X_val_selected
    ).all()

    assert (
        X_val_selected.shape[1]
        == len(selector.selected_feature_names_)
    )

    # --------------------------------------------------------
    # 6. Extract fold-level Boruta results
    # --------------------------------------------------------

    confirmed_n = int(
        selector.confirmed_mask_.sum()
    )

    tentative_n = int(
        selector.tentative_mask_.sum()
    )

    rejected_n = int(
        len(selector.confirmed_mask_)
        - confirmed_n
        - tentative_n
    )

    selected_original = list(
        selector.selected_groups_
    )

    retained_encoded_n = int(
        selector.retain_mask_.sum()
    )

    pca_pipeline = (
        selector.preprocessor_
        .named_transformers_["wealth"]
    )

    pca_model = (
        pca_pipeline.named_steps["pca"]
    )

    pca_variance_percent = (
        pca_model.explained_variance_ratio_
        * 100
    )

    # --------------------------------------------------------
    # 7. Record fold-level results
    # --------------------------------------------------------

    fold_results.append({
        "Outer_Fold": 1,
        "Inner_Fold": inner_fold + 1,
        "Train_N": len(X_tr),
        "Validation_N": len(X_val),
        "Encoded_Candidates": len(
            selector.confirmed_mask_
        ),
        "Confirmed_Encoded": confirmed_n,
        "Tentative_Encoded": tentative_n,
        "Rejected_Encoded": rejected_n,
        "Selected_Original_Predictors": len(
            selected_original
        ),
        "Retained_Model_Columns": retained_encoded_n,
        "Wealth_PCA_Variance_Percent":
            pca_variance_percent,
        "PSU_Overlap": psu_overlap
    })

    # --------------------------------------------------------
    # 8. Record all 35 original predictor decisions
    # --------------------------------------------------------

    for predictor in model_feature_names:

        parent_mask = (
            selector.parent_names_ == predictor
        )

        assert parent_mask.any()

        predictor_results.append({
            "Outer_Fold": 1,
            "Inner_Fold": inner_fold + 1,
            "Original_Predictor": predictor,
            "Selected": bool(
                predictor in selected_original
            ),
            "Confirmed_Encoded_Columns": int(
                selector.confirmed_mask_[
                    parent_mask
                ].sum()
            ),
            "Tentative_Encoded_Columns": int(
                selector.tentative_mask_[
                    parent_mask
                ].sum()
            ),
            "Total_Encoded_Columns": int(
                parent_mask.sum()
            ),
            "Best_Boruta_Rank": int(
                selector.ranking_[
                    parent_mask
                ].min()
            )
        })

    # --------------------------------------------------------
    # 9. Print per-fold summary
    # --------------------------------------------------------

    print("Training N:", len(X_tr))
    print("Validation N:", len(X_val))

    print("Encoded candidates:",
          len(selector.confirmed_mask_))

    print("Confirmed encoded features:",
          confirmed_n)

    print("Tentative encoded features:",
          tentative_n)

    print("Selected original predictors:",
          len(selected_original))

    print("Retained model columns:",
          retained_encoded_n)

    print(
        "Wealth PCA explained variance (%):",
        round(pca_variance_percent, 3)
    )

    print("PSU overlap:", psu_overlap)

    print("\nSelected original predictors:")
    print(selected_original)

# ------------------------------------------------------------
# 10. Combine audit results
# ------------------------------------------------------------

boruta_stability_folds = pd.DataFrame(
    fold_results
)

boruta_stability_details = pd.DataFrame(
    predictor_results
)

assert len(boruta_stability_folds) == 3

assert len(boruta_stability_details) == 105

# ------------------------------------------------------------
# 11. Calculate original predictor selection frequency
# ------------------------------------------------------------

boruta_selection_frequency = (
    boruta_stability_details
    .groupby("Original_Predictor")
    .agg(
        Times_Selected=("Selected", "sum"),
        Total_Folds=("Selected", "size"),
        Total_Confirmed_Dummies=(
            "Confirmed_Encoded_Columns",
            "sum"
        ),
        Mean_Best_Rank=("Best_Boruta_Rank", "mean")
    )
    .reset_index()
)

boruta_selection_frequency["Selection_Percent"] = (
    100
    * boruta_selection_frequency["Times_Selected"]
    / boruta_selection_frequency["Total_Folds"]
)

boruta_selection_frequency = (
    boruta_selection_frequency
    .sort_values(
        [
            "Times_Selected",
            "Total_Confirmed_Dummies",
            "Original_Predictor"
        ],
        ascending=[False, False, True]
    )
    .reset_index(drop=True)
)

# ------------------------------------------------------------
# 12. Print stability results
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("BORUTA STABILITY SUMMARY")
print("=" * 65)

print("\nFOLD-LEVEL AUDIT:")

print(
    boruta_stability_folds
    .round(3)
    .to_string(index=False)
)

print("\nORIGINAL PREDICTOR SELECTION FREQUENCY:")

print(
    boruta_selection_frequency[
        [
            "Original_Predictor",
            "Times_Selected",
            "Total_Folds",
            "Selection_Percent",
            "Total_Confirmed_Dummies"
        ]
    ].to_string(index=False)
)

print("\nPREDICTORS SELECTED IN ALL 3 INNER FOLDS:")

selected_all_three = (
    boruta_selection_frequency.loc[
        boruta_selection_frequency[
            "Times_Selected"
        ] == 3,
        "Original_Predictor"
    ].tolist()
)

print(selected_all_three)

print("\nPREDICTORS SELECTED IN AT LEAST 1 FOLD:")

selected_any = (
    boruta_selection_frequency.loc[
        boruta_selection_frequency[
            "Times_Selected"
        ] >= 1,
        "Original_Predictor"
    ].tolist()
)

print(selected_any)

# ------------------------------------------------------------
# 13. Verify all folds and predictor counts
# ------------------------------------------------------------

assert (
    boruta_stability_folds["PSU_Overlap"]
    == 0
).all()

assert (
    boruta_stability_folds["Train_N"]
    + boruta_stability_folds["Validation_N"]
    == 1689
).all()

assert (
    boruta_stability_details
    .groupby("Inner_Fold")["Original_Predictor"]
    .nunique()
    == 35
).all()

assert (
    boruta_selection_frequency["Total_Folds"]
    == 3
).all()

assert (
    boruta_selection_frequency[
        "Times_Selected"
    ].between(0, 3)
).all()

assert len(women_analysis) == 2110

# ------------------------------------------------------------
# 14. Save results
# ------------------------------------------------------------

boruta_stability_folds.to_csv(
    os.path.join(
        output_dir,
        "Step1_31_Boruta_Inner_Fold_Audit.csv"
    ),
    index=False
)

boruta_stability_details.to_csv(
    os.path.join(
        output_dir,
        "Step1_31_Boruta_Predictor_Decisions.csv"
    ),
    index=False
)

boruta_selection_frequency.to_csv(
    os.path.join(
        output_dir,
        "Step1_31_Boruta_Selection_Frequency.csv"
    ),
    index=False
)

print("\nSTEP 1.31 VERIFICATION SUCCESSFUL.")






# %% STEP 1.32: COMPLETE BORUTA SELECTION - OUTER FOLDS 2 TO 5

# ============================================================
# HIES 2022
#
# Complete remaining 12 inner-fold Boruta analyses.
#
# Outer Fold 1: Reuse verified Step 1.31 results
# Outer Folds 2-5: Fit 3 inner folds each
#
# Includes:
# - Training-only preprocessing and PCA
# - Survey-weighted Boruta
# - Group-level feature retention
# - PSU leakage verification
# - Checkpoint saving after each completed fold
# - Selection frequency across all 15 inner folds
#
# No hyperparameter tuning or validation scoring.
# ============================================================

import os
import numpy as np
import pandas as pd
from sklearn.base import clone

# ------------------------------------------------------------
# 1. Verify previous work
# ------------------------------------------------------------

assert len(women_analysis) == 2110
assert len(X_raw) == 2110
assert len(model_feature_names) == 35

assert len(inner_cv_manifest) == 8440

assert len(boruta_stability_folds) == 3
assert len(boruta_stability_details) == 105

assert (
    boruta_stability_folds["Outer_Fold"] == 1
).all()

# ------------------------------------------------------------
# 2. Define reproducible checkpoint files
# ------------------------------------------------------------

RUN_CONFIG = (
    "HIES2022_Boruta160_Iter40_"
    "PreprocessV1_Seed20261030"
)

checkpoint_folds = os.path.join(
    output_dir,
    "Step1_32_Remaining_Boruta_Fold_Checkpoint.csv"
)

checkpoint_details = os.path.join(
    output_dir,
    "Step1_32_Remaining_Boruta_Detail_Checkpoint.csv"
)

# Load existing checkpoints if a previous run stopped.
if (
    os.path.exists(checkpoint_folds)
    and os.path.exists(checkpoint_details)
):

    new_fold_results = pd.read_csv(checkpoint_folds)
    new_predictor_results = pd.read_csv(
        checkpoint_details
    )

    assert (
        new_fold_results["Run_Config"] == RUN_CONFIG
    ).all(), "Checkpoint configuration mismatch."

    assert (
        new_predictor_results["Run_Config"] == RUN_CONFIG
    ).all(), "Checkpoint configuration mismatch."

    print("Existing checkpoints loaded.")

else:

    assert (
        not os.path.exists(checkpoint_folds)
        and not os.path.exists(checkpoint_details)
    ), "Only one checkpoint file exists; inspect files."

    new_fold_results = pd.DataFrame()
    new_predictor_results = pd.DataFrame()

    print("Starting a new Boruta checkpoint.")

# ------------------------------------------------------------
# 3. Process Outer Folds 2 to 5
# ------------------------------------------------------------

print("\n========== STEP 1.32 RESULTS ==========")

for outer_fold in range(1, 5):

    outer_manifest = inner_cv_manifest.loc[
        inner_cv_manifest["outer_fold"] == outer_fold
    ].copy()

    assert (
        outer_manifest["inner_fold"].nunique() == 3
    )

    # Verify outer-training and outer-validation
    # PSUs are completely separate.
    outer_validation = cv_manifest.loc[
        cv_manifest["outer_fold"] == outer_fold
    ]

    outer_val_groups = set(
        zip(
            outer_validation["stratum16"],
            outer_validation["psu"]
        )
    )

    for inner_fold in range(3):

        outer_number = outer_fold + 1
        inner_number = inner_fold + 1

        # ----------------------------------------------------
        # 4. Skip already completed folds
        # ----------------------------------------------------

        if not new_fold_results.empty:

            completed = (
                (new_fold_results["Outer_Fold"] == outer_number)
                & (new_fold_results["Inner_Fold"] == inner_number)
            ).any()

            if completed:

                detail_count = len(
                    new_predictor_results.loc[
                        (new_predictor_results["Outer_Fold"]
                         == outer_number)
                        & (new_predictor_results["Inner_Fold"]
                           == inner_number)
                    ]
                )

                assert detail_count == 35

                print(
                    f"Outer {outer_number}, Inner {inner_number}: "
                    "already completed - skipped."
                )

                continue

        print("\n" + "=" * 60, flush=True)
        print(
            f"FITTING OUTER {outer_number} "
            f"- INNER {inner_number}",
            flush=True
        )
        print("=" * 60, flush=True)

        # ----------------------------------------------------
        # 5. Separate inner-training and validation rows
        # ----------------------------------------------------

        train_rows = outer_manifest.loc[
            outer_manifest["inner_fold"] != inner_fold
        ]

        validation_rows = outer_manifest.loc[
            outer_manifest["inner_fold"] == inner_fold
        ]

        train_idx = train_rows[
            "original_row_index"
        ].to_numpy()

        validation_idx = validation_rows[
            "original_row_index"
        ].to_numpy()

        assert len(
            set(train_idx) & set(validation_idx)
        ) == 0

        train_groups = set(
            zip(
                train_rows["stratum16"],
                train_rows["psu"]
            )
        )

        validation_groups = set(
            zip(
                validation_rows["stratum16"],
                validation_rows["psu"]
            )
        )

        psu_overlap = len(
            train_groups & validation_groups
        )

        outer_psu_overlap = len(
            validation_groups & outer_val_groups
        )

        assert psu_overlap == 0
        assert outer_psu_overlap == 0

        assert (
            validation_rows["stratum16"].nunique() == 16
        )

        # ----------------------------------------------------
        # 6. Training inputs and survey weights
        # ----------------------------------------------------

        X_tr = X_raw.loc[train_idx]
        X_val = X_raw.loc[validation_idx]

        y_tr = women_analysis.loc[
            train_idx,
            "low_work_intensity"
        ].to_numpy(dtype=int)

        w_tr = women_analysis.loc[
            train_idx,
            "population_weight"
        ].to_numpy(dtype=float)

        assert len(X_tr) == len(y_tr) == len(w_tr)
        assert np.isfinite(w_tr).all()
        assert (w_tr > 0).all()

        # ----------------------------------------------------
        # 7. Fit preprocessing + PCA + Boruta
        # TRAINING DATA ONLY
        # ----------------------------------------------------

        selector = clone(
            boruta_group_template
        )

        selector.fit(
            X_tr,
            y_tr,
            sample_weight=w_tr
        )

        # Validation: transform only, never refit
        X_val_selected = selector.transform(X_val)

        assert np.isfinite(
            X_val_selected
        ).all()

        assert (
            X_val_selected.shape[1]
            == len(selector.selected_feature_names_)
        )

        # ----------------------------------------------------
        # 8. Extract fitted Boruta and PCA information
        # ----------------------------------------------------

        confirmed_n = int(
            selector.confirmed_mask_.sum()
        )

        tentative_n = int(
            selector.tentative_mask_.sum()
        )

        candidate_n = len(
            selector.confirmed_mask_
        )

        rejected_n = (
            candidate_n - confirmed_n - tentative_n
        )

        selected_groups = list(
            selector.selected_groups_
        )

        retained_columns = int(
            selector.retain_mask_.sum()
        )

        pca_model = (
            selector.preprocessor_
            .named_transformers_["wealth"]
            .named_steps["pca"]
        )

        explained_variance = (
            pca_model.explained_variance_ratio_ * 100
        )

        # ----------------------------------------------------
        # 9. Save fold summary
        # ----------------------------------------------------

        fold_row = pd.DataFrame([{
            "Run_Config": RUN_CONFIG,
            "Outer_Fold": outer_number,
            "Inner_Fold": inner_number,
            "Train_N": len(train_idx),
            "Validation_N": len(validation_idx),
            "Encoded_Candidates": candidate_n,
            "Confirmed_Encoded": confirmed_n,
            "Tentative_Encoded": tentative_n,
            "Rejected_Encoded": rejected_n,
            "Selected_Original_Predictors":
                len(selected_groups),
            "Retained_Model_Columns":
                retained_columns,
            "Wealth_PCA_Variance_Percent":
                explained_variance,
            "PSU_Overlap": psu_overlap
        }])

        # ----------------------------------------------------
        # 10. Save all 35 predictor decisions
        # ----------------------------------------------------

        predictor_rows = []

        for predictor in model_feature_names:

            parent_mask = (
                selector.parent_names_ == predictor
            )

            assert parent_mask.any()

            predictor_rows.append({
                "Run_Config": RUN_CONFIG,
                "Outer_Fold": outer_number,
                "Inner_Fold": inner_number,
                "Original_Predictor": predictor,
                "Selected": bool(
                    predictor in selected_groups
                ),
                "Confirmed_Encoded_Columns": int(
                    selector.confirmed_mask_[
                        parent_mask
                    ].sum()
                ),
                "Tentative_Encoded_Columns": int(
                    selector.tentative_mask_[
                        parent_mask
                    ].sum()
                ),
                "Total_Encoded_Columns": int(
                    parent_mask.sum()
                ),
                "Best_Boruta_Rank": int(
                    selector.ranking_[
                        parent_mask
                    ].min()
                )
            })

        details_new = pd.DataFrame(
            predictor_rows
        )

        # ----------------------------------------------------
        # 11. Append and checkpoint after every fold
        # ----------------------------------------------------

        new_fold_results = pd.concat(
            [new_fold_results, fold_row],
            ignore_index=True
        )

        new_predictor_results = pd.concat(
            [new_predictor_results, details_new],
            ignore_index=True
        )

        new_fold_results.to_csv(
            checkpoint_folds,
            index=False
        )

        new_predictor_results.to_csv(
            checkpoint_details,
            index=False
        )

        print(
            "Confirmed encoded features:",
            confirmed_n,
            flush=True
        )

        print(
            "Selected original predictors:",
            selected_groups,
            flush=True
        )

        print(
            "Retained columns:",
            retained_columns,
            flush=True
        )

        print(
            "PCA variance (%):",
            round(explained_variance, 3),
            flush=True
        )

        print(
            "Checkpoint saved.",
            flush=True
        )

# ------------------------------------------------------------
# 12. Combine all 15 inner folds
# ------------------------------------------------------------

assert len(new_fold_results) == 12
assert len(new_predictor_results) == 420

first_fold_audit = boruta_stability_folds.copy()

first_fold_details = boruta_stability_details.copy()

boruta_all_folds = pd.concat(
    [
        first_fold_audit,
        new_fold_results.drop(columns=["Run_Config"])
    ],
    ignore_index=True
)

boruta_all_details = pd.concat(
    [
        first_fold_details,
        new_predictor_results.drop(columns=["Run_Config"])
    ],
    ignore_index=True
)

assert len(boruta_all_folds) == 15
assert len(boruta_all_details) == 525

assert not boruta_all_folds.duplicated(
    ["Outer_Fold", "Inner_Fold"]
).any()

assert not boruta_all_details.duplicated(
    ["Outer_Fold", "Inner_Fold", "Original_Predictor"]
).any()

assert (
    boruta_all_folds["PSU_Overlap"] == 0
).all()

# ------------------------------------------------------------
# 13. Calculate selection frequency over all 15 folds
# ------------------------------------------------------------

boruta_all_frequency = (
    boruta_all_details
    .groupby("Original_Predictor")
    .agg(
        Times_Selected=("Selected", "sum"),
        Total_Folds=("Selected", "size"),
        Confirmed_Dummies_Total=(
            "Confirmed_Encoded_Columns",
            "sum"
        )
    )
    .reset_index()
)

boruta_all_frequency["Selection_Percent"] = (
    100
    * boruta_all_frequency["Times_Selected"]
    / boruta_all_frequency["Total_Folds"]
)

boruta_all_frequency = (
    boruta_all_frequency.sort_values(
        [
            "Times_Selected",
            "Confirmed_Dummies_Total",
            "Original_Predictor"
        ],
        ascending=[False, False, True]
    ).reset_index(drop=True)
)

assert len(boruta_all_frequency) == 35

assert (
    boruta_all_frequency["Total_Folds"] == 15
).all()

# ------------------------------------------------------------
# 14. Print final stability results
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("ALL 15 INNER FOLDS - BORUTA STABILITY")
print("=" * 65)

print("\nFOLD-LEVEL RESULTS:")

print(
    boruta_all_folds[
        [
            "Outer_Fold",
            "Inner_Fold",
            "Encoded_Candidates",
            "Confirmed_Encoded",
            "Tentative_Encoded",
            "Selected_Original_Predictors",
            "Retained_Model_Columns",
            "Wealth_PCA_Variance_Percent",
            "PSU_Overlap"
        ]
    ].round(3).to_string(index=False)
)

print("\nORIGINAL PREDICTOR SELECTION FREQUENCY:")

print(
    boruta_all_frequency[
        [
            "Original_Predictor",
            "Times_Selected",
            "Total_Folds",
            "Selection_Percent"
        ]
    ].round(2).to_string(index=False)
)

print("\n========== VERIFICATION ==========")

print("Total outer folds:", 5)
print("Inner folds per outer fold:", 3)
print("Total Boruta folds:", len(boruta_all_folds))
print("Total predictor decisions:", len(boruta_all_details))

print(
    "Maximum PSU overlap:",
    boruta_all_folds["PSU_Overlap"].max()
)

print(
    "Predictors selected in all 15 folds:",
    boruta_all_frequency.loc[
        boruta_all_frequency["Times_Selected"] == 15,
        "Original_Predictor"
    ].tolist()
)

# ------------------------------------------------------------
# 15. Save complete results
# ------------------------------------------------------------

boruta_all_folds.to_csv(
    os.path.join(
        output_dir,
        "Step1_32_All_15_Boruta_Fold_Audit.csv"
    ),
    index=False
)

boruta_all_details.to_csv(
    os.path.join(
        output_dir,
        "Step1_32_All_15_Boruta_Decisions.csv"
    ),
    index=False
)

boruta_all_frequency.to_csv(
    os.path.join(
        output_dir,
        "Step1_32_All_15_Boruta_Selection_Frequency.csv"
    ),
    index=False
)

print("\nSTEP 1.32 VERIFICATION SUCCESSFUL.")















# %% STEP 1.33: SIX ML MODELS - COMPATIBILITY TEST

# ============================================================
# HIES 2022 - Six original manuscript classifiers
#
# Models:
# 1. Logistic Regression
# 2. Support Vector Machine
# 3. Random Forest
# 4. Extra Trees
# 5. Gradient Boosting
# 6. AdaBoost
#
# Uses Outer Fold 1 / Inner Fold 1 TRAINING data only.
#
# Purpose:
# - Confirm model compatibility
# - Confirm survey sample_weight is accepted
# - Verify probability predictions
#
# NO hyperparameter tuning.
# NO validation performance evaluation.
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.base import clone

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

MODEL_SEED = 20261030

# ------------------------------------------------------------
# 1. Define six models with initial test settings
# These are NOT final tuned hyperparameters.
# ------------------------------------------------------------

model_templates = {

    "Logistic Regression": LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=3000,
        random_state=MODEL_SEED
    ),

    "Support Vector Machine": SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        random_state=MODEL_SEED
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_leaf=3,
        n_jobs=2,
        random_state=MODEL_SEED
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_leaf=3,
        n_jobs=2,
        random_state=MODEL_SEED
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=5,
        random_state=MODEL_SEED
    ),

    "AdaBoost": AdaBoostClassifier(
        n_estimators=150,
        learning_rate=0.05,
        random_state=MODEL_SEED
    )
}

assert len(model_templates) == 6

# ------------------------------------------------------------
# 2. Use existing training-only Boruta-selected matrix
# from Step 1.30
# ------------------------------------------------------------

X_compat = np.asarray(
    X_train_selected,
    dtype=np.float64
)

y_compat = np.asarray(
    y_boruta,
    dtype=int
)

w_compat = np.asarray(
    w_boruta,
    dtype=np.float64
)

assert X_compat.shape == (1128, 22)
assert len(y_compat) == 1128
assert len(w_compat) == 1128

assert np.isfinite(X_compat).all()
assert np.isfinite(w_compat).all()
assert (w_compat > 0).all()

assert np.array_equal(
    np.unique(y_compat),
    [0, 1]
)

assert np.isclose(
    w_compat.mean(), 1.0
)

# ------------------------------------------------------------
# 3. Fit each model on training data only
# ------------------------------------------------------------

compatibility_results = []

print("========== STEP 1.33 RESULTS ==========")

print("\nTRAINING DATA:")
print("Observations:", len(y_compat))
print("Low cases:", int(y_compat.sum()))
print("Model input columns:", X_compat.shape[1])
print("Survey weights: supplied")
print("Validation observations used: 0")

for model_name, template in model_templates.items():

    print("\n" + "=" * 60)
    print("MODEL:", model_name)
    print("=" * 60)

    model = clone(template)

    try:

        with warnings.catch_warnings(record=True) as caught:

            warnings.simplefilter("always")

            model.fit(
                X_compat,
                y_compat,
                sample_weight=w_compat
            )

            # Training-only probability output
            probabilities = model.predict_proba(
                X_compat
            )

        assert probabilities.shape == (1128, 2)

        assert np.array_equal(
            model.classes_,
            [0, 1]
        )

        assert np.isfinite(probabilities).all()

        assert (
            (probabilities >= 0)
            & (probabilities <= 1)
        ).all()

        assert np.allclose(
            probabilities.sum(axis=1),
            1.0
        )

        positive_probability = probabilities[:, 1]

        assert positive_probability.std() > 0

        status = "PASS"

        result = {
            "Model": model_name,
            "Fit_Status": status,
            "Sample_Weight_Accepted": True,
            "Probability_Output_Valid": True,
            "Probability_Min": float(
                positive_probability.min()
            ),
            "Probability_Max": float(
                positive_probability.max()
            ),
            "Probability_SD": float(
                positive_probability.std()
            ),
            "Warning_Count": len(caught),
            "Error": ""
        }

        print("Model fit:", status)
        print("Probability output: PASS")

        print(
            "Predicted probability range:",
            f"{positive_probability.min():.5f}",
            "to",
            f"{positive_probability.max():.5f}"
        )

        print("Warnings:", len(caught))

        for item in caught[:3]:
            print(
                "Warning:",
                str(item.message)[:250]
            )

    except Exception as exc:

        result = {
            "Model": model_name,
            "Fit_Status": "FAIL",
            "Sample_Weight_Accepted": False,
            "Probability_Output_Valid": False,
            "Probability_Min": np.nan,
            "Probability_Max": np.nan,
            "Probability_SD": np.nan,
            "Warning_Count": np.nan,
            "Error": repr(exc)
        }

        print("Model fit: FAIL")
        print("Error:", repr(exc))

    compatibility_results.append(result)

# ------------------------------------------------------------
# 4. Build compatibility summary
# ------------------------------------------------------------

model_compatibility_table = pd.DataFrame(
    compatibility_results
)

print("\n========== COMPATIBILITY SUMMARY ==========")

print(
    model_compatibility_table[
        [
            "Model",
            "Fit_Status",
            "Sample_Weight_Accepted",
            "Probability_Output_Valid",
            "Warning_Count"
        ]
    ].to_string(index=False)
)

print("\nTotal models:",
      len(model_compatibility_table))

print(
    "Models passed:",
    int(
        (
            model_compatibility_table[
                "Fit_Status"
            ] == "PASS"
        ).sum()
    )
)

# ------------------------------------------------------------
# 5. Save results
# ------------------------------------------------------------

model_compatibility_table.to_csv(
    os.path.join(
        output_dir,
        "Step1_33_Six_Model_Compatibility.csv"
    ),
    index=False
)

# ------------------------------------------------------------
# 6. Final verification
# ------------------------------------------------------------

assert len(model_compatibility_table) == 6

assert (
    model_compatibility_table[
        "Fit_Status"
    ] == "PASS"
).all(), "A model failed. Inspect console output."

assert (
    model_compatibility_table[
        "Probability_Output_Valid"
    ]
).all()

assert len(women_analysis) == 2110

print("\nSTEP 1.33 VERIFICATION SUCCESSFUL.")




# %% STEP 1.34: DEFINE SIX-MODEL HYPERPARAMETER GRIDS

# ============================================================
# HIES 2022 - Reproducible hyperparameter search plan
#
# Six original manuscript models.
#
# Primary tuning metric:
# Survey-weighted Average Precision (AP).
#
# Secondary metric:
# Survey-weighted ROC-AUC.
#
# IMPORTANT:
# - This step defines candidate settings only.
# - No training or validation evaluation occurs.
# - No outer-validation data are used.
# ============================================================

import os
import json
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import ParameterGrid

# ------------------------------------------------------------
# 1. Preserve the six verified model templates
# ------------------------------------------------------------

tuned_model_templates = {
    name: clone(model)
    for name, model in model_templates.items()
}

# Logistic Regression: both L1 and L2 regularization
# require a compatible solver.
tuned_model_templates[
    "Logistic Regression"
].set_params(
    solver="liblinear",
    max_iter=3000
)

# Avoid deprecated SAMME.R in sklearn 1.5.2.
tuned_model_templates[
    "AdaBoost"
].set_params(
    algorithm="SAMME"
)

# ------------------------------------------------------------
# 2. Define modest, explicit hyperparameter grids
# ------------------------------------------------------------

tuning_grids = {

    "Logistic Regression": {
        "C": [0.03, 0.3, 3.0, 30.0],
        "penalty": ["l1", "l2"]
    },

    "Support Vector Machine": {
        "C": [0.1, 1.0, 10.0],
        "gamma": ["scale", 0.03, 0.3]
    },

    "Random Forest": {
        "max_depth": [None, 8, 16],
        "min_samples_leaf": [2, 8],
        "max_features": ["sqrt", 0.7]
    },

    "Extra Trees": {
        "max_depth": [None, 8],
        "min_samples_leaf": [2, 8],
        "max_features": ["sqrt", 0.7]
    },

    "Gradient Boosting": {
        "n_estimators": [100, 250],
        "learning_rate": [0.03, 0.10],
        "max_depth": [1, 3]
    },

    "AdaBoost": {
        "n_estimators": [50, 150, 300],
        "learning_rate": [0.03, 0.10, 0.50]
    }
}

assert set(tuning_grids) == set(tuned_model_templates)

# ------------------------------------------------------------
# 3. Define tuning criteria and CV design
# ------------------------------------------------------------

TUNING_PRIMARY_METRIC = "weighted_average_precision"
TUNING_SECONDARY_METRIC = "weighted_roc_auc"

TUNING_OUTER_FOLDS = 5
TUNING_INNER_FOLDS = 3

# ------------------------------------------------------------
# 4. Expand grids and validate estimator parameters
# ------------------------------------------------------------

tuning_candidate_rows = []
tuning_summary_rows = []

for model_name, grid in tuning_grids.items():

    candidates = list(ParameterGrid(grid))

    # Check that every setting is supported
    # by the installed sklearn version.
    for candidate_number, params in enumerate(
        candidates, start=1
    ):

        trial_model = clone(
            tuned_model_templates[model_name]
        )

        trial_model.set_params(**params)

        tuning_candidate_rows.append({
            "Model": model_name,
            "Candidate_ID": candidate_number,
            "Parameters_JSON": json.dumps(
                params, sort_keys=True
            )
        })

    tuning_summary_rows.append({
        "Model": model_name,
        "Hyperparameters_Tuned": ", ".join(
            grid.keys()
        ),
        "Candidate_Combinations": len(candidates),
        "Planned_Inner_Fits": (
            len(candidates)
            * TUNING_OUTER_FOLDS
            * TUNING_INNER_FOLDS
        )
    })

tuning_candidates = pd.DataFrame(
    tuning_candidate_rows
)

tuning_grid_summary = pd.DataFrame(
    tuning_summary_rows
)

# ------------------------------------------------------------
# 5. Check AdaBoost SAMME compatibility on TRAINING only
# ------------------------------------------------------------

ada_samme = clone(
    tuned_model_templates["AdaBoost"]
)

ada_samme.fit(
    X_train_selected,
    y_boruta,
    sample_weight=w_boruta
)

ada_probabilities = ada_samme.predict_proba(
    X_train_selected
)

assert ada_probabilities.shape == (1128, 2)
assert np.isfinite(ada_probabilities).all()
assert np.allclose(
    ada_probabilities.sum(axis=1), 1.0
)

# ------------------------------------------------------------
# 6. Verify candidate counts
# ------------------------------------------------------------

expected_counts = {
    "Logistic Regression": 8,
    "Support Vector Machine": 9,
    "Random Forest": 12,
    "Extra Trees": 8,
    "Gradient Boosting": 8,
    "AdaBoost": 9
}

for model_name, expected in expected_counts.items():

    actual = len(
        list(ParameterGrid(
            tuning_grids[model_name]
        ))
    )

    assert actual == expected

assert len(tuning_candidates) == 54

assert (
    tuning_grid_summary[
        "Planned_Inner_Fits"
    ].sum() == 810
)

assert (
    len(
        tuning_candidates[
            ["Model", "Candidate_ID"]
        ].drop_duplicates()
    ) == 54
)

# ------------------------------------------------------------
# 7. Print results
# ------------------------------------------------------------

print("========== STEP 1.34 RESULTS ==========")

print("\nHYPERPARAMETER GRID SUMMARY:")

print(
    tuning_grid_summary.to_string(index=False)
)

print("\nTUNING DESIGN:")
print(
    "Primary metric:",
    TUNING_PRIMARY_METRIC
)

print(
    "Secondary metric:",
    TUNING_SECONDARY_METRIC
)

print("Outer folds:", TUNING_OUTER_FOLDS)
print("Inner folds:", TUNING_INNER_FOLDS)

print(
    "Total candidate combinations:",
    len(tuning_candidates)
)

print(
    "Total planned inner model fits:",
    int(
        tuning_grid_summary[
            "Planned_Inner_Fits"
        ].sum()
    )
)

print("\nADABOOST SAMME CHECK:")
print("Algorithm:", ada_samme.algorithm)
print(
    "Valid training probabilities:",
    bool(np.isfinite(ada_probabilities).all())
)

# ------------------------------------------------------------
# 8. Save reproducible tuning design
# ------------------------------------------------------------

tuning_candidates.to_csv(
    os.path.join(
        output_dir,
        "Step1_34_Hyperparameter_Candidates.csv"
    ),
    index=False
)

tuning_grid_summary.to_csv(
    os.path.join(
        output_dir,
        "Step1_34_Hyperparameter_Grid_Summary.csv"
    ),
    index=False
)

# ------------------------------------------------------------
# 9. Final verification
# ------------------------------------------------------------

assert len(women_analysis) == 2110
assert len(boruta_all_folds) == 15
assert len(model_templates) == 6

print("\nSTEP 1.34 VERIFICATION SUCCESSFUL.")





# %% STEP 1.35: LOGISTIC REGRESSION INNER-CV TUNING - OUTER 1

# ============================================================
# HIES 2022 - First genuine nested-CV tuning experiment
#
# Model: Logistic Regression
# Outer fold: 1
# Inner folds: 3
# Hyperparameter combinations: 8
#
# PRIMARY: Survey-weighted Average Precision (AP)
# SECONDARY: Survey-weighted ROC-AUC
#
# Preprocessing, PCA, Boruta:
# Fitted using each INNER-TRAINING set only.
#
# Outer-validation data remain completely unused.
# ============================================================

import os
import json
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score
)

OUTER_FOLD_135 = 0

cache_folder_135 = os.path.join(
    output_dir,
    "Step1_35_Inner_Fold_Cache"
)

os.makedirs(cache_folder_135, exist_ok=True)

params_135 = list(
    ParameterGrid(
        tuning_grids["Logistic Regression"]
    )
)

assert len(params_135) == 8

manifest_135 = inner_cv_manifest.loc[
    inner_cv_manifest["outer_fold"] == OUTER_FOLD_135
].copy()

assert len(manifest_135) == 1689

score_rows_135 = []
cache_audit_rows_135 = []

print("========== STEP 1.35 RESULTS ==========")

# ------------------------------------------------------------
# 1. Process all three inner folds
# ------------------------------------------------------------

for inner_fold in range(3):

    fold_number = inner_fold + 1

    print("\n" + "=" * 60, flush=True)
    print(
        "OUTER 1 - INNER",
        fold_number,
        flush=True
    )
    print("=" * 60, flush=True)

    train_rows = manifest_135.loc[
        manifest_135["inner_fold"] != inner_fold
    ]

    val_rows = manifest_135.loc[
        manifest_135["inner_fold"] == inner_fold
    ]

    train_idx_135 = train_rows[
        "original_row_index"
    ].to_numpy()

    val_idx_135 = val_rows[
        "original_row_index"
    ].to_numpy()

    # No observation or PSU overlap
    assert not (
        set(train_idx_135) & set(val_idx_135)
    )

    train_psus = set(zip(
        train_rows["stratum16"],
        train_rows["psu"]
    ))

    val_psus = set(zip(
        val_rows["stratum16"],
        val_rows["psu"]
    ))

    assert not (train_psus & val_psus)

    # Verify these are INNER folds of Outer Fold 1
    assert (
        cv_manifest.loc[
            train_idx_135, "outer_fold"
        ] != OUTER_FOLD_135
    ).all()

    assert (
        cv_manifest.loc[
            val_idx_135, "outer_fold"
        ] != OUTER_FOLD_135
    ).all()

    y_tr_135 = women_analysis.loc[
        train_idx_135,
        "low_work_intensity"
    ].to_numpy(dtype=int)

    y_val_135 = women_analysis.loc[
        val_idx_135,
        "low_work_intensity"
    ].to_numpy(dtype=int)

    w_tr_135 = women_analysis.loc[
        train_idx_135,
        "population_weight"
    ].to_numpy(dtype=float)

    w_val_135 = women_analysis.loc[
        val_idx_135,
        "population_weight"
    ].to_numpy(dtype=float)

    assert np.isfinite(w_tr_135).all()
    assert np.isfinite(w_val_135).all()
    assert (w_tr_135 > 0).all()
    assert (w_val_135 > 0).all()

    cache_path = os.path.join(
        cache_folder_135,
        f"Outer1_Inner{fold_number}_Boruta_Matrices.npz"
    )

    # --------------------------------------------------------
    # 2. Reuse previously cached transformations if available
    # --------------------------------------------------------

    if os.path.exists(cache_path):

        with np.load(
            cache_path,
            allow_pickle=False
        ) as cached:

            assert cached["version"].item() == (
                "HIES2022_Boruta_InnerCache_v1"
            )

            assert np.array_equal(
                cached["train_idx"],
                train_idx_135
            )

            assert np.array_equal(
                cached["val_idx"],
                val_idx_135
            )

            X_tr_selected_135 = cached[
                "X_train"
            ].copy()

            X_val_selected_135 = cached[
                "X_validation"
            ].copy()

            selected_groups_135 = (
                cached["selected_groups"]
                .tolist()
            )

            selected_names_135 = (
                cached["feature_names"]
                .tolist()
            )

        print(
            "Previously fitted training-only "
            "matrices loaded.",
            flush=True
        )

    else:

        # ----------------------------------------------------
        # 3. Fit preprocessing, PCA and Boruta
        # on INNER-TRAINING observations only
        # ----------------------------------------------------

        if inner_fold == 0:

            selector_135 = boruta_group_demo

            # Verify that the previously fitted selector
            # belongs to exactly this training split.
            assert np.array_equal(
                train_idx_135,
                train_indices
            )

            print(
                "Reusing verified Inner Fold 1 selector.",
                flush=True
            )

        else:

            selector_135 = clone(
                boruta_group_template
            )

            selector_135.fit(
                X_raw.loc[train_idx_135],
                y_tr_135,
                sample_weight=w_tr_135
            )

            print(
                "New training-only Boruta fit completed.",
                flush=True
            )

        X_tr_selected_135 = selector_135.transform(
            X_raw.loc[train_idx_135]
        )

        X_val_selected_135 = selector_135.transform(
            X_raw.loc[val_idx_135]
        )

        selected_groups_135 = list(
            selector_135.selected_groups_
        )

        selected_names_135 = list(
            selector_135.get_feature_names_out()
        )

        # ----------------------------------------------------
        # 4. Verify reproducibility against Step 1.31
        # ----------------------------------------------------

        old_decisions = boruta_stability_details.loc[
            (
                boruta_stability_details["Outer_Fold"]
                == 1
            )
            & (
                boruta_stability_details["Inner_Fold"]
                == fold_number
            )
        ]

        expected_groups = set(
            old_decisions.loc[
                old_decisions["Selected"],
                "Original_Predictor"
            ]
        )

        assert set(selected_groups_135) == expected_groups

        # Cache training-only matrices for other classifiers
        np.savez_compressed(
            cache_path,
            version=np.array(
                "HIES2022_Boruta_InnerCache_v1"
            ),
            train_idx=train_idx_135,
            val_idx=val_idx_135,
            X_train=X_tr_selected_135,
            X_validation=X_val_selected_135,
            selected_groups=np.asarray(
                selected_groups_135, dtype=str
            ),
            feature_names=np.asarray(
                selected_names_135, dtype=str
            )
        )

        print(
            "Training-only matrices cached.",
            flush=True
        )

    # --------------------------------------------------------
    # 5. Validate cached or newly transformed matrices
    # --------------------------------------------------------

    assert X_tr_selected_135.shape[0] == len(y_tr_135)
    assert X_val_selected_135.shape[0] == len(y_val_135)

    assert (
        X_tr_selected_135.shape[1]
        == X_val_selected_135.shape[1]
    )

    assert (
        X_tr_selected_135.shape[1]
        == len(selected_names_135)
    )

    assert np.isfinite(X_tr_selected_135).all()
    assert np.isfinite(X_val_selected_135).all()

    old_fold = boruta_stability_folds.loc[
        boruta_stability_folds["Inner_Fold"]
        == fold_number
    ].iloc[0]

    assert (
        X_tr_selected_135.shape[1]
        == old_fold["Retained_Model_Columns"]
    )

    cache_audit_rows_135.append({
        "Inner_Fold": fold_number,
        "Train_N": len(y_tr_135),
        "Validation_N": len(y_val_135),
        "Retained_Columns": X_tr_selected_135.shape[1],
        "Selected_Groups": ", ".join(
            selected_groups_135
        ),
        "PSU_Overlap": 0
    })

    print("Training N:", len(y_tr_135))
    print("Validation N:", len(y_val_135))
    print("Selected columns:", X_tr_selected_135.shape[1])
    print("Selected groups:", selected_groups_135)

    # --------------------------------------------------------
    # 6. Evaluate all eight Logistic Regression candidates
    # --------------------------------------------------------

    # Normalize training weights for comparable
    # regularization strength across inner folds.
    fitting_weights = (
        w_tr_135 / w_tr_135.mean()
    )

    for candidate_id, params in enumerate(
        params_135, start=1
    ):

        model = clone(
            tuned_model_templates[
                "Logistic Regression"
            ]
        )

        model.set_params(**params)

        model.fit(
            X_tr_selected_135,
            y_tr_135,
            sample_weight=fitting_weights
        )

        # INNER-validation predictions only
        probabilities = model.predict_proba(
            X_val_selected_135
        )[:, 1]

        assert np.isfinite(probabilities).all()

        weighted_ap = average_precision_score(
            y_val_135,
            probabilities,
            sample_weight=w_val_135
        )

        weighted_auc = roc_auc_score(
            y_val_135,
            probabilities,
            sample_weight=w_val_135
        )

        score_rows_135.append({
            "Outer_Fold": 1,
            "Inner_Fold": fold_number,
            "Model": "Logistic Regression",
            "Candidate_ID": candidate_id,
            "Parameters_JSON": json.dumps(
                params, sort_keys=True
            ),
            "Weighted_AP": weighted_ap,
            "Weighted_ROC_AUC": weighted_auc
        })

    print(
        "Eight LR candidates evaluated.",
        flush=True
    )

# ------------------------------------------------------------
# 7. Aggregate the three inner-fold scores
# ------------------------------------------------------------

lr_inner_scores_135 = pd.DataFrame(
    score_rows_135
)

assert len(lr_inner_scores_135) == 24

lr_tuning_summary_135 = (
    lr_inner_scores_135
    .groupby(
        ["Candidate_ID", "Parameters_JSON"],
        as_index=False
    )
    .agg(
        Inner_Folds=("Inner_Fold", "nunique"),
        Mean_Weighted_AP=("Weighted_AP", "mean"),
        SD_Weighted_AP=("Weighted_AP", "std"),
        Mean_Weighted_ROC_AUC=(
            "Weighted_ROC_AUC", "mean"
        )
    )
)

assert len(lr_tuning_summary_135) == 8

assert (
    lr_tuning_summary_135["Inner_Folds"] == 3
).all()

# Select using primary AP; ROC-AUC breaks ties.
lr_tuning_summary_135 = (
    lr_tuning_summary_135
    .sort_values(
        [
            "Mean_Weighted_AP",
            "Mean_Weighted_ROC_AUC",
            "Candidate_ID"
        ],
        ascending=[False, False, True]
    )
    .reset_index(drop=True)
)

lr_tuning_summary_135.insert(
    0,
    "Rank",
    np.arange(1, 9)
)

best_lr_params_outer1 = json.loads(
    lr_tuning_summary_135.iloc[0][
        "Parameters_JSON"
    ]
)

# ------------------------------------------------------------
# 8. Print tuning results
# ------------------------------------------------------------

print("\n========== STEP 1.35 TUNING SUMMARY ==========")

print("\nINNER-FOLD PREPROCESSING AUDIT:")

lr_cache_audit_135 = pd.DataFrame(
    cache_audit_rows_135
)

print(
    lr_cache_audit_135[
        [
            "Inner_Fold",
            "Train_N",
            "Validation_N",
            "Retained_Columns",
            "PSU_Overlap"
        ]
    ].to_string(index=False)
)

print("\nLOGISTIC REGRESSION CANDIDATE RANKING:")

print(
    lr_tuning_summary_135.round(5)
    .to_string(index=False)
)

print("\nBEST INNER-CV PARAMETERS:")
print(best_lr_params_outer1)

print(
    "Best mean weighted AP:",
    round(
        lr_tuning_summary_135.iloc[0][
            "Mean_Weighted_AP"
        ],
        5
    )
)

print(
    "Corresponding mean weighted ROC-AUC:",
    round(
        lr_tuning_summary_135.iloc[0][
            "Mean_Weighted_ROC_AUC"
        ],
        5
    )
)

# ------------------------------------------------------------
# 9. Save reproducible tuning outputs
# ------------------------------------------------------------

lr_inner_scores_135.to_csv(
    os.path.join(
        output_dir,
        "Step1_35_LR_Outer1_Inner_Scores.csv"
    ),
    index=False
)

lr_tuning_summary_135.to_csv(
    os.path.join(
        output_dir,
        "Step1_35_LR_Outer1_Tuning_Ranking.csv"
    ),
    index=False
)

lr_cache_audit_135.to_csv(
    os.path.join(
        output_dir,
        "Step1_35_Inner_Preprocessing_Cache_Audit.csv"
    ),
    index=False
)

# ------------------------------------------------------------
# 10. Final verification
# ------------------------------------------------------------

assert len(lr_inner_scores_135) == 24

assert lr_tuning_summary_135[
    "Mean_Weighted_AP"
].between(0, 1).all()

assert lr_tuning_summary_135[
    "Mean_Weighted_ROC_AUC"
].between(0, 1).all()

assert (
    lr_cache_audit_135["PSU_Overlap"] == 0
).all()

assert len(women_analysis) == 2110

print("\nOuter-validation observations used: 0")
print("Total LR inner model fits: 24")
print("STEP 1.35 VERIFICATION SUCCESSFUL.")






# %% STEP 1.36: SVM NESTED-CV TUNING - OUTER FOLD 1

# ============================================================
# HIES 2022
# Model: Support Vector Machine (RBF kernel)
#
# Outer Fold: 1
# Inner Folds: 3
# Hyperparameter combinations: 9
# Total model fits: 27
#
# PRIMARY METRIC:
# Survey-weighted Average Precision
#
# SECONDARY METRIC:
# Survey-weighted ROC-AUC
#
# Uses cached, training-only:
# - Preprocessing
# - Wealth PCA
# - Boruta feature selection
#
# SVM decision scores are used for ranking metrics.
# No probability calibration or outer-fold evaluation.
# ============================================================

import os
import json
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score
)

# ------------------------------------------------------------
# 1. Verify tuning configuration
# ------------------------------------------------------------

SVM_OUTER_FOLD = 0

svm_candidates = list(
    ParameterGrid(
        tuning_grids["Support Vector Machine"]
    )
)

assert len(svm_candidates) == 9
assert len(women_analysis) == 2110

assert (
    "Support Vector Machine"
    in tuned_model_templates
)

assert os.path.isdir(cache_folder_135)

# ------------------------------------------------------------
# 2. Identify the correct outer-training sample
# ------------------------------------------------------------

svm_manifest = inner_cv_manifest.loc[
    inner_cv_manifest["outer_fold"]
    == SVM_OUTER_FOLD
].copy()

assert len(svm_manifest) == 1689

outer_validation_indices = set(
    cv_manifest.index[
        cv_manifest["outer_fold"]
        == SVM_OUTER_FOLD
    ]
)

assert len(outer_validation_indices) == 421

svm_score_rows = []
svm_cache_rows = []

print("========== STEP 1.36 RESULTS ==========")

# ------------------------------------------------------------
# 3. Evaluate all three inner folds
# ------------------------------------------------------------

for inner_fold in range(3):

    fold_number = inner_fold + 1

    print("\n" + "=" * 60, flush=True)
    print(
        f"SVM: OUTER 1 - INNER {fold_number}",
        flush=True
    )
    print("=" * 60, flush=True)

    train_rows = svm_manifest.loc[
        svm_manifest["inner_fold"] != inner_fold
    ]

    val_rows = svm_manifest.loc[
        svm_manifest["inner_fold"] == inner_fold
    ]

    train_idx = train_rows[
        "original_row_index"
    ].to_numpy()

    val_idx = val_rows[
        "original_row_index"
    ].to_numpy()

    # --------------------------------------------------------
    # 4. Verify observation and PSU isolation
    # --------------------------------------------------------

    assert not (
        set(train_idx) & set(val_idx)
    )

    assert not (
        set(train_idx) & outer_validation_indices
    )

    assert not (
        set(val_idx) & outer_validation_indices
    )

    train_psus = set(zip(
        train_rows["stratum16"],
        train_rows["psu"]
    ))

    val_psus = set(zip(
        val_rows["stratum16"],
        val_rows["psu"]
    ))

    assert not (train_psus & val_psus)

    # --------------------------------------------------------
    # 5. Load cached preprocessing + Boruta matrices
    # --------------------------------------------------------

    cache_path = os.path.join(
        cache_folder_135,
        f"Outer1_Inner{fold_number}_Boruta_Matrices.npz"
    )

    assert os.path.exists(cache_path), (
        f"Missing cache: {cache_path}"
    )

    with np.load(
        cache_path,
        allow_pickle=False
    ) as cached:

        assert cached["version"].item() == (
            "HIES2022_Boruta_InnerCache_v1"
        )

        assert np.array_equal(
            cached["train_idx"], train_idx
        )

        assert np.array_equal(
            cached["val_idx"], val_idx
        )

        X_tr = cached["X_train"].copy()

        X_val = cached["X_validation"].copy()

        selected_groups = (
            cached["selected_groups"].tolist()
        )

        feature_names = (
            cached["feature_names"].tolist()
        )

    # --------------------------------------------------------
    # 6. Verify cached selection against Step 1.31
    # --------------------------------------------------------

    previous_decisions = (
        boruta_stability_details.loc[
            (boruta_stability_details["Outer_Fold"] == 1)
            &
            (boruta_stability_details["Inner_Fold"]
             == fold_number)
        ]
    )

    expected_groups = set(
        previous_decisions.loc[
            previous_decisions["Selected"],
            "Original_Predictor"
        ]
    )

    assert set(selected_groups) == expected_groups

    expected_columns = int(
        boruta_stability_folds.loc[
            boruta_stability_folds["Inner_Fold"]
            == fold_number,
            "Retained_Model_Columns"
        ].iloc[0]
    )

    assert X_tr.shape[1] == expected_columns
    assert X_val.shape[1] == expected_columns

    assert len(feature_names) == expected_columns

    assert np.isfinite(X_tr).all()
    assert np.isfinite(X_val).all()

    # --------------------------------------------------------
    # 7. Retrieve outcomes and survey weights
    # --------------------------------------------------------

    y_tr = women_analysis.loc[
        train_idx,
        "low_work_intensity"
    ].to_numpy(dtype=int)

    y_val = women_analysis.loc[
        val_idx,
        "low_work_intensity"
    ].to_numpy(dtype=int)

    w_tr = women_analysis.loc[
        train_idx,
        "population_weight"
    ].to_numpy(dtype=float)

    w_val = women_analysis.loc[
        val_idx,
        "population_weight"
    ].to_numpy(dtype=float)

    assert len(X_tr) == len(y_tr) == len(w_tr)
    assert len(X_val) == len(y_val) == len(w_val)

    assert np.isfinite(w_tr).all()
    assert np.isfinite(w_val).all()

    assert (w_tr > 0).all()
    assert (w_val > 0).all()

    assert set(np.unique(y_tr)) == {0, 1}
    assert set(np.unique(y_val)) == {0, 1}

    # Normalize fitting weights.
    # Use original survey weights for evaluation.
    fitting_weights = w_tr / w_tr.mean()

    svm_cache_rows.append({
        "Outer_Fold": 1,
        "Inner_Fold": fold_number,
        "Train_N": len(y_tr),
        "Validation_N": len(y_val),
        "Retained_Columns": X_tr.shape[1],
        "PSU_Overlap": 0
    })

    print("Training N:", len(y_tr))
    print("Validation N:", len(y_val))
    print("Retained columns:", X_tr.shape[1])
    print("Selected predictors:", selected_groups)

    # --------------------------------------------------------
    # 8. Fit and evaluate nine SVM candidates
    # --------------------------------------------------------

    for candidate_id, params in enumerate(
        svm_candidates,
        start=1
    ):

        svm_model = clone(
            tuned_model_templates[
                "Support Vector Machine"
            ]
        )

        svm_model.set_params(
            **params,
            probability=False
        )

        svm_model.fit(
            X_tr,
            y_tr,
            sample_weight=fitting_weights
        )

        assert np.array_equal(
            svm_model.classes_, [0, 1]
        )

        # Positive decision score corresponds to class 1.
        validation_scores = (
            svm_model.decision_function(X_val)
        )

        assert validation_scores.shape == (
            len(y_val),
        )

        assert np.isfinite(
            validation_scores
        ).all()

        weighted_ap = average_precision_score(
            y_val,
            validation_scores,
            sample_weight=w_val
        )

        weighted_auc = roc_auc_score(
            y_val,
            validation_scores,
            sample_weight=w_val
        )

        assert 0 <= weighted_ap <= 1
        assert 0 <= weighted_auc <= 1

        svm_score_rows.append({
            "Outer_Fold": 1,
            "Inner_Fold": fold_number,
            "Model": "Support Vector Machine",
            "Candidate_ID": candidate_id,
            "Parameters_JSON": json.dumps(
                params,
                sort_keys=True
            ),
            "Weighted_AP": weighted_ap,
            "Weighted_ROC_AUC": weighted_auc
        })

    print(
        "Nine SVM candidates evaluated successfully.",
        flush=True
    )

# ------------------------------------------------------------
# 9. Aggregate tuning scores across inner folds
# ------------------------------------------------------------

svm_inner_scores_136 = pd.DataFrame(
    svm_score_rows
)

assert len(svm_inner_scores_136) == 27

svm_tuning_summary_136 = (
    svm_inner_scores_136
    .groupby(
        ["Candidate_ID", "Parameters_JSON"],
        as_index=False
    )
    .agg(
        Inner_Folds=("Inner_Fold", "nunique"),
        Mean_Weighted_AP=("Weighted_AP", "mean"),
        SD_Weighted_AP=("Weighted_AP", "std"),
        Mean_Weighted_ROC_AUC=(
            "Weighted_ROC_AUC", "mean"
        )
    )
)

assert len(svm_tuning_summary_136) == 9

assert (
    svm_tuning_summary_136["Inner_Folds"] == 3
).all()

# Primary: weighted AP
# Secondary: weighted ROC-AUC
# Final tie-break: candidate ID
svm_tuning_summary_136 = (
    svm_tuning_summary_136
    .sort_values(
        [
            "Mean_Weighted_AP",
            "Mean_Weighted_ROC_AUC",
            "Candidate_ID"
        ],
        ascending=[False, False, True]
    )
    .reset_index(drop=True)
)

svm_tuning_summary_136.insert(
    0,
    "Rank",
    np.arange(1, 10)
)

best_svm_params_outer1 = json.loads(
    svm_tuning_summary_136.iloc[0][
        "Parameters_JSON"
    ]
)

# ------------------------------------------------------------
# 10. Display complete tuning results
# ------------------------------------------------------------

print("\n========== STEP 1.36 TUNING SUMMARY ==========")

print("\nINNER-FOLD MATRIX AUDIT:")

svm_cache_audit_136 = pd.DataFrame(
    svm_cache_rows
)

print(
    svm_cache_audit_136.to_string(index=False)
)

print("\nSVM HYPERPARAMETER RANKING:")

print(
    svm_tuning_summary_136.round(5)
    .to_string(index=False)
)

print("\nBEST SVM PARAMETERS:")
print(best_svm_params_outer1)

print(
    "Best mean weighted AP:",
    round(
        svm_tuning_summary_136.iloc[0][
            "Mean_Weighted_AP"
        ],
        5
    )
)

print(
    "Corresponding mean weighted ROC-AUC:",
    round(
        svm_tuning_summary_136.iloc[0][
            "Mean_Weighted_ROC_AUC"
        ],
        5
    )
)

# ------------------------------------------------------------
# 11. Save results
# ------------------------------------------------------------

svm_inner_scores_136.to_csv(
    os.path.join(
        output_dir,
        "Step1_36_SVM_Outer1_Inner_Scores.csv"
    ),
    index=False
)

svm_tuning_summary_136.to_csv(
    os.path.join(
        output_dir,
        "Step1_36_SVM_Outer1_Tuning_Ranking.csv"
    ),
    index=False
)

svm_cache_audit_136.to_csv(
    os.path.join(
        output_dir,
        "Step1_36_SVM_Cache_Audit.csv"
    ),
    index=False
)

# ------------------------------------------------------------
# 12. Final verification
# ------------------------------------------------------------

assert len(svm_inner_scores_136) == 27

assert svm_tuning_summary_136[
    "Mean_Weighted_AP"
].between(0, 1).all()

assert svm_tuning_summary_136[
    "Mean_Weighted_ROC_AUC"
].between(0, 1).all()

assert (
    svm_cache_audit_136["PSU_Overlap"] == 0
).all()

assert (
    svm_cache_audit_136["Retained_Columns"]
    .tolist() == [22, 18, 13]
)

print("\nOuter-validation observations used: 0")
print("Total SVM inner model fits: 27")
print("STEP 1.36 VERIFICATION SUCCESSFUL.")



# %% STEP 1.37: RANDOM FOREST INNER-CV TUNING - OUTER 1

# ============================================================
# HIES 2022 - Random Forest nested-CV tuning
#
# Outer fold: 1
# Inner folds: 3
# Hyperparameter combinations: 12
# Total model fits: 36
#
# PRIMARY: Survey-weighted Average Precision (AP)
# SECONDARY: Survey-weighted ROC-AUC
#
# Reuses training-only preprocessing/PCA/Boruta caches.
# Outer-validation observations are never used.
# ============================================================

import os
import json
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score
)

# ------------------------------------------------------------
# 1. Configure the Random Forest experiment
# ------------------------------------------------------------

RF_OUTER_FOLD = 0

rf_candidates = list(
    ParameterGrid(
        tuning_grids["Random Forest"]
    )
)

assert len(rf_candidates) == 12

assert "Random Forest" in tuned_model_templates

assert os.path.isdir(cache_folder_135)

rf_manifest = inner_cv_manifest.loc[
    inner_cv_manifest["outer_fold"] == RF_OUTER_FOLD
].copy()

assert len(rf_manifest) == 1689

# These 421 women belong to Outer Fold 1 validation.
# They must never enter inner training or validation.
outer_validation_indices = set(
    cv_manifest.index[
        cv_manifest["outer_fold"] == RF_OUTER_FOLD
    ]
)

assert len(outer_validation_indices) == 421

rf_score_rows = []
rf_cache_rows = []

print("========== STEP 1.37 RESULTS ==========")

# ------------------------------------------------------------
# 2. Process each of the three inner folds
# ------------------------------------------------------------

for inner_fold in range(3):

    fold_number = inner_fold + 1

    print("\n" + "=" * 60, flush=True)
    print(
        f"RANDOM FOREST: OUTER 1 - INNER {fold_number}",
        flush=True
    )
    print("=" * 60, flush=True)

    train_rows = rf_manifest.loc[
        rf_manifest["inner_fold"] != inner_fold
    ].copy()

    validation_rows = rf_manifest.loc[
        rf_manifest["inner_fold"] == inner_fold
    ].copy()

    train_idx = train_rows[
        "original_row_index"
    ].to_numpy()

    val_idx = validation_rows[
        "original_row_index"
    ].to_numpy()

    # --------------------------------------------------------
    # 3. Verify observation and PSU separation
    # --------------------------------------------------------

    assert len(set(train_idx) & set(val_idx)) == 0

    assert len(
        set(train_idx) & outer_validation_indices
    ) == 0

    assert len(
        set(val_idx) & outer_validation_indices
    ) == 0

    train_psus = set(zip(
        train_rows["stratum16"],
        train_rows["psu"]
    ))

    val_psus = set(zip(
        validation_rows["stratum16"],
        validation_rows["psu"]
    ))

    outer_val_rows = cv_manifest.loc[
        cv_manifest["outer_fold"] == RF_OUTER_FOLD
    ]

    outer_val_psus = set(zip(
        outer_val_rows["stratum16"],
        outer_val_rows["psu"]
    ))

    assert not (train_psus & val_psus)
    assert not (train_psus & outer_val_psus)
    assert not (val_psus & outer_val_psus)

    assert validation_rows["stratum16"].nunique() == 16

    # --------------------------------------------------------
    # 4. Load previously fitted inner-fold matrices
    # --------------------------------------------------------

    cache_path = os.path.join(
        cache_folder_135,
        f"Outer1_Inner{fold_number}_Boruta_Matrices.npz"
    )

    assert os.path.exists(cache_path), (
        f"Missing cache: {cache_path}"
    )

    with np.load(
        cache_path,
        allow_pickle=False
    ) as cached:

        assert cached["version"].item() == (
            "HIES2022_Boruta_InnerCache_v1"
        )

        assert np.array_equal(
            cached["train_idx"],
            train_idx
        )

        assert np.array_equal(
            cached["val_idx"],
            val_idx
        )

        X_tr = cached["X_train"].copy()

        X_val = cached["X_validation"].copy()

        selected_groups = (
            cached["selected_groups"].tolist()
        )

        feature_names = (
            cached["feature_names"].tolist()
        )

    # --------------------------------------------------------
    # 5. Verify Boruta selection matches Step 1.31
    # --------------------------------------------------------

    previous_decisions = boruta_stability_details.loc[
        (
            boruta_stability_details["Outer_Fold"] == 1
        )
        &
        (
            boruta_stability_details["Inner_Fold"]
            == fold_number
        )
    ]

    expected_groups = set(
        previous_decisions.loc[
            previous_decisions["Selected"],
            "Original_Predictor"
        ]
    )

    assert set(selected_groups) == expected_groups

    expected_columns = int(
        boruta_stability_folds.loc[
            boruta_stability_folds["Inner_Fold"]
            == fold_number,
            "Retained_Model_Columns"
        ].iloc[0]
    )

    assert X_tr.shape[1] == expected_columns
    assert X_val.shape[1] == expected_columns

    assert len(feature_names) == expected_columns

    assert np.isfinite(X_tr).all()
    assert np.isfinite(X_val).all()

    # --------------------------------------------------------
    # 6. Retrieve labels and survey sampling weights
    # --------------------------------------------------------

    y_tr = women_analysis.loc[
        train_idx,
        "low_work_intensity"
    ].to_numpy(dtype=int)

    y_val = women_analysis.loc[
        val_idx,
        "low_work_intensity"
    ].to_numpy(dtype=int)

    w_tr = women_analysis.loc[
        train_idx,
        "population_weight"
    ].to_numpy(dtype=float)

    w_val = women_analysis.loc[
        val_idx,
        "population_weight"
    ].to_numpy(dtype=float)

    assert len(X_tr) == len(y_tr) == len(w_tr)
    assert len(X_val) == len(y_val) == len(w_val)

    assert np.isfinite(w_tr).all()
    assert np.isfinite(w_val).all()

    assert (w_tr > 0).all()
    assert (w_val > 0).all()

    assert set(np.unique(y_tr)) == {0, 1}
    assert set(np.unique(y_val)) == {0, 1}

    # Normalize weights for fitting only.
    # Evaluation uses the original survey weights.
    fitting_weights = w_tr / w_tr.mean()

    rf_cache_rows.append({
        "Outer_Fold": 1,
        "Inner_Fold": fold_number,
        "Train_N": len(y_tr),
        "Validation_N": len(y_val),
        "Retained_Columns": X_tr.shape[1],
        "PSU_Overlap": 0
    })

    print("Training N:", len(y_tr))
    print("Validation N:", len(y_val))
    print("Selected columns:", X_tr.shape[1])
    print("Selected predictors:", selected_groups)

    # --------------------------------------------------------
    # 7. Fit and evaluate 12 Random Forest candidates
    # --------------------------------------------------------

    for candidate_id, params in enumerate(
        rf_candidates,
        start=1
    ):

        rf_model = clone(
            tuned_model_templates["Random Forest"]
        )

        rf_model.set_params(**params)

        rf_model.fit(
            X_tr,
            y_tr,
            sample_weight=fitting_weights
        )

        assert np.array_equal(
            rf_model.classes_,
            [0, 1]
        )

        validation_probabilities = (
            rf_model.predict_proba(X_val)[:, 1]
        )

        assert validation_probabilities.shape == (
            len(y_val),
        )

        assert np.isfinite(
            validation_probabilities
        ).all()

        assert (
            (validation_probabilities >= 0)
            & (validation_probabilities <= 1)
        ).all()

        weighted_ap = average_precision_score(
            y_val,
            validation_probabilities,
            sample_weight=w_val
        )

        weighted_auc = roc_auc_score(
            y_val,
            validation_probabilities,
            sample_weight=w_val
        )

        assert np.isfinite(weighted_ap)
        assert np.isfinite(weighted_auc)

        assert 0 <= weighted_ap <= 1
        assert 0 <= weighted_auc <= 1

        rf_score_rows.append({
            "Outer_Fold": 1,
            "Inner_Fold": fold_number,
            "Model": "Random Forest",
            "Candidate_ID": candidate_id,
            "Parameters_JSON": json.dumps(
                params,
                sort_keys=True
            ),
            "Weighted_AP": weighted_ap,
            "Weighted_ROC_AUC": weighted_auc
        })

    print(
        "Twelve RF candidates evaluated successfully.",
        flush=True
    )

# ------------------------------------------------------------
# 8. Aggregate tuning results across inner folds
# ------------------------------------------------------------

rf_inner_scores_137 = pd.DataFrame(
    rf_score_rows
)

assert len(rf_inner_scores_137) == 36

rf_tuning_summary_137 = (
    rf_inner_scores_137
    .groupby(
        ["Candidate_ID", "Parameters_JSON"],
        as_index=False
    )
    .agg(
        Inner_Folds=("Inner_Fold", "nunique"),
        Mean_Weighted_AP=("Weighted_AP", "mean"),
        SD_Weighted_AP=("Weighted_AP", "std"),
        Mean_Weighted_ROC_AUC=(
            "Weighted_ROC_AUC",
            "mean"
        )
    )
)

assert len(rf_tuning_summary_137) == 12

assert (
    rf_tuning_summary_137["Inner_Folds"] == 3
).all()

# ------------------------------------------------------------
# 9. Rank candidates using the predefined criterion
# ------------------------------------------------------------

# Primary: highest mean weighted AP
# Secondary: highest mean weighted ROC-AUC
# Final tie-break: smallest Candidate_ID

rf_tuning_summary_137 = (
    rf_tuning_summary_137
    .sort_values(
        [
            "Mean_Weighted_AP",
            "Mean_Weighted_ROC_AUC",
            "Candidate_ID"
        ],
        ascending=[False, False, True]
    )
    .reset_index(drop=True)
)

rf_tuning_summary_137.insert(
    0,
    "Rank",
    np.arange(1, 13)
)

best_rf_params_outer1 = json.loads(
    rf_tuning_summary_137.iloc[0][
        "Parameters_JSON"
    ]
)

# ------------------------------------------------------------
# 10. Print diagnostics and candidate ranking
# ------------------------------------------------------------

print("\n========== STEP 1.37 TUNING SUMMARY ==========")

rf_cache_audit_137 = pd.DataFrame(
    rf_cache_rows
)

print("\nINNER-FOLD CACHE AUDIT:")

print(
    rf_cache_audit_137.to_string(index=False)
)

print("\nRANDOM FOREST HYPERPARAMETER RANKING:")

print(
    rf_tuning_summary_137
    .round(5)
    .to_string(index=False)
)

print("\nBEST RANDOM FOREST PARAMETERS:")
print(best_rf_params_outer1)

print(
    "Best mean weighted AP:",
    round(
        rf_tuning_summary_137.iloc[0][
            "Mean_Weighted_AP"
        ],
        5
    )
)

print(
    "Corresponding mean weighted ROC-AUC:",
    round(
        rf_tuning_summary_137.iloc[0][
            "Mean_Weighted_ROC_AUC"
        ],
        5
    )
)

# ------------------------------------------------------------
# 11. Save reproducible tuning outputs
# ------------------------------------------------------------

rf_inner_scores_137.to_csv(
    os.path.join(
        output_dir,
        "Step1_37_RF_Outer1_Inner_Scores.csv"
    ),
    index=False
)

rf_tuning_summary_137.to_csv(
    os.path.join(
        output_dir,
        "Step1_37_RF_Outer1_Tuning_Ranking.csv"
    ),
    index=False
)

rf_cache_audit_137.to_csv(
    os.path.join(
        output_dir,
        "Step1_37_RF_Cache_Audit.csv"
    ),
    index=False
)

# ------------------------------------------------------------
# 12. Final verification
# ------------------------------------------------------------

assert len(rf_inner_scores_137) == 36

assert (
    rf_inner_scores_137
    .groupby("Candidate_ID")["Inner_Fold"]
    .nunique()
    == 3
).all()

assert rf_tuning_summary_137[
    "Mean_Weighted_AP"
].between(0, 1).all()

assert rf_tuning_summary_137[
    "Mean_Weighted_ROC_AUC"
].between(0, 1).all()

assert (
    rf_cache_audit_137["PSU_Overlap"] == 0
).all()

assert (
    rf_cache_audit_137["Retained_Columns"]
    .tolist() == [22, 18, 13]
)

assert len(women_analysis) == 2110

print("\nOuter-validation observations used: 0")
print("Total RF inner model fits: 36")
print("STEP 1.37 VERIFICATION SUCCESSFUL.")



# %% STEP 1.38: EXTRA TREES INNER-CV TUNING - OUTER 1

# ============================================================
# HIES 2022 - Extra Trees nested-CV hyperparameter tuning
#
# Outer Fold: 1
# Inner Folds: 3
# Hyperparameter candidates: 8
# Planned fits: 24
#
# PRIMARY metric: Survey-weighted Average Precision
# SECONDARY metric: Survey-weighted ROC-AUC
#
# Reuse previously verified training-only:
# - Preprocessing
# - Wealth PCA
# - Boruta group selection
#
# No outer-validation data used.
# ============================================================

import os
import json
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score
)

# ------------------------------------------------------------
# 1. Define the experiment
# ------------------------------------------------------------

ET_OUTER_FOLD = 0

et_candidates = list(
    ParameterGrid(
        tuning_grids["Extra Trees"]
    )
)

assert len(et_candidates) == 8
assert "Extra Trees" in tuned_model_templates
assert os.path.isdir(cache_folder_135)

et_manifest = inner_cv_manifest.loc[
    inner_cv_manifest["outer_fold"] == ET_OUTER_FOLD
].copy()

assert len(et_manifest) == 1689

outer_val_rows_et = cv_manifest.loc[
    cv_manifest["outer_fold"] == ET_OUTER_FOLD
]

outer_val_indices_et = set(
    outer_val_rows_et.index
)

outer_val_psus_et = set(
    zip(
        outer_val_rows_et["stratum16"],
        outer_val_rows_et["psu"]
    )
)

assert len(outer_val_indices_et) == 421

et_scores = []
et_cache_audit = []

print("========== STEP 1.38 RESULTS ==========")

# ------------------------------------------------------------
# 2. Loop through the three inner folds
# ------------------------------------------------------------

for inner_fold in range(3):

    fold_number = inner_fold + 1

    print("\n" + "=" * 60, flush=True)
    print(
        f"EXTRA TREES: OUTER 1 - INNER {fold_number}",
        flush=True
    )
    print("=" * 60, flush=True)

    train_rows = et_manifest.loc[
        et_manifest["inner_fold"] != inner_fold
    ]

    val_rows = et_manifest.loc[
        et_manifest["inner_fold"] == inner_fold
    ]

    train_idx = train_rows[
        "original_row_index"
    ].to_numpy()

    val_idx = val_rows[
        "original_row_index"
    ].to_numpy()

    # --------------------------------------------------------
    # 3. Verify observation and PSU isolation
    # --------------------------------------------------------

    assert not (set(train_idx) & set(val_idx))
    assert not (set(train_idx) & outer_val_indices_et)
    assert not (set(val_idx) & outer_val_indices_et)

    train_psus = set(
        zip(train_rows["stratum16"], train_rows["psu"])
    )

    val_psus = set(
        zip(val_rows["stratum16"], val_rows["psu"])
    )

    assert not (train_psus & val_psus)
    assert not (train_psus & outer_val_psus_et)
    assert not (val_psus & outer_val_psus_et)

    assert val_rows["stratum16"].nunique() == 16

    # --------------------------------------------------------
    # 4. Load saved preprocessing + Boruta matrices
    # --------------------------------------------------------

    cache_path = os.path.join(
        cache_folder_135,
        f"Outer1_Inner{fold_number}_Boruta_Matrices.npz"
    )

    assert os.path.exists(cache_path)

    with np.load(
        cache_path,
        allow_pickle=False
    ) as cached:

        assert cached["version"].item() == (
            "HIES2022_Boruta_InnerCache_v1"
        )

        assert np.array_equal(
            cached["train_idx"], train_idx
        )

        assert np.array_equal(
            cached["val_idx"], val_idx
        )

        X_tr = cached["X_train"].copy()
        X_val = cached["X_validation"].copy()

        selected_groups = (
            cached["selected_groups"].tolist()
        )

        feature_names = (
            cached["feature_names"].tolist()
        )

    # --------------------------------------------------------
    # 5. Check cache against earlier Boruta results
    # --------------------------------------------------------

    previous_decisions = (
        boruta_stability_details.loc[
            (
                boruta_stability_details["Outer_Fold"]
                == 1
            )
            &
            (
                boruta_stability_details["Inner_Fold"]
                == fold_number
            )
        ]
    )

    expected_groups = set(
        previous_decisions.loc[
            previous_decisions["Selected"],
            "Original_Predictor"
        ]
    )

    assert set(selected_groups) == expected_groups

    expected_columns = int(
        boruta_stability_folds.loc[
            boruta_stability_folds["Inner_Fold"]
            == fold_number,
            "Retained_Model_Columns"
        ].iloc[0]
    )

    assert X_tr.shape[1] == expected_columns
    assert X_val.shape[1] == expected_columns
    assert len(feature_names) == expected_columns

    assert np.isfinite(X_tr).all()
    assert np.isfinite(X_val).all()

    # --------------------------------------------------------
    # 6. Retrieve outcome and survey weights
    # --------------------------------------------------------

    y_tr = women_analysis.loc[
        train_idx, "low_work_intensity"
    ].to_numpy(dtype=int)

    y_val = women_analysis.loc[
        val_idx, "low_work_intensity"
    ].to_numpy(dtype=int)

    w_tr = women_analysis.loc[
        train_idx, "population_weight"
    ].to_numpy(dtype=float)

    w_val = women_analysis.loc[
        val_idx, "population_weight"
    ].to_numpy(dtype=float)

    assert len(X_tr) == len(y_tr) == len(w_tr)
    assert len(X_val) == len(y_val) == len(w_val)

    assert np.isfinite(w_tr).all()
    assert np.isfinite(w_val).all()
    assert (w_tr > 0).all()
    assert (w_val > 0).all()

    assert set(np.unique(y_tr)) == {0, 1}
    assert set(np.unique(y_val)) == {0, 1}

    # Training weights normalized to mean 1.
    # Original survey weights retained for evaluation.
    fitting_weights = w_tr / w_tr.mean()

    et_cache_audit.append({
        "Outer_Fold": 1,
        "Inner_Fold": fold_number,
        "Train_N": len(y_tr),
        "Validation_N": len(y_val),
        "Retained_Columns": X_tr.shape[1],
        "PSU_Overlap": 0
    })

    print("Training N:", len(y_tr))
    print("Validation N:", len(y_val))
    print("Selected columns:", X_tr.shape[1])
    print("Selected predictors:", selected_groups)

    # --------------------------------------------------------
    # 7. Evaluate all eight Extra Trees candidates
    # --------------------------------------------------------

    for candidate_id, params in enumerate(
        et_candidates,
        start=1
    ):

        et_model = clone(
            tuned_model_templates["Extra Trees"]
        )

        et_model.set_params(**params)

        et_model.fit(
            X_tr,
            y_tr,
            sample_weight=fitting_weights
        )

        assert np.array_equal(
            et_model.classes_, [0, 1]
        )

        # Validation predictions only
        probabilities = et_model.predict_proba(
            X_val
        )[:, 1]

        assert probabilities.shape == (len(y_val),)
        assert np.isfinite(probabilities).all()

        assert (
            (probabilities >= 0)
            & (probabilities <= 1)
        ).all()

        weighted_ap = average_precision_score(
            y_val,
            probabilities,
            sample_weight=w_val
        )

        weighted_auc = roc_auc_score(
            y_val,
            probabilities,
            sample_weight=w_val
        )

        assert np.isfinite(weighted_ap)
        assert np.isfinite(weighted_auc)
        assert 0 <= weighted_ap <= 1
        assert 0 <= weighted_auc <= 1

        et_scores.append({
            "Outer_Fold": 1,
            "Inner_Fold": fold_number,
            "Model": "Extra Trees",
            "Candidate_ID": candidate_id,
            "Parameters_JSON": json.dumps(
                params,
                sort_keys=True
            ),
            "Weighted_AP": weighted_ap,
            "Weighted_ROC_AUC": weighted_auc
        })

    print(
        "Eight Extra Trees candidates evaluated.",
        flush=True
    )

# ------------------------------------------------------------
# 8. Combine all 24 inner-fold results
# ------------------------------------------------------------

et_inner_scores_138 = pd.DataFrame(
    et_scores
)

assert len(et_inner_scores_138) == 24

assert (
    et_inner_scores_138
    .groupby("Candidate_ID")["Inner_Fold"]
    .nunique()
    == 3
).all()

# ------------------------------------------------------------
# 9. Summarize performance across folds
# ------------------------------------------------------------

et_tuning_summary_138 = (
    et_inner_scores_138
    .groupby(
        ["Candidate_ID", "Parameters_JSON"],
        as_index=False
    )
    .agg(
        Inner_Folds=("Inner_Fold", "nunique"),
        Mean_Weighted_AP=("Weighted_AP", "mean"),
        SD_Weighted_AP=("Weighted_AP", "std"),
        Mean_Weighted_ROC_AUC=(
            "Weighted_ROC_AUC", "mean"
        )
    )
)

assert len(et_tuning_summary_138) == 8

assert (
    et_tuning_summary_138["Inner_Folds"] == 3
).all()

# ------------------------------------------------------------
# 10. Rank hyperparameter candidates
# ------------------------------------------------------------

# Primary: weighted Average Precision
# Secondary: weighted ROC-AUC
# Final tie-break: smaller candidate ID

et_tuning_summary_138 = (
    et_tuning_summary_138
    .sort_values(
        [
            "Mean_Weighted_AP",
            "Mean_Weighted_ROC_AUC",
            "Candidate_ID"
        ],
        ascending=[False, False, True]
    )
    .reset_index(drop=True)
)

et_tuning_summary_138.insert(
    0,
    "Rank",
    np.arange(1, 9)
)

best_et_params_outer1 = json.loads(
    et_tuning_summary_138.iloc[0][
        "Parameters_JSON"
    ]
)

# ------------------------------------------------------------
# 11. Print results
# ------------------------------------------------------------

print("\n========== STEP 1.38 TUNING SUMMARY ==========")

et_cache_audit_138 = pd.DataFrame(
    et_cache_audit
)

print("\nINNER-FOLD CACHE AUDIT:")

print(
    et_cache_audit_138.to_string(index=False)
)

print("\nEXTRA TREES HYPERPARAMETER RANKING:")

print(
    et_tuning_summary_138.round(5)
    .to_string(index=False)
)

print("\nBEST EXTRA TREES PARAMETERS:")
print(best_et_params_outer1)

print(
    "Best mean weighted AP:",
    round(
        et_tuning_summary_138.iloc[0][
            "Mean_Weighted_AP"
        ],
        5
    )
)

print(
    "Corresponding mean weighted ROC-AUC:",
    round(
        et_tuning_summary_138.iloc[0][
            "Mean_Weighted_ROC_AUC"
        ],
        5
    )
)

# ------------------------------------------------------------
# 12. Save reproducible outputs
# ------------------------------------------------------------

et_inner_scores_138.to_csv(
    os.path.join(
        output_dir,
        "Step1_38_ET_Outer1_Inner_Scores.csv"
    ),
    index=False
)

et_tuning_summary_138.to_csv(
    os.path.join(
        output_dir,
        "Step1_38_ET_Outer1_Tuning_Ranking.csv"
    ),
    index=False
)

et_cache_audit_138.to_csv(
    os.path.join(
        output_dir,
        "Step1_38_ET_Cache_Audit.csv"
    ),
    index=False
)

# ------------------------------------------------------------
# 13. Final verification
# ------------------------------------------------------------

assert len(et_inner_scores_138) == 24

assert et_tuning_summary_138[
    "Mean_Weighted_AP"
].between(0, 1).all()

assert et_tuning_summary_138[
    "Mean_Weighted_ROC_AUC"
].between(0, 1).all()

assert (
    et_cache_audit_138["PSU_Overlap"] == 0
).all()

assert (
    et_cache_audit_138["Retained_Columns"]
    .tolist() == [22, 18, 13]
)

assert len(women_analysis) == 2110

print("\nOuter-validation observations used: 0")
print("Total Extra Trees inner model fits: 24")
print("STEP 1.38 VERIFICATION SUCCESSFUL.")






# %% STEP 1.39: GRADIENT BOOSTING INNER-CV TUNING - OUTER 1

# ============================================================
# HIES 2022 - Gradient Boosting hyperparameter tuning
#
# Outer Fold: 1
# Inner Folds: 3
# Hyperparameter combinations: 8
# Total model fits: 24
#
# Primary metric: Survey-weighted Average Precision
# Secondary metric: Survey-weighted ROC-AUC
#
# Reuses training-only preprocessing, PCA and Boruta caches.
# Outer-validation observations are completely excluded.
# ============================================================

import os
import json
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score
)

# ------------------------------------------------------------
# 1. Configure Gradient Boosting tuning
# ------------------------------------------------------------

GB_OUTER_FOLD = 0

gb_candidates = list(
    ParameterGrid(
        tuning_grids["Gradient Boosting"]
    )
)

assert len(gb_candidates) == 8
assert "Gradient Boosting" in tuned_model_templates
assert os.path.isdir(cache_folder_135)

gb_manifest = inner_cv_manifest.loc[
    inner_cv_manifest["outer_fold"] == GB_OUTER_FOLD
].copy()

assert len(gb_manifest) == 1689

outer_val_rows_gb = cv_manifest.loc[
    cv_manifest["outer_fold"] == GB_OUTER_FOLD
]

outer_val_indices_gb = set(
    outer_val_rows_gb.index
)

outer_val_psus_gb = set(
    zip(
        outer_val_rows_gb["stratum16"],
        outer_val_rows_gb["psu"]
    )
)

assert len(outer_val_indices_gb) == 421

gb_score_rows = []
gb_cache_rows = []

print("========== STEP 1.39 RESULTS ==========")

# ------------------------------------------------------------
# 2. Process the three inner folds
# ------------------------------------------------------------

for inner_fold in range(3):

    fold_number = inner_fold + 1

    print("\n" + "=" * 60, flush=True)
    print(
        f"GRADIENT BOOSTING: OUTER 1 - INNER {fold_number}",
        flush=True
    )
    print("=" * 60, flush=True)

    train_rows = gb_manifest.loc[
        gb_manifest["inner_fold"] != inner_fold
    ]

    val_rows = gb_manifest.loc[
        gb_manifest["inner_fold"] == inner_fold
    ]

    train_idx = train_rows[
        "original_row_index"
    ].to_numpy()

    val_idx = val_rows[
        "original_row_index"
    ].to_numpy()

    # --------------------------------------------------------
    # 3. Check observation and PSU separation
    # --------------------------------------------------------

    assert not (set(train_idx) & set(val_idx))
    assert not (set(train_idx) & outer_val_indices_gb)
    assert not (set(val_idx) & outer_val_indices_gb)

    train_psus = set(
        zip(train_rows["stratum16"], train_rows["psu"])
    )

    val_psus = set(
        zip(val_rows["stratum16"], val_rows["psu"])
    )

    assert not (train_psus & val_psus)
    assert not (train_psus & outer_val_psus_gb)
    assert not (val_psus & outer_val_psus_gb)

    assert val_rows["stratum16"].nunique() == 16

    # --------------------------------------------------------
    # 4. Load previously fitted preprocessing/Boruta data
    # --------------------------------------------------------

    cache_path = os.path.join(
        cache_folder_135,
        f"Outer1_Inner{fold_number}_Boruta_Matrices.npz"
    )

    assert os.path.exists(cache_path), (
        f"Cache not found: {cache_path}"
    )

    with np.load(
        cache_path,
        allow_pickle=False
    ) as cached:

        assert cached["version"].item() == (
            "HIES2022_Boruta_InnerCache_v1"
        )

        assert np.array_equal(
            cached["train_idx"],
            train_idx
        )

        assert np.array_equal(
            cached["val_idx"],
            val_idx
        )

        X_tr = cached["X_train"].copy()
        X_val = cached["X_validation"].copy()

        selected_groups = (
            cached["selected_groups"].tolist()
        )

        feature_names = (
            cached["feature_names"].tolist()
        )

    # --------------------------------------------------------
    # 5. Verify selected features from Step 1.31
    # --------------------------------------------------------

    previous_decisions = (
        boruta_stability_details.loc[
            (
                boruta_stability_details["Outer_Fold"] == 1
            )
            &
            (
                boruta_stability_details["Inner_Fold"]
                == fold_number
            )
        ]
    )

    expected_groups = set(
        previous_decisions.loc[
            previous_decisions["Selected"],
            "Original_Predictor"
        ]
    )

    assert set(selected_groups) == expected_groups

    expected_columns = int(
        boruta_stability_folds.loc[
            boruta_stability_folds["Inner_Fold"]
            == fold_number,
            "Retained_Model_Columns"
        ].iloc[0]
    )

    assert X_tr.shape[1] == expected_columns
    assert X_val.shape[1] == expected_columns
    assert len(feature_names) == expected_columns

    assert np.isfinite(X_tr).all()
    assert np.isfinite(X_val).all()

    # --------------------------------------------------------
    # 6. Retrieve training and validation outcomes/weights
    # --------------------------------------------------------

    y_tr = women_analysis.loc[
        train_idx,
        "low_work_intensity"
    ].to_numpy(dtype=int)

    y_val = women_analysis.loc[
        val_idx,
        "low_work_intensity"
    ].to_numpy(dtype=int)

    w_tr = women_analysis.loc[
        train_idx,
        "population_weight"
    ].to_numpy(dtype=float)

    w_val = women_analysis.loc[
        val_idx,
        "population_weight"
    ].to_numpy(dtype=float)

    assert len(X_tr) == len(y_tr) == len(w_tr)
    assert len(X_val) == len(y_val) == len(w_val)

    assert np.isfinite(w_tr).all()
    assert np.isfinite(w_val).all()
    assert (w_tr > 0).all()
    assert (w_val > 0).all()

    assert set(np.unique(y_tr)) == {0, 1}
    assert set(np.unique(y_val)) == {0, 1}

    # Normalize weights only for fitting.
    # Evaluate with the original survey sampling weights.
    fitting_weights = w_tr / w_tr.mean()

    gb_cache_rows.append({
        "Outer_Fold": 1,
        "Inner_Fold": fold_number,
        "Train_N": len(y_tr),
        "Validation_N": len(y_val),
        "Retained_Columns": X_tr.shape[1],
        "PSU_Overlap": 0
    })

    print("Training N:", len(y_tr))
    print("Validation N:", len(y_val))
    print("Retained columns:", X_tr.shape[1])
    print("Selected predictors:", selected_groups)

    # --------------------------------------------------------
    # 7. Evaluate all eight Gradient Boosting candidates
    # --------------------------------------------------------

    for candidate_id, params in enumerate(
        gb_candidates,
        start=1
    ):

        gb_model = clone(
            tuned_model_templates["Gradient Boosting"]
        )

        gb_model.set_params(**params)

        gb_model.fit(
            X_tr,
            y_tr,
            sample_weight=fitting_weights
        )

        assert np.array_equal(
            gb_model.classes_, [0, 1]
        )

        probabilities = gb_model.predict_proba(
            X_val
        )[:, 1]

        assert probabilities.shape == (len(y_val),)
        assert np.isfinite(probabilities).all()

        assert (
            (probabilities >= 0)
            & (probabilities <= 1)
        ).all()

        weighted_ap = average_precision_score(
            y_val,
            probabilities,
            sample_weight=w_val
        )

        weighted_auc = roc_auc_score(
            y_val,
            probabilities,
            sample_weight=w_val
        )

        assert np.isfinite(weighted_ap)
        assert np.isfinite(weighted_auc)

        assert 0 <= weighted_ap <= 1
        assert 0 <= weighted_auc <= 1

        gb_score_rows.append({
            "Outer_Fold": 1,
            "Inner_Fold": fold_number,
            "Model": "Gradient Boosting",
            "Candidate_ID": candidate_id,
            "Parameters_JSON": json.dumps(
                params,
                sort_keys=True
            ),
            "Weighted_AP": weighted_ap,
            "Weighted_ROC_AUC": weighted_auc
        })

    print(
        "Eight Gradient Boosting candidates evaluated.",
        flush=True
    )

# ------------------------------------------------------------
# 8. Combine all 24 inner-fold results
# ------------------------------------------------------------

gb_inner_scores_139 = pd.DataFrame(
    gb_score_rows
)

assert len(gb_inner_scores_139) == 24

assert (
    gb_inner_scores_139
    .groupby("Candidate_ID")["Inner_Fold"]
    .nunique()
    == 3
).all()

# ------------------------------------------------------------
# 9. Summarize performance across inner folds
# ------------------------------------------------------------

gb_tuning_summary_139 = (
    gb_inner_scores_139
    .groupby(
        ["Candidate_ID", "Parameters_JSON"],
        as_index=False
    )
    .agg(
        Inner_Folds=("Inner_Fold", "nunique"),
        Mean_Weighted_AP=("Weighted_AP", "mean"),
        SD_Weighted_AP=("Weighted_AP", "std"),
        Mean_Weighted_ROC_AUC=(
            "Weighted_ROC_AUC", "mean"
        )
    )
)

assert len(gb_tuning_summary_139) == 8

assert (
    gb_tuning_summary_139["Inner_Folds"] == 3
).all()

# ------------------------------------------------------------
# 10. Rank hyperparameters using predefined criteria
# ------------------------------------------------------------

# Primary: Mean weighted AP (higher is better)
# Secondary: Mean weighted ROC-AUC
# Final tie-break: Candidate ID

gb_tuning_summary_139 = (
    gb_tuning_summary_139
    .sort_values(
        [
            "Mean_Weighted_AP",
            "Mean_Weighted_ROC_AUC",
            "Candidate_ID"
        ],
        ascending=[False, False, True]
    )
    .reset_index(drop=True)
)

gb_tuning_summary_139.insert(
    0,
    "Rank",
    np.arange(1, 9)
)

best_gb_params_outer1 = json.loads(
    gb_tuning_summary_139.iloc[0][
        "Parameters_JSON"
    ]
)

# ------------------------------------------------------------
# 11. Print complete results
# ------------------------------------------------------------

print("\n========== STEP 1.39 TUNING SUMMARY ==========")

gb_cache_audit_139 = pd.DataFrame(
    gb_cache_rows
)

print("\nINNER-FOLD CACHE AUDIT:")

print(
    gb_cache_audit_139.to_string(index=False)
)

print("\nGRADIENT BOOSTING HYPERPARAMETER RANKING:")

print(
    gb_tuning_summary_139
    .round(5)
    .to_string(index=False)
)

print("\nBEST GRADIENT BOOSTING PARAMETERS:")
print(best_gb_params_outer1)

print(
    "Best mean weighted AP:",
    round(
        gb_tuning_summary_139.iloc[0][
            "Mean_Weighted_AP"
        ],
        5
    )
)

print(
    "Corresponding mean weighted ROC-AUC:",
    round(
        gb_tuning_summary_139.iloc[0][
            "Mean_Weighted_ROC_AUC"
        ],
        5
    )
)

# ------------------------------------------------------------
# 12. Save reproducible results
# ------------------------------------------------------------

gb_inner_scores_139.to_csv(
    os.path.join(
        output_dir,
        "Step1_39_GB_Outer1_Inner_Scores.csv"
    ),
    index=False
)

gb_tuning_summary_139.to_csv(
    os.path.join(
        output_dir,
        "Step1_39_GB_Outer1_Tuning_Ranking.csv"
    ),
    index=False
)

gb_cache_audit_139.to_csv(
    os.path.join(
        output_dir,
        "Step1_39_GB_Cache_Audit.csv"
    ),
    index=False
)

# ------------------------------------------------------------
# 13. Final checks
# ------------------------------------------------------------

assert len(gb_inner_scores_139) == 24

assert gb_tuning_summary_139[
    "Mean_Weighted_AP"
].between(0, 1).all()

assert gb_tuning_summary_139[
    "Mean_Weighted_ROC_AUC"
].between(0, 1).all()

assert (
    gb_cache_audit_139["PSU_Overlap"] == 0
).all()

assert (
    gb_cache_audit_139["Retained_Columns"]
    .tolist() == [22, 18, 13]
)

assert len(women_analysis) == 2110

print("\nOuter-validation observations used: 0")
print("Total Gradient Boosting inner model fits: 24")
print("STEP 1.39 VERIFICATION SUCCESSFUL.")




# %% STEP 1.40: RUN ALL REMAINING NESTED-CV TUNING

from pathlib import Path

script = Path(
    str(CODE_DIR / "Step1_40_Overnight_Nested_Tuning.py")
)

assert script.is_file(), f"Script not found: {script}"

exec(
    compile(
        script.read_text(encoding="utf-8"),
        str(script),
        "exec"
    ),
    globals()
)






# %% STEP 1.41: RUN NESTED OUTER-FOLD EVALUATION

from pathlib import Path

script = Path(
    str(CODE_DIR / "Step1_41_Nested_Outer_Predictions.py")
)

assert script.is_file(), f"File not found: {script}"

exec(
    compile(
        script.read_text(encoding="utf-8"),
        str(script),
        "exec"
    ),
    globals()
)





# %% STEP 1.42: CALIBRATION AND PSU BOOTSTRAP

from pathlib import Path

script = Path(
    str(CODE_DIR / "Step1_42_Calibration_Uncertainty.py")
)

assert script.is_file(), f"File not found: {script}"

exec(
    compile(
        script.read_text(encoding="utf-8"),
        str(script),
        "exec"
    ),
    globals()
)




# %% STEP 1.43: SHAP + PUBLICATION FIGURE EXPORTS

from pathlib import Path

script = Path(
    str(CODE_DIR / "Step1_43_RF_OuterFold_SHAP_FigureVariants.py")
)

assert script.is_file(), f"File not found: {script}"

exec(
    compile(
        script.read_text(encoding="utf-8"),
        str(script),
        "exec"
    ),
    globals()
)


# %% STEP 1.43A: RECOVERY ERROR DIAGNOSIS

import subprocess
import sys
import os
from pathlib import Path

script = Path(
    str(CODE_DIR / "Step1_43_Recovery_LowMemory.py")
)

output_folder = Path(
    str(OUTPUT_DIR_PATH)
)

fingerprint = "7244d907321a2e433a44"

log_path = output_folder / (
    "Step1_43_Recovery_Error_Log.txt"
)

assert script.is_file(), f"Script missing: {script}"
assert output_folder.is_dir()

print("========== STEP 1.43 ERROR DIAGNOSIS ==========")
print("Python executable:", sys.executable)
print("Recovery script:", script)
print("Diagnostic log:", log_path)

command = [
    sys.executable,
    "-u",
    str(script),
    "--output-dir",
    str(output_folder),
    "--fingerprint",
    fingerprint
]

environment = os.environ.copy()
environment["MPLBACKEND"] = "Agg"
environment["PYTHONUNBUFFERED"] = "1"

# Capture both standard output and the real traceback
# without keeping large outputs in Spyder memory.
with open(
    log_path, "w", encoding="utf-8",
    errors="replace"
) as log_file:

    result = subprocess.run(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=environment,
        check=False
    )

print("\nRECOVERY EXIT CODE:", result.returncode)

# Display the final 60 lines, including the traceback.
with open(
    log_path, "r", encoding="utf-8",
    errors="replace"
) as log_file:
    lines = log_file.readlines()

print("\n========== LAST 60 LOG LINES ==========")

for line in lines[-60:]:
    print(line, end="")

print("\n========== DIAGNOSIS FINISHED ==========")

if result.returncode == 0:
    print("RECOVERY SUCCESSFUL")
else:
    print("RECOVERY FAILED - SEE ERROR ABOVE")

# %% STEP 1.43 RECOVERY - FRESH SPYDER KERNEL

import subprocess
import sys
from pathlib import Path

script = Path(
    str(CODE_DIR / "Step1_43_Recovery_LowMemory.py")
)

output_folder = str(OUTPUT_DIR_PATH)

assert script.is_file(), f"Script not found: {script}"

result = subprocess.run(
    [
        sys.executable,
        "-u",
        str(script),
        "--output-dir",
        output_folder,
        "--fingerprint",
        "7244d907321a2e433a44"
    ],
    check=False
)

print("Recovery exit code:", result.returncode)

assert result.returncode == 0, (
    "Recovery stopped. Check the console output."
)




# %% STEP 1.43B: FIX TIFF CRASH AND RESUME FIGURE EXPORT

from pathlib import Path
import subprocess
import sys
import os

script = Path(
    str(CODE_DIR / "Step1_43_Recovery_LowMemory.py")
)

output_dir = Path(
    str(OUTPUT_DIR_PATH)
)

fingerprint = "7244d907321a2e433a44"

assert script.is_file(), f"Script not found: {script}"
assert output_dir.is_dir()

# -------------------------------------------------
# 1. Replace problematic LZW TIFF compression
# -------------------------------------------------

source = script.read_text(encoding="utf-8")

old = "'compression': 'tiff_lzw'"
new = "'compression': 'raw'"

if old in source:

    backup = script.with_suffix(".py.backup")

    if not backup.exists():
        backup.write_text(source, encoding="utf-8")

    source = source.replace(old, new)

    script.write_text(source, encoding="utf-8")

    print("TIFF compression changed: LZW -> RAW")

elif new in source:

    print("RAW TIFF compression already configured.")

else:

    raise RuntimeError(
        "TIFF export code not found. "
        "Original script was not modified."
    )

# -------------------------------------------------
# 2. Run recovery in a separate process
# -------------------------------------------------

environment = os.environ.copy()

environment["MPLBACKEND"] = "Agg"
environment["OMP_NUM_THREADS"] = "1"
environment["OPENBLAS_NUM_THREADS"] = "1"
environment["MKL_NUM_THREADS"] = "1"
environment["PYTHONUNBUFFERED"] = "1"

log_path = output_dir / "Step1_43_Recovery_Fixed_Log.txt"

command = [
    sys.executable,
    "-u",
    str(script),
    "--output-dir",
    str(output_dir),
    "--fingerprint",
    fingerprint
]

print("\nStarting recovery...")
print("Previous SHAP checkpoints will be reused.")

with log_path.open(
    "w", encoding="utf-8", errors="replace"
) as log_file:

    result = subprocess.run(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=environment,
        check=False
    )

# -------------------------------------------------
# 3. Display results or real error
# -------------------------------------------------

print("\nRECOVERY EXIT CODE:", result.returncode)

with log_path.open(
    "r", encoding="utf-8", errors="replace"
) as log_file:

    lines = log_file.readlines()

print("\n========== RECOVERY LOG ==========")

for line in lines[-45:]:
    print(line, end="")

print("\n========== FINAL STATUS ==========")

if result.returncode == 0:

    print("STEP 1.43 RECOVERY SUCCESSFUL")
    print("TIFF: 600 DPI, uncompressed")
    print("PNG: 300 DPI")
    print("PDF: Vector")
    print("WithTitle and NoTitle: Included")

else:

    print("RECOVERY STILL FAILED")
    print("Please share the last 45 log lines.")





# %% STEP 1.43C: JOURNAL-STYLE FIGURES

import subprocess
import sys
from pathlib import Path

script = Path(
    str(CODE_DIR / "Step1_43C_Journal_Style_Figures.py")
)

assert script.is_file(), f"Script not found: {script}"

log_path = Path(
    str(OUTPUT_DIR_PATH)
) / "Step1_43C_Journal_Figure_Log.txt"

with log_path.open("w", encoding="utf-8") as log:
    result = subprocess.run(
        [
            sys.executable,
            "-u",
            str(script),
            "--output-dir",
            str(OUTPUT_DIR_PATH),
            "--fingerprint",
            "7244d907321a2e433a44"
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        check=False
    )

print("EXIT CODE:", result.returncode)

lines = log_path.read_text(
    encoding="utf-8",
    errors="replace"
).splitlines()

print("\n".join(lines[-30:]))



# %% STEP 1.43D: FIX DISK SPACE + RESUME ALL JOURNAL FIGURES

from pathlib import Path
import shutil
import subprocess
import sys
import os

print("========== STEP 1.43D: STORAGE FIX ==========")

# ----------------------------------------------------------
# 1. Paths
# ----------------------------------------------------------

root = Path(
    str(OUTPUT_DIR_PATH)
)

fingerprint = "7244d907321a2e433a44"

original_script = Path(
    str(CODE_DIR / "Step1_43C_Journal_Style_Figures.py")
)

fixed_script = Path(
    str(CODE_DIR / "Step1_43D_Compressed_Figures.py")
)

figure_dir = root / (
    f"Step1_43C_Journal_Style_Figures_{fingerprint}"
)

assert original_script.is_file()
assert root.is_dir()
assert figure_dir.is_dir()

# Safety: operate only within the Step 1.43C output folder
assert figure_dir.name == (
    f"Step1_43C_Journal_Style_Figures_{fingerprint}"
)

def free_gb():
    return shutil.disk_usage(str(root)).free / (1024 ** 3)

print("Available disk space before:", round(free_gb(), 3), "GB")

# ----------------------------------------------------------
# 2. Identify old uncompressed TIFF exports
# ----------------------------------------------------------

old_tiffs = sorted(
    figure_dir.glob("Figure_*.tiff")
)

partial_tiffs = sorted(
    figure_dir.glob("Figure_*.partial.tiff")
)

# Expected maximum: 10 figures x 2 title variants
assert len(old_tiffs) <= 20

old_size = sum(
    p.stat().st_size
    for p in old_tiffs + partial_tiffs
)

print("\nOld TIFF files:", len(old_tiffs))
print("Incomplete TIFF files:", len(partial_tiffs))
print("TIFF storage:", round(old_size / 1e6, 2), "MB")

# ----------------------------------------------------------
# 3. Prepare compressed script
# ----------------------------------------------------------

source = original_script.read_text(encoding="utf-8")

old_setting = "'compression':'raw'"
new_setting = "'compression':'tiff_adobe_deflate'"

assert old_setting in source, (
    "Expected TIFF export setting not found. "
    "Original script has not been changed."
)

compressed_source = source.replace(
    old_setting,
    new_setting
)

# Correct the verification text too
compressed_source = compressed_source.replace(
    "600-dpi uncompressed TIFF",
    "600-dpi lossless Deflate-compressed TIFF"
)

compile(
    compressed_source,
    str(fixed_script),
    "exec"
)

print("\nCompressed TIFF script verified.")

# ----------------------------------------------------------
# 4. Free space ONLY from Step 1.43C TIFF exports
# ----------------------------------------------------------

# These TIFFs will be regenerated using lossless compression.
# Do NOT delete Step 1.41, 1.42 or 1.43 analytical outputs.

for file in old_tiffs + partial_tiffs:
    file.unlink()
    print("Cleared previous TIFF:", file.name)

print("\nAvailable disk space after cleanup:",
      round(free_gb(), 3), "GB")

# ----------------------------------------------------------
# 5. Save corrected figure script
# ----------------------------------------------------------

fixed_script.write_text(
    compressed_source,
    encoding="utf-8"
)

print("\nNew script:", fixed_script)
print("TIFF compression: Adobe Deflate (lossless)")
print("TIFF resolution: 600 DPI")
print("PNG resolution: 300 DPI")
print("PDF: retained")
print("Title versions: WithTitle + NoTitle")

# ----------------------------------------------------------
# 6. Resume export from existing analytical results
# ----------------------------------------------------------

log_path = root / "Step1_43D_Compressed_Export_Log.txt"

environment = os.environ.copy()

environment["MPLBACKEND"] = "Agg"
environment["OMP_NUM_THREADS"] = "1"
environment["OPENBLAS_NUM_THREADS"] = "1"
environment["MKL_NUM_THREADS"] = "1"
environment["PYTHONUNBUFFERED"] = "1"

print("\nResuming figure export...")
print("Existing PNG and PDF files will be reused.")

with log_path.open("w", encoding="utf-8") as log:

    result = subprocess.run(
        [
            sys.executable,
            "-u",
            str(fixed_script),
            "--output-dir",
            str(root),
            "--fingerprint",
            fingerprint
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=environment,
        check=False
    )

# ----------------------------------------------------------
# 7. Display final results
# ----------------------------------------------------------

print("\nEXPORT EXIT CODE:", result.returncode)

lines = log_path.read_text(
    encoding="utf-8",
    errors="replace"
).splitlines()

print("\n========== LAST 40 LOG LINES ==========")

print("\n".join(lines[-40:]))

print("\n========== FILE VERIFICATION ==========")

for extension in ["tiff", "png", "pdf"]:

    files = list(
        figure_dir.glob(f"Figure_*.{extension}")
    )

    total_mb = sum(
        p.stat().st_size for p in files
    ) / 1e6

    print(
        extension.upper(),
        "| Files:", len(files),
        "| Total MB:", round(total_mb, 2)
    )

print(
    "\nRemaining free space:",
    round(free_gb(), 3),
    "GB"
)

if result.returncode == 0:

    assert len(list(figure_dir.glob("Figure_*.tiff"))) == 20
    assert len(list(figure_dir.glob("Figure_*.png"))) == 20
    assert len(list(figure_dir.glob("Figure_*.pdf"))) == 20

    print("\nSTEP 1.43D VERIFICATION SUCCESSFUL.")

else:

    print("\nSTEP 1.43D INCOMPLETE.")
    print("Share the last 40 log lines.")





# %% STEP 1.43D: FIX DUPLICATE DELETION + RESUME EXPORT

from pathlib import Path
import subprocess
import shutil
import sys
import os

root = Path(str(OUTPUT_DIR_PATH))
fingerprint = "7244d907321a2e433a44"

original_script = Path(
    str(CODE_DIR / "Step1_43C_Journal_Style_Figures.py")
)

fixed_script = Path(
    str(CODE_DIR / "Step1_43D_Compressed_Figures.py")
)

figure_dir = root / (
    f"Step1_43C_Journal_Style_Figures_{fingerprint}"
)

assert original_script.is_file()
assert figure_dir.is_dir()

print("========== STEP 1.43D CORRECTED ==========")

# 1. Prepare lossless-compressed TIFF export
source = original_script.read_text(encoding="utf-8")

old_setting = "'compression':'raw'"
new_setting = "'compression':'tiff_adobe_deflate'"

assert old_setting in source or new_setting in source, (
    "Expected TIFF compression setting not found."
)

source = source.replace(old_setting, new_setting)

compile(source, str(fixed_script), "exec")

# 2. Identify TIFF files WITHOUT duplicates
old_files = sorted(set(
    list(figure_dir.glob("Figure_*.tiff"))
    + list(figure_dir.glob("Figure_*.partial.tiff"))
))

assert len(old_files) <= 40

print("Existing TIFF files to replace:", len(old_files))

recovered_mb = 0

for file in old_files:
    if file.exists():
        recovered_mb += file.stat().st_size / 1e6
        file.unlink(missing_ok=True)

print("Storage cleared:", round(recovered_mb, 2), "MB")

free_gb = shutil.disk_usage(str(root)).free / (1024 ** 3)

print("Available storage:", round(free_gb, 3), "GB")

if free_gb < 0.25:
    raise RuntimeError(
        "Less than 250 MB available on G: drive. "
        "Please free additional disk space before exporting."
    )

# 3. Save corrected export script
fixed_script.write_text(source, encoding="utf-8")

print("Compressed TIFF code saved.")
print("TIFF: 600 DPI, lossless Deflate")
print("PNG: 300 DPI")
print("PDF: Vector")

# 4. Run figure generation
log_path = root / "Step1_43D_Corrected_Log.txt"

env = os.environ.copy()
env["MPLBACKEND"] = "Agg"
env["OMP_NUM_THREADS"] = "1"
env["OPENBLAS_NUM_THREADS"] = "1"
env["MKL_NUM_THREADS"] = "1"
env["PYTHONUNBUFFERED"] = "1"

with log_path.open("w", encoding="utf-8") as log:
    result = subprocess.run(
        [
            sys.executable,
            "-u",
            str(fixed_script),
            "--output-dir",
            str(root),
            "--fingerprint",
            fingerprint
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
        check=False
    )

# 5. Display verification
print("\nEXPORT EXIT CODE:", result.returncode)

lines = log_path.read_text(
    encoding="utf-8",
    errors="replace"
).splitlines()

print("\n========== LAST 35 LOG LINES ==========")
print("\n".join(lines[-35:]))

print("\n========== FIGURE FILE AUDIT ==========")

for ext in ["tiff", "png", "pdf"]:
    files = list(figure_dir.glob(f"Figure_*.{ext}"))

    size_mb = sum(
        f.stat().st_size for f in files
    ) / 1e6

    print(
        f"{ext.upper()}: {len(files)}/20 files | "
        f"{size_mb:.2f} MB"
    )

if result.returncode == 0:
    print("\nSTEP 1.43D VERIFICATION SUCCESSFUL.")
else:
    print("\nSTEP 1.43D INCOMPLETE.")
    print("Share the last 35 log lines.")




# %% STEP 1.43E: SAFE TIFF EXPORT - NO PILLOW TIFF WRITER

from pathlib import Path
import subprocess
import importlib.util
import os
import sys

print("========== STEP 1.43E: SAFE FIGURE EXPORT ==========")

root = Path(str(OUTPUT_DIR_PATH))
fingerprint = "7244d907321a2e433a44"

original_script = Path(
    str(CODE_DIR / "Step1_43C_Journal_Style_Figures.py")
)

safe_script = Path(
    str(CODE_DIR / "Step1_43E_Journal_TiffSafe.py")
)

figure_dir = root / (
    f"Step1_43C_Journal_Style_Figures_{fingerprint}"
)

assert original_script.is_file()
assert root.is_dir()
assert figure_dir.is_dir()

# ---------------------------------------------------------
# 1. Check TIFF writer availability
# ---------------------------------------------------------

if importlib.util.find_spec("tifffile") is None:
    raise ImportError(
        "tifffile is not installed. "
        "Run '%pip install tifffile' in Spyder first."
    )

print("TIFF writer: tifffile")
print("TIFF compression: lossless DEFLATE")
print("Resolution: 600 DPI")

# ---------------------------------------------------------
# 2. Read original journal-style plotting script
# ---------------------------------------------------------

source = original_script.read_text(encoding="utf-8")

start = source.index("\ndef export(")
end = source.index("\ndef plot_one(", start)

# ---------------------------------------------------------
# 3. Replacement export function
# ---------------------------------------------------------

replacement = '''
def export(fig, output_folder, fig_id, version):

    import os
    import gc
    import numpy as np
    import tifffile
    from PIL import Image
    import matplotlib.pyplot as plt

    output_folder.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # A. Save normal publication figures
    # -----------------------------------------------------

    for extension, dpi in [
        ("png", 300),
        ("pdf", 150)
    ]:

        dest = output_folder / (
            f"{fig_id}_{version}.{extension}"
        )

        if dest.is_file() and dest.stat().st_size > 1024:
            print("    Existing:", dest.name, flush=True)
            continue

        tmp = output_folder / (
            f"{fig_id}_{version}.partial.{extension}"
        )

        tmp.unlink(missing_ok=True)

        fig.savefig(
            str(tmp),
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white"
        )

        assert tmp.is_file()
        assert tmp.stat().st_size > 1024

        os.replace(tmp, dest)

        print("    Saved:", dest.name, flush=True)

    # -----------------------------------------------------
    # B. Prepare TIFF from a TRUE 600-DPI rendering
    # -----------------------------------------------------

    tiff_dest = output_folder / (
        f"{fig_id}_{version}.tiff"
    )

    # Reuse only a verified, already-compressed 600-DPI TIFF.
    valid_existing = False

    if tiff_dest.is_file() and tiff_dest.stat().st_size > 1024:

        try:
            with tifffile.TiffFile(str(tiff_dest)) as tf:
                page = tf.pages[0]
                xres = page.tags["XResolution"].value
                dpi_value = xres[0] / xres[1]
                valid_existing = (
                    int(page.compression) == 8
                    and abs(dpi_value - 600) < 0.01
                )
        except Exception:
            valid_existing = False

    if valid_existing:

        print("    Existing verified TIFF:",
              tiff_dest.name, flush=True)

        plt.close(fig)
        gc.collect()
        return

    hi_res_png = output_folder / (
        f"{fig_id}_{version}.600DPI_TEMP.png"
    )

    temp_tiff = output_folder / (
        f"{fig_id}_{version}.partial.tiff"
    )

    hi_res_png.unlink(missing_ok=True)
    temp_tiff.unlink(missing_ok=True)

    # TIFF is generated from a fresh 600-DPI rendering,
    # not an enlarged 300-DPI PNG.

    fig.savefig(
        str(hi_res_png),
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)
    gc.collect()

    # -----------------------------------------------------
    # C. Convert with tifffile instead of Pillow TIFF save
    # -----------------------------------------------------

    try:

        # Generated by this script; dimensions are trusted.
        Image.MAX_IMAGE_PIXELS = None

        with Image.open(hi_res_png) as image:
            rgb = image.convert("RGB")
            array = np.asarray(rgb)

        tifffile.imwrite(
            str(temp_tiff),
            array,
            photometric="rgb",
            compression="zlib",
            compressionargs={"level": 6},
            resolution=(600, 600),
            resolutionunit="INCH",
            metadata=None
        )

        del array
        gc.collect()

        # -------------------------------------------------
        # D. Verify actual TIFF metadata
        # -------------------------------------------------

        with tifffile.TiffFile(str(temp_tiff)) as tf:

            page = tf.pages[0]

            assert int(page.compression) == 8

            assert int(
                page.tags["ResolutionUnit"].value
            ) == 2

            resolution = page.tags["XResolution"].value

            actual_dpi = resolution[0] / resolution[1]

            assert abs(actual_dpi - 600) < 0.01

            assert page.imagewidth > 0
            assert page.imagelength > 0

        os.replace(temp_tiff, tiff_dest)

        print(
            "    Saved:",
            tiff_dest.name,
            "|",
            round(tiff_dest.stat().st_size / 1e6, 2),
            "MB | 600 DPI",
            flush=True
        )

    finally:

        hi_res_png.unlink(missing_ok=True)

'''

# ---------------------------------------------------------
# 4. Build corrected script safely
# ---------------------------------------------------------

corrected_source = (
    source[:start]
    + "\n"
    + replacement
    + source[end:]
)

corrected_source = corrected_source.replace(
    "600-dpi uncompressed TIFF",
    "600-dpi lossless Deflate TIFF"
)

compile(
    corrected_source,
    str(safe_script),
    "exec"
)

safe_script.write_text(
    corrected_source,
    encoding="utf-8"
)

print("\nCorrected script created:", safe_script)

# ---------------------------------------------------------
# 5. Run all journal figures in isolated processes
# ---------------------------------------------------------

log_path = root / "Step1_43E_Safe_TIFF_Export_Log.txt"

env = os.environ.copy()

env["MPLBACKEND"] = "Agg"
env["OMP_NUM_THREADS"] = "1"
env["OPENBLAS_NUM_THREADS"] = "1"
env["MKL_NUM_THREADS"] = "1"
env["PYTHONUNBUFFERED"] = "1"

print("\nExporting journal-style figures...")

with log_path.open(
    "w",
    encoding="utf-8",
    errors="replace"
) as log:

    result = subprocess.run(
        [
            sys.executable,
            "-u",
            str(safe_script),
            "--output-dir",
            str(root),
            "--fingerprint",
            fingerprint
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
        check=False
    )

# ---------------------------------------------------------
# 6. Show result
# ---------------------------------------------------------

print("\nEXPORT EXIT CODE:", result.returncode)

lines = log_path.read_text(
    encoding="utf-8",
    errors="replace"
).splitlines()

print("\n========== LAST 45 LOG LINES ==========")

print("\n".join(lines[-45:]))

# ---------------------------------------------------------
# 7. Audit all exported figures
# ---------------------------------------------------------

print("\n========== FIGURE AUDIT ==========")

for extension in ["png", "tiff", "pdf"]:

    files = list(
        figure_dir.glob(f"Figure_*.{extension}")
    )

    valid_files = [
        f for f in files
        if f.stat().st_size > 1024
    ]

    total_mb = sum(
        f.stat().st_size
        for f in valid_files
    ) / 1e6

    print(
        extension.upper(),
        "| Valid files:", len(valid_files),
        "/20 | Total:", round(total_mb, 2), "MB"
    )

if result.returncode == 0:

    assert all(
        len([
            f for f in figure_dir.glob(
                f"Figure_*.{ext}"
            )
            if f.stat().st_size > 1024
        ]) == 20
        for ext in ["png", "tiff", "pdf"]
    )

    print("\nSTEP 1.43E VERIFICATION SUCCESSFUL.")

else:

    print("\nSTEP 1.43E INCOMPLETE.")
    print("Please share the final log output.")






# %% STEP 1.44: STRUCTURAL MISSINGNESS AND SHAP AUDIT

import numpy as np
import pandas as pd
from pathlib import Path

print("========== STEP 1.44: EMPLOYMENT MISSINGNESS AUDIT ==========")

# ------------------------------------------------------------
# 1. Load verified saved SHAP results
# ------------------------------------------------------------

root = Path(str(OUTPUT_DIR_PATH))
fp = "7244d907321a2e433a44"

shap_dir = root / f"Step1_43_RF_OuterHoldout_SHAP_{fp}"

shap_path = (
    shap_dir / "Step1_43_RF_HeldOut_GroupSHAP_2110.csv"
)

stata_path = Path(str(DATA_FILE))

audit_dir = root / f"Step1_44_Missingness_Audit_{fp}"
audit_dir.mkdir(parents=True, exist_ok=True)

assert shap_path.is_file(), shap_path
assert stata_path.is_file(), stata_path

shap_df = pd.read_csv(shap_path)

assert len(shap_df) == 2110
assert shap_df["original_row_index"].is_unique
assert shap_df["Outer_Fold"].nunique() == 5
assert shap_df["Actual"].sum() == 553

# ------------------------------------------------------------
# 2. Load source variables without fitting any model
# ------------------------------------------------------------

source_columns = [
    "sample_earnings_women_15plus",
    "earn_annual_hours",
    "population_weight",
    "earn_sector",
    "earn_emp_status_nonag",
    "earn_org_type",
    "earn_emp_status_ag"
]

raw = pd.read_stata(
    stata_path,
    columns=source_columns,
    convert_categoricals=False
)

assert len(raw) == 62407

idx = shap_df["original_row_index"].to_numpy(dtype=int)

assert idx.min() >= 0
assert idx.max() < len(raw)

source = raw.iloc[idx].reset_index(drop=True)
data = shap_df.reset_index(drop=True).copy()

assert source["sample_earnings_women_15plus"].eq(1).all()

# Check original outcome and survey weights against saved OOF.
source_y = (
    source["earn_annual_hours"].to_numpy(dtype=float) <= 1440
).astype(int)

assert np.array_equal(
    source_y,
    data["Actual"].to_numpy(dtype=int)
), "Source rows or outcome definition do not match."

assert np.allclose(
    source["population_weight"].to_numpy(dtype=float),
    data["population_weight"].to_numpy(dtype=float),
    rtol=1e-5,
    atol=1e-5
), "Source and SHAP weights do not match."

for field in [
    "earn_sector",
    "earn_emp_status_nonag",
    "earn_org_type",
    "earn_emp_status_ag"
]:
    data[field] = source[field].to_numpy()

w_all = data["population_weight"].to_numpy(dtype=float)

assert np.isfinite(w_all).all()
assert (w_all > 0).all()

# ------------------------------------------------------------
# 3. Define the three employment-related predictors
# ------------------------------------------------------------

predictors = {
    "earn_emp_status_nonag": (
        "Non-agricultural employment status",
        "earn_emp_status_nonag"
    ),
    "earn_org_type_grp": (
        "Organization/workplace type",
        "earn_org_type"
    ),
    "earn_emp_status_ag": (
        "Agricultural employment status",
        "earn_emp_status_ag"
    )
}

def weighted_mean(values, weights):
    return float(
        np.average(
            np.asarray(values, dtype=float),
            weights=np.asarray(weights, dtype=float)
        )
    )

def summarize_subset(part):
    weights = part["population_weight"].to_numpy(dtype=float)
    return {
        "N": int(len(part)),
        "Weighted_Share": float(weights.sum() / w_all.sum()),
        "Observed_Low_Work_Intensity": weighted_mean(
            part["Actual"], weights
        ),
        "Mean_Predicted_Risk": weighted_mean(
            part["Predicted_Probability"], weights
        )
    }

# ------------------------------------------------------------
# 4. Audit missing / not-applicable categories
# ------------------------------------------------------------

missing_rows = []
sector_rows = []

for parent, (label, raw_field) in predictors.items():

    shap_column = f"SHAP__{parent}"
    assert shap_column in data.columns

    # Original HIES missing values are NaN.
    # Also recognize any explicit -99 sentinel if present.
    missing_flag = (
        data[raw_field].isna()
        | data[raw_field].eq(-99)
    )

    data[f"Missing__{parent}"] = missing_flag

    for status, mask in [
        ("Missing / not applicable", missing_flag),
        ("Observed", ~missing_flag)
    ]:

        part = data.loc[mask]
        if len(part) == 0:
            continue

        result = summarize_subset(part)

        result.update({
            "Predictor": label,
            "Source_Status": status,
            "Weighted_Mean_Absolute_SHAP": weighted_mean(
                part[shap_column].abs(),
                part["population_weight"]
            ),
            "Weighted_Mean_Signed_SHAP": weighted_mean(
                part[shap_column],
                part["population_weight"]
            )
        })

        missing_rows.append(result)

    # Sector-level check to identify skip-pattern relationships.
    for sector_code, sector_label in [
        (1, "Agriculture"),
        (2, "Non-agriculture")
    ]:

        for status, is_missing in [
            ("Missing / not applicable", True),
            ("Observed", False)
        ]:

            mask = (
                data["earn_sector"].eq(sector_code)
                & missing_flag.eq(is_missing)
            )

            part = data.loc[mask]

            if len(part) == 0:
                continue

            result = summarize_subset(part)

            result.update({
                "Predictor": label,
                "Employment_Sector": sector_label,
                "Source_Status": status,
                "Weighted_Mean_Absolute_SHAP": weighted_mean(
                    part[shap_column].abs(),
                    part["population_weight"]
                )
            })

            sector_rows.append(result)

missing_summary_144 = pd.DataFrame(missing_rows)
sector_summary_144 = pd.DataFrame(sector_rows)

# ------------------------------------------------------------
# 5. Examine joint missingness of top two predictors
# ------------------------------------------------------------

flag_nonag = data["Missing__earn_emp_status_nonag"]
flag_org = data["Missing__earn_org_type_grp"]

joint_rows = []

for nonag_missing in [False, True]:
    for org_missing in [False, True]:

        mask = (
            flag_nonag.eq(nonag_missing)
            & flag_org.eq(org_missing)
        )

        part = data.loc[mask]

        if len(part) == 0:
            continue

        result = summarize_subset(part)

        result.update({
            "Nonag_Status": (
                "Missing" if nonag_missing else "Observed"
            ),
            "Organization_Type": (
                "Missing" if org_missing else "Observed"
            )
        })

        joint_rows.append(result)

joint_summary_144 = pd.DataFrame(joint_rows)

# ------------------------------------------------------------
# 6. Calculate share of total grouped SHAP importance
# ------------------------------------------------------------

all_shap_columns = [
    c for c in data.columns if c.startswith("SHAP__")
]

assert len(all_shap_columns) == 35

employment_columns = [
    f"SHAP__{parent}" for parent in predictors
]

total_absolute_shap = np.abs(
    data[all_shap_columns].to_numpy(dtype=float)
).sum(axis=1)

employment_absolute_shap = np.abs(
    data[employment_columns].to_numpy(dtype=float)
).sum(axis=1)

total_importance = weighted_mean(
    total_absolute_shap, w_all
)

employment_importance = weighted_mean(
    employment_absolute_shap, w_all
)

assert total_importance > 0

importance_share = (
    employment_importance / total_importance
)

# ------------------------------------------------------------
# 7. Save results
# ------------------------------------------------------------

missing_summary_144.to_csv(
    audit_dir / "Step1_44_Missingness_SHAP.csv",
    index=False
)

sector_summary_144.to_csv(
    audit_dir / "Step1_44_Sector_Missingness_SHAP.csv",
    index=False
)

joint_summary_144.to_csv(
    audit_dir / "Step1_44_Joint_Missingness.csv",
    index=False
)

# ------------------------------------------------------------
# 8. Display results
# ------------------------------------------------------------

print("\n========== SOURCE MISSINGNESS AND SHAP ==========")

print(
    missing_summary_144.round(5).to_string(index=False)
)

print("\n========== MISSINGNESS BY EMPLOYMENT SECTOR ==========")

print(
    sector_summary_144.round(5).to_string(index=False)
)

print("\n========== JOINT MISSINGNESS ==========")

print(
    joint_summary_144.round(5).to_string(index=False)
)

print("\n========== SHAP IMPORTANCE SHARE ==========")

print(
    "Employment-related predictors:",
    round(importance_share * 100, 2),
    "% of total grouped absolute SHAP importance"
)

# ------------------------------------------------------------
# 9. Final verification
# ------------------------------------------------------------

weighted_prevalence = weighted_mean(
    data["Actual"], w_all
)

assert np.isclose(
    weighted_prevalence, 0.237812, atol=1e-5
)

assert 0 <= importance_share <= 1

print("\n========== STEP 1.44 FINAL VERIFICATION ==========")

print("Held-out observations:", len(data))
print("Outer folds:", data["Outer_Fold"].nunique())
print("Original predictor groups:", len(all_shap_columns))
print("Employment predictors audited:", len(predictors))
print("Weighted low-work-intensity prevalence:",
      round(weighted_prevalence, 6))
print("Model refitting: NO")
print("SHAP recalculation: NO")
print("Output folder:", audit_dir)
print("STEP 1.44 VERIFICATION SUCCESSFUL.")






# %% STEP 1.45: EMPLOYMENT-PREDICTOR SENSITIVITY ANALYSIS

from pathlib import Path

script = Path(
    str(CODE_DIR / "Step1_45_Frozen_Boruta_RF_Ablation.py")
)

assert script.is_file(), f"File not found: {script}"

exec(
    compile(
        script.read_text(encoding="utf-8"),
        str(script),
        "exec"
    ),
    globals()
)






# %% STEP 1.46: FULLY NESTED RF SENSITIVITY ANALYSIS

import subprocess
import sys
from pathlib import Path

script = Path(
    str(CODE_DIR / "Step1_46_Fully_Nested_RF_Sensitivity.py")
)

root = Path(
    str(OUTPUT_DIR_PATH)
)

log_path = root / "Step1_46_Nested_Sensitivity_Log.txt"

assert script.is_file(), f"Script not found: {script}"

print("Starting Step 1.46...")
print("Log file:", log_path)

with log_path.open("w", encoding="utf-8") as log:

    result = subprocess.run(
        [sys.executable, "-u", str(script)],
        stdout=log,
        stderr=subprocess.STDOUT,
        check=False
    )

print("\nEXIT CODE:", result.returncode)

lines = log_path.read_text(
    encoding="utf-8",
    errors="replace"
).splitlines()

print("\n========== LAST 45 LOG LINES ==========")
print("\n".join(lines[-45:]))

if result.returncode == 0:
    print("\nSTEP 1.46 EXECUTION SUCCESSFUL.")
else:
    print("\nSTEP 1.46 INCOMPLETE - SEE LOG ABOVE.")





# %% STEP 1.47: PAIRED PSU BOOTSTRAP SENSITIVITY

import subprocess
import sys
from pathlib import Path

script = Path(
    str(CODE_DIR / "Step1_47_Paired_PSU_Bootstrap_Sensitivity.py")
)

root = Path(
    str(OUTPUT_DIR_PATH)
)

log_path = root / "Step1_47_Bootstrap_Log.txt"

assert script.is_file(), f"Script not found: {script}"

with log_path.open("w", encoding="utf-8") as log:
    result = subprocess.run(
        [sys.executable, "-u", str(script)],
        stdout=log,
        stderr=subprocess.STDOUT,
        check=False
    )

print("EXIT CODE:", result.returncode)

lines = log_path.read_text(
    encoding="utf-8",
    errors="replace"
).splitlines()

print("\n========== LAST 55 LOG LINES ==========")
print("\n".join(lines[-55:]))

if result.returncode == 0:
    print("\nSTEP 1.47 EXECUTION SUCCESSFUL.")
else:
    print("\nSTEP 1.47 INCOMPLETE - SEE ERROR ABOVE.")













