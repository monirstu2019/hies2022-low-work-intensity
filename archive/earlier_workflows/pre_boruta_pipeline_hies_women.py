"""
Pre-Boruta pipeline for:
Explaining Low Work Intensity among Working Women in Bangladesh:
An Explainable Machine Learning Analysis Using HIES 2022

What this script does BEFORE Boruta:
1. Loads HIES women dataset (.dta)
2. Keeps working women aged 15+ with valid earnings/hours data
3. Creates binary outcome: low_work_intensity
4. Creates class weights for imbalance handling
5. Handles HIES skip-pattern / structural missing values
6. Creates PCA-based wealth_living_index
7. Reduces predictors using theory-based selection
8. Groups rare categories for high-cardinality categorical variables
9. Produces descriptive EDA tables and charts
10. Saves Boruta-ready dataset

Author note:
- This script does NOT run Boruta yet.
- The next script should use X_boruta_raw, y_boruta, class_weights,
  numeric_boruta_vars, categorical_boruta_vars, and reduced_features_boruta.
"""

# ============================================================
# 0. Import libraries
# ============================================================

import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu, chi2_contingency
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ============================================================
# 1. User settings
# ============================================================

# Local Windows path. Change this if your file is somewhere else.
USER_FILE_PATH = os.environ.get("HIES_MERGED_DATA", "restricted_data/final_hies_women_lfp_earnings_ml_ready.dta")

# Output folder. All tables and figures will be saved here.
OUTPUT_DIR = os.environ.get("HIES_OUTPUT_DIR", "pre_boruta_outputs")

# Rare-category grouping threshold.
# Categories with fewer than 30 observations will be grouped as -88 = Rare/Other.
RARE_MIN_COUNT = 30


# ============================================================
# 2. Helper functions
# ============================================================

def resolve_file_path(user_path: str) -> str:
    """
    Resolve dataset path.
    Priority:
    1. User-supplied Windows/local path
    2. women.dta in current working directory
    3. /mnt/data/women.dta in ChatGPT environment
    """
    candidate_paths = [
        Path(user_path),
        Path("women.dta"),
        Path("/mnt/data/women.dta"),
    ]

    for path in candidate_paths:
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        "Could not find women.dta. Please update USER_FILE_PATH at the top of this script."
    )


def ensure_output_dir(output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def format_p_value(x):
    if pd.isna(x):
        return np.nan
    if x < 0.001:
        return "<0.001"
    return round(float(x), 3)


def fill_structural_zero(df: pd.DataFrame, value_var: str, indicator_var: str, no_values=[0, 2]):
    """
    For HIES skip-pattern value variables:
    If indicator says 'No', missing value_var is structurally zero.
    Example: if purchased_land_12m == No, land_purchase_value_12m = 0.
    """
    if value_var in df.columns and indicator_var in df.columns:
        before_missing = df[value_var].isna().sum()
        condition = df[value_var].isna() & df[indicator_var].isin(no_values)
        df.loc[condition, value_var] = 0
        after_missing = df[value_var].isna().sum()
        print(f"{value_var}: missing before = {before_missing}, after = {after_missing}")


def fill_not_applicable(df: pd.DataFrame, var: str, fill_value=-99):
    """
    For categorical HIES skip-pattern variables:
    Missing means not applicable due to survey skip.
    """
    if var in df.columns:
        before_missing = df[var].isna().sum()
        df[var] = df[var].fillna(fill_value)
        after_missing = df[var].isna().sum()
        print(f"{var}: missing before = {before_missing}, after = {after_missing}")


def group_rare_categories(
    df: pd.DataFrame,
    var: str,
    min_count: int = 30,
    rare_code=-88,
    missing_code=-99,
):
    """
    Groups small categories into rare/other category.
    This reduces sparse dummy variables before Boruta and ML.

    missing_code = -99 means Missing/Not applicable
    rare_code = -88 means Rare/Other
    """
    if var not in df.columns:
        return []

    temp = df[var].copy()
    temp = temp.fillna(missing_code)

    counts = temp.value_counts(dropna=False)
    rare_categories = counts[
        (counts < min_count) & (counts.index != missing_code)
    ].index

    temp = temp.replace(list(rare_categories), rare_code)
    df[var + "_grp"] = temp

    print("\nVariable:", var)
    print("Original categories:", df[var].nunique(dropna=False))
    print("Grouped categories:", df[var + "_grp"].nunique(dropna=False))
    print("Rare categories grouped:", len(rare_categories))

    return list(rare_categories)


def save_figure(output_dir: str, filename: str):
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {path}")


# ============================================================
# 3. Main pipeline
# ============================================================

def main():

    output_dir = ensure_output_dir(OUTPUT_DIR)
    file_path = resolve_file_path(USER_FILE_PATH)

    print("Dataset path:", file_path)
    print("Output folder:", output_dir)

    # ========================================================
    # 3.1 Load data
    # ========================================================

    df = pd.read_stata(file_path, convert_categoricals=False)
    print("\nFull dataset shape:", df.shape)

    # ========================================================
    # 3.2 Analytical sample
    # ========================================================

    if "sample_earnings_women_15plus" not in df.columns:
        raise KeyError("sample_earnings_women_15plus not found in dataset.")

    women = df[df["sample_earnings_women_15plus"] == 1].copy()
    print("Final working women sample:", women.shape)

    # ========================================================
    # 3.3 Outcome variable
    # ========================================================

    if "earn_annual_hours" not in women.columns:
        raise KeyError("earn_annual_hours not found in dataset.")

    p25_hours = women["earn_annual_hours"].quantile(0.25)

    women["low_work_intensity"] = np.where(
        women["earn_annual_hours"] <= p25_hours,
        1,
        0,
    )

    print("25th percentile annual working hours:", p25_hours)

    print("\nOutcome frequency:")
    print(women["low_work_intensity"].value_counts().sort_index())

    print("\nOutcome percentage:")
    print((women["low_work_intensity"].value_counts(normalize=True).sort_index() * 100).round(2))

    # ========================================================
    # 3.4 Class weights for imbalance
    # ========================================================

    classes = np.array([0, 1])
    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=women["low_work_intensity"],
    )

    class_weights = {
        int(k): float(v)
        for k, v in zip(classes, class_weights_array)
    }

    print("\nClass weights:")
    print(class_weights)

    # ========================================================
    # 3.5 HIES skip-pattern missing value handling
    # ========================================================

    women_ml = women.copy()

    # Asset / purchase / sale value variables
    fill_structural_zero(women_ml, "land_property_value", "owns_land_property_nonoperated")
    fill_structural_zero(women_ml, "land_purchase_value_12m", "purchased_land_12m")
    fill_structural_zero(women_ml, "house_flat_purchase_value_12m", "purchased_house_flat_12m")
    fill_structural_zero(women_ml, "other_asset_value", "owns_other_assets")
    fill_structural_zero(women_ml, "other_asset_purchase_value_12m", "purchased_other_assets_12m")
    fill_structural_zero(women_ml, "asset_sale_value_12m", "sold_assets_12m")

    # Education skip-pattern categorical variables
    education_skip_vars = [
        "current_class",
        "receives_scholarship_stipend",
        "education_place",
        "school_type_last",
        "curriculum",
        "reason_not_attending_school",
        "free_study_opportunity",
    ]

    for var in education_skip_vars:
        fill_not_applicable(women_ml, var)

    # Stipend amount: structurally zero if not receiving stipend or not applicable
    if "stipend_amount_12m" in women_ml.columns and "receives_scholarship_stipend" in women_ml.columns:
        before_missing = women_ml["stipend_amount_12m"].isna().sum()
        condition = (
            women_ml["stipend_amount_12m"].isna()
            & women_ml["receives_scholarship_stipend"].isin([0, 2, -99])
        )
        women_ml.loc[condition, "stipend_amount_12m"] = 0
        after_missing = women_ml["stipend_amount_12m"].isna().sum()
        print(f"stipend_amount_12m: missing before = {before_missing}, after = {after_missing}")

    # Employment and infrastructure skip-pattern variables
    fill_not_applicable(women_ml, "earn_emp_status_ag")
    fill_not_applicable(women_ml, "all_season_road_2km")

    # ========================================================
    # 3.6 PCA wealth/living index
    # ========================================================

    wealth_living_vars = [
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
        "sec9_total_observed_value",
    ]

    wealth_living_vars = [var for var in wealth_living_vars if var in women_ml.columns]

    print("\nVariables used for wealth/living PCA:")
    print(wealth_living_vars)

    X_wealth = women_ml[wealth_living_vars].copy()
    X_wealth = X_wealth.mask(X_wealth < 0, np.nan)
    X_wealth_log = np.log1p(X_wealth)

    wealth_imputer = SimpleImputer(strategy="median")
    X_wealth_imputed = wealth_imputer.fit_transform(X_wealth_log)

    wealth_scaler = StandardScaler()
    X_wealth_scaled = wealth_scaler.fit_transform(X_wealth_imputed)

    wealth_pca = PCA(n_components=1, random_state=42)
    wealth_index = wealth_pca.fit_transform(X_wealth_scaled).flatten()

    women_ml["wealth_living_index"] = wealth_index

    # Ensure higher index means better living condition
    if "sec9_total_observed_value" in women_ml.columns:
        corr_check = np.corrcoef(
            women_ml["wealth_living_index"],
            np.log1p(
                women_ml["sec9_total_observed_value"].fillna(
                    women_ml["sec9_total_observed_value"].median()
                )
            ),
        )[0, 1]

        if corr_check < 0:
            women_ml["wealth_living_index"] = -women_ml["wealth_living_index"]

    print("\nPCA explained variance ratio:")
    print(round(float(wealth_pca.explained_variance_ratio_[0]), 4))

    print("\nWealth/living index summary:")
    print(women_ml["wealth_living_index"].describe().round(3))

    # ========================================================
    # 3.7 Reduced theoretically relevant feature set
    # ========================================================

    reduced_features = [
        # Demographic
        "age",
        "marital_status",
        "relation_head",

        # Education
        "literate_basic",
        "highest_class_completed",

        # Digital and financial access
        "has_mobile",
        "has_bank_account",
        "has_mfs_account",
        "internet_used",
        "has_computer",

        # Functional difficulty
        "any_functional_difficulty",

        # Geography
        "rural_urban",
        "division_code",
        "rmo_code",
        "domain16",

        # Household and food security
        "hh_MEMBER",
        "fies_raw_score",
        "wealth_living_index",

        # Housing and infrastructure
        "toilet_type",
        "drinking_water_source",
        "light_source",
        "cooking_stove_type",
        "cooking_fuel",
        "electricity_hours_24h",
        "transport_stop_halfkm",
        "transport_stop_time_min",
        "regular_waste_collection",
        "slum_household",

        # Employment structure
        "earn_sector",
        "earn_emp_status_ag",
        "earn_emp_status_nonag",
        "earn_org_type",
        "earn_ind_code",
        "earn_occ_code",
    ]

    reduced_features = [var for var in reduced_features if var in women_ml.columns]

    leakage_vars = [
        "low_work_intensity",
        "earn_annual_hours",
        "earn_valid_hours_count",
        "earn_annual_total",
        "earn_cash_daily_annual",
        "earn_inkind_daily_annual",
        "earn_gross_monthly_annual",
        "earn_net_monthly_annual",
        "earn_inkind_other_annual",
        "earn_hourly",
        "log_earn_hourly",
        "sample_earnings_women_15plus",
        "sample_earnings_15plus",
        "sample_lfp_women_15plus",
        "sample_lfp_all_15plus",
        "hhid",
        "pid",
        "psu",
        "household_weight",
        "population_weight",
    ]

    reduced_features = [var for var in reduced_features if var not in leakage_vars]

    print("\nNumber of reduced features before grouping:", len(reduced_features))
    print(reduced_features)

    # Reduced feature quality
    reduced_quality = []
    for var in reduced_features:
        reduced_quality.append({
            "Variable": var,
            "Missing_N": women_ml[var].isna().sum(),
            "Missing_percent": women_ml[var].isna().mean() * 100,
            "Unique_values": women_ml[var].nunique(dropna=True),
            "Data_type": str(women_ml[var].dtype),
        })

    reduced_quality = pd.DataFrame(reduced_quality).sort_values(
        by="Unique_values",
        ascending=False,
    )

    print("\nReduced feature quality:")
    print(reduced_quality.round(2))

    reduced_quality.to_excel(
        os.path.join(output_dir, "Reduced_Feature_Quality.xlsx"),
        index=False,
    )

    # ========================================================
    # 3.8 Group rare categories for occupation and industry
    # ========================================================

    women_boruta = women_ml.copy()

    rare_occ_categories = group_rare_categories(
        women_boruta,
        "earn_occ_code",
        min_count=RARE_MIN_COUNT,
    )

    rare_ind_categories = group_rare_categories(
        women_boruta,
        "earn_ind_code",
        min_count=RARE_MIN_COUNT,
    )

    reduced_features_boruta = [
        var for var in reduced_features
        if var not in ["earn_occ_code", "earn_ind_code"]
    ]

    for var in ["earn_occ_code_grp", "earn_ind_code_grp"]:
        if var in women_boruta.columns:
            reduced_features_boruta.append(var)

    print("\nFinal number of Boruta-ready features:", len(reduced_features_boruta))
    print(reduced_features_boruta)

    # ========================================================
    # 3.9 Outcome distribution table
    # ========================================================

    lowwi_freq = women_boruta["low_work_intensity"].value_counts().sort_index()
    lowwi_percent = (
        women_boruta["low_work_intensity"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    outcome_distribution = pd.DataFrame({
        "Outcome_Code": [0, 1],
        "Outcome": ["Not low work intensity", "Low work intensity"],
        "Frequency": lowwi_freq.values,
        "Percent": lowwi_percent.round(2).values,
    })

    print("\nOutcome distribution:")
    print(outcome_distribution)

    outcome_distribution.to_excel(
        os.path.join(output_dir, "Table_Outcome_Distribution.xlsx"),
        index=False,
    )

    # ========================================================
    # 3.10 Donut chart for outcome variable
    # ========================================================

    plt.figure(figsize=(6, 6))

    plt.pie(
        outcome_distribution["Frequency"],
        labels=outcome_distribution["Outcome"],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"width": 0.40},
    )

    plt.title("Distribution of Low Work Intensity among Working Women")
    plt.tight_layout()
    save_figure(output_dir, "Figure_1_Donut_Low_Work_Intensity.png")
    plt.show()

    # ========================================================
    # 3.11 Histogram of annual working hours
    # ========================================================

    plt.figure(figsize=(8, 5))

    plt.hist(
        women_boruta["earn_annual_hours"].dropna(),
        bins=30,
    )

    plt.axvline(
        p25_hours,
        linestyle="--",
        linewidth=2,
        label=f"25th percentile = {p25_hours:.0f} hours",
    )

    plt.xlabel("Annual working hours")
    plt.ylabel("Number of women")
    plt.title("Distribution of Annual Working Hours")
    plt.legend()
    plt.tight_layout()
    save_figure(output_dir, "Figure_2_Annual_Working_Hours_Distribution.png")
    plt.show()

    # ========================================================
    # 3.12 Continuous descriptive table by outcome
    # ========================================================

    continuous_eda_vars = [
        "age",
        "highest_class_completed",
        "hh_MEMBER",
        "fies_raw_score",
        "wealth_living_index",
        "electricity_hours_24h",
        "transport_stop_time_min",
        "earn_annual_hours",
    ]

    continuous_eda_vars = [var for var in continuous_eda_vars if var in women_boruta.columns]

    continuous_table = []

    for var in continuous_eda_vars:
        group0 = women_boruta.loc[women_boruta["low_work_intensity"] == 0, var].dropna()
        group1 = women_boruta.loc[women_boruta["low_work_intensity"] == 1, var].dropna()

        if len(group0) > 0 and len(group1) > 0:
            stat, p_value = mannwhitneyu(group0, group1, alternative="two-sided")
        else:
            p_value = np.nan

        continuous_table.append({
            "Variable": var,
            "Overall_N": women_boruta[var].notna().sum(),
            "Overall_Mean": women_boruta[var].mean(),
            "Overall_SD": women_boruta[var].std(),
            "Overall_Median": women_boruta[var].median(),
            "Overall_P25": women_boruta[var].quantile(0.25),
            "Overall_P75": women_boruta[var].quantile(0.75),
            "Not_Low_N": group0.count(),
            "Not_Low_Mean": group0.mean(),
            "Not_Low_SD": group0.std(),
            "Not_Low_Median": group0.median(),
            "Low_N": group1.count(),
            "Low_Mean": group1.mean(),
            "Low_SD": group1.std(),
            "Low_Median": group1.median(),
            "Mean_Difference_Low_minus_NotLow": group1.mean() - group0.mean(),
            "P_value": p_value,
        })

    table_continuous = pd.DataFrame(continuous_table)
    table_continuous["P_value_display"] = table_continuous["P_value"].apply(format_p_value)
    table_continuous = table_continuous.round(3)

    print("\nContinuous descriptive table:")
    print(table_continuous)

    table_continuous.to_excel(
        os.path.join(output_dir, "Table_Continuous_By_Low_Work_Intensity.xlsx"),
        index=False,
    )

    # ========================================================
    # 3.13 Categorical descriptive table by outcome
    # ========================================================

    categorical_eda_vars = [
        "rural_urban",
        "division_code",
        "rmo_code",
        "domain16",
        "marital_status",
        "relation_head",
        "literate_basic",
        "has_mobile",
        "has_bank_account",
        "has_mfs_account",
        "internet_used",
        "has_computer",
        "any_functional_difficulty",
        "toilet_type",
        "drinking_water_source",
        "light_source",
        "cooking_stove_type",
        "cooking_fuel",
        "transport_stop_halfkm",
        "regular_waste_collection",
        "slum_household",
        "earn_sector",
        "earn_emp_status_ag",
        "earn_emp_status_nonag",
        "earn_org_type",
        "earn_occ_code_grp",
        "earn_ind_code_grp",
    ]

    categorical_eda_vars = [var for var in categorical_eda_vars if var in women_boruta.columns]

    categorical_tables = []

    for var in categorical_eda_vars:
        temp = women_boruta[[var, "low_work_intensity"]].copy()
        temp[var] = temp[var].astype("object").where(temp[var].notna(), "Missing/Not applicable")

        tab = pd.crosstab(temp[var], temp["low_work_intensity"])

        if 0 not in tab.columns:
            tab[0] = 0
        if 1 not in tab.columns:
            tab[1] = 0

        tab = tab[[0, 1]]
        tab["Total_N"] = tab[0] + tab[1]
        tab["Not_Low_Percent_within_Category"] = (tab[0] / tab["Total_N"] * 100).round(2)
        tab["Low_Percent_within_Category"] = (tab[1] / tab["Total_N"] * 100).round(2)

        try:
            chi2, p_value, dof, expected = chi2_contingency(tab[[0, 1]])
        except Exception:
            p_value = np.nan

        tab = tab.reset_index()
        tab.columns = [
            "Category",
            "Not_Low_N",
            "Low_N",
            "Total_N",
            "Not_Low_Percent_within_Category",
            "Low_Percent_within_Category",
        ]

        tab.insert(0, "Variable", var)
        tab["P_value"] = p_value
        tab["P_value_display"] = tab["P_value"].apply(format_p_value)

        categorical_tables.append(tab)

    table_categorical = pd.concat(categorical_tables, ignore_index=True)

    print("\nCategorical descriptive table first rows:")
    print(table_categorical.head())

    table_categorical.to_excel(
        os.path.join(output_dir, "Table_Categorical_By_Low_Work_Intensity.xlsx"),
        index=False,
    )

    significant_categorical_vars = (
        table_categorical[table_categorical["P_value"] < 0.05]["Variable"]
        .drop_duplicates()
        .tolist()
    )

    print("\nSignificant categorical variables:")
    print(significant_categorical_vars)

    # ========================================================
    # 3.14 Top low-work-intensity groups table and bar chart
    # ========================================================

    top_lowwi_groups = table_categorical[
        table_categorical["Total_N"] >= 30
    ].sort_values(
        by="Low_Percent_within_Category",
        ascending=False,
    )

    print("\nTop low-work-intensity groups:")
    print(top_lowwi_groups.head(25))

    top_lowwi_groups.to_excel(
        os.path.join(output_dir, "Table_Top_Low_Work_Intensity_Groups.xlsx"),
        index=False,
    )

    top_plot = top_lowwi_groups.head(15).copy()
    top_plot["Label"] = top_plot["Variable"].astype(str) + " = " + top_plot["Category"].astype(str)

    plt.figure(figsize=(9, 7))
    plt.barh(top_plot["Label"], top_plot["Low_Percent_within_Category"])
    plt.xlabel("Low work intensity (%)")
    plt.ylabel("Group")
    plt.title("Top Groups with Highest Low Work Intensity")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    save_figure(output_dir, "Figure_3_Top_Low_Work_Intensity_Groups.png")
    plt.show()

    # ========================================================
    # 3.15 Stacked bar charts for selected categorical variables
    # ========================================================

    def plot_stacked_bar_by_outcome(df_input, var, min_count=30):
        temp = df_input[[var, "low_work_intensity"]].copy()
        temp[var] = temp[var].astype("object").where(temp[var].notna(), "Missing/Not applicable")

        tab = pd.crosstab(temp[var], temp["low_work_intensity"])

        if 0 not in tab.columns:
            tab[0] = 0
        if 1 not in tab.columns:
            tab[1] = 0

        tab = tab[[0, 1]]
        tab["Total"] = tab[0] + tab[1]
        tab = tab[tab["Total"] >= min_count].copy()

        if tab.shape[0] == 0:
            print(f"Skipping {var}: no category with at least {min_count} observations.")
            return

        tab_pct = tab[[0, 1]].div(tab["Total"], axis=0) * 100
        tab_pct.columns = ["Not low work intensity", "Low work intensity"]
        tab_pct = tab_pct.sort_values(by="Low work intensity", ascending=False)

        plt.figure(figsize=(9, 5))

        bottom_values = np.zeros(len(tab_pct))

        for col in tab_pct.columns:
            plt.bar(tab_pct.index.astype(str), tab_pct[col], bottom=bottom_values, label=col)
            bottom_values += tab_pct[col].values

        plt.ylabel("Percent within category")
        plt.xlabel(var)
        plt.title(f"Low Work Intensity by {var}")
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()

        file_name = f"Stacked_Bar_{var}.png".replace("/", "_")
        save_figure(output_dir, file_name)
        plt.show()

    stacked_bar_vars = [
        "rural_urban",
        "division_code",
        "marital_status",
        "literate_basic",
        "has_bank_account",
        "has_mfs_account",
        "internet_used",
        "earn_sector",
        "earn_org_type",
    ]

    stacked_bar_vars = [var for var in stacked_bar_vars if var in women_boruta.columns]

    for var in stacked_bar_vars:
        plot_stacked_bar_by_outcome(women_boruta, var, min_count=30)

    # ========================================================
    # 3.16 Bar charts for selected continuous variables by outcome
    # ========================================================

    continuous_bar_vars = [
        "highest_class_completed",
        "hh_MEMBER",
        "fies_raw_score",
        "wealth_living_index",
    ]

    continuous_bar_vars = [var for var in continuous_bar_vars if var in women_boruta.columns]

    for var in continuous_bar_vars:
        temp = women_boruta.groupby("low_work_intensity")[var].mean().reset_index()
        temp["Outcome"] = temp["low_work_intensity"].map({
            0: "Not low work intensity",
            1: "Low work intensity",
        })

        plt.figure(figsize=(6, 4))
        plt.bar(temp["Outcome"], temp[var])
        plt.ylabel(f"Mean {var}")
        plt.xlabel("Outcome group")
        plt.title(f"Mean {var} by Low Work Intensity")
        plt.xticks(rotation=15)
        plt.tight_layout()

        file_name = f"Bar_Mean_{var}_by_Outcome.png".replace("/", "_")
        save_figure(output_dir, file_name)
        plt.show()

    # ========================================================
    # 3.17 Final Boruta-ready objects and quality table
    # ========================================================

    target = "low_work_intensity"

    X_boruta_raw = women_boruta[reduced_features_boruta].copy()
    y_boruta = women_boruta[target].copy()

    print("\nBoruta-ready X shape:", X_boruta_raw.shape)
    print("Boruta-ready y shape:", y_boruta.shape)

    print("\nOutcome distribution for Boruta:")
    print(y_boruta.value_counts().sort_index())

    print("\nClass weights to use in Boruta Random Forest:")
    print(class_weights)

    numeric_boruta_vars = [
        "age",
        "highest_class_completed",
        "hh_MEMBER",
        "fies_raw_score",
        "wealth_living_index",
        "electricity_hours_24h",
        "transport_stop_time_min",
    ]

    numeric_boruta_vars = [var for var in numeric_boruta_vars if var in reduced_features_boruta]

    categorical_boruta_vars = [
        var for var in reduced_features_boruta
        if var not in numeric_boruta_vars
    ]

    print("\nNumeric variables for Boruta:")
    print(numeric_boruta_vars)

    print("\nCategorical variables for Boruta:")
    print(categorical_boruta_vars)

    boruta_quality = []

    for var in reduced_features_boruta:
        boruta_quality.append({
            "Variable": var,
            "Missing_N": women_boruta[var].isna().sum(),
            "Missing_percent": women_boruta[var].isna().mean() * 100,
            "Unique_values": women_boruta[var].nunique(dropna=True),
            "Data_type": str(women_boruta[var].dtype),
        })

    boruta_quality = pd.DataFrame(boruta_quality).sort_values(
        by="Unique_values",
        ascending=False,
    )

    print("\nBoruta-ready feature quality:")
    print(boruta_quality.round(2))

    boruta_quality.to_excel(
        os.path.join(output_dir, "Boruta_Ready_Feature_Quality.xlsx"),
        index=False,
    )

    # Save Boruta-ready dataset
    boruta_ready_df = women_boruta[reduced_features_boruta + ["low_work_intensity"]].copy()

    boruta_ready_df.to_csv(
        os.path.join(output_dir, "Boruta_Ready_Dataset.csv"),
        index=False,
    )

    print("\nBoruta-ready dataset saved successfully.")

    # Export all important tables into one workbook
    excel_path = os.path.join(output_dir, "Pre_Boruta_EDA_Tables.xlsx")

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        outcome_distribution.to_excel(writer, sheet_name="Outcome_Distribution", index=False)
        reduced_quality.to_excel(writer, sheet_name="Reduced_Feature_Quality", index=False)
        boruta_quality.to_excel(writer, sheet_name="Boruta_Feature_Quality", index=False)
        table_continuous.to_excel(writer, sheet_name="Continuous_By_Outcome", index=False)
        table_categorical.to_excel(writer, sheet_name="Categorical_By_Outcome", index=False)
        top_lowwi_groups.to_excel(writer, sheet_name="Top_LowWI_Groups", index=False)

    print("\nAll pre-Boruta EDA tables exported:")
    print(excel_path)

    print("\nEND OF PRE-BORUTA PIPELINE")
    print("Next step: Run Boruta with class-weighted Random Forest.")

    # Return useful objects if running interactively
    return {
        "women_boruta": women_boruta,
        "X_boruta_raw": X_boruta_raw,
        "y_boruta": y_boruta,
        "class_weights": class_weights,
        "numeric_boruta_vars": numeric_boruta_vars,
        "categorical_boruta_vars": categorical_boruta_vars,
        "reduced_features_boruta": reduced_features_boruta,
        "boruta_quality": boruta_quality,
        "table_continuous": table_continuous,
        "table_categorical": table_categorical,
        "top_lowwi_groups": top_lowwi_groups,
    }


# ============================================================
# 4. Run script
# ============================================================

if __name__ == "__main__":
    main()
