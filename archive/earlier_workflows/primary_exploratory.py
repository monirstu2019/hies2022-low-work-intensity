# ==============================
# Step 5.1: Sample + Outcome + Class Weight
# ==============================

import pandas as pd
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Use this path in your own computer:
file_path = os.environ.get("HIES_MERGED_DATA", "restricted_data/final_hies_women_lfp_earnings_ml_ready.dta")

# If running inside ChatGPT uploaded file environment, use:
# file_path = "/mnt/data/women.dta"

# Load dataset
df = pd.read_stata(file_path, convert_categoricals=False)

print("Full dataset shape:", df.shape)

# Keep only working women aged 15+ with valid earnings/hours information
women = df[df["sample_earnings_women_15plus"] == 1].copy()

print("Final working women sample:", women.shape)

# Check 25th percentile of annual working hours
p25_hours = women["earn_annual_hours"].quantile(0.25)

print("25th percentile annual working hours:", p25_hours)

# Create outcome variable
women["low_work_intensity"] = np.where(
    women["earn_annual_hours"] <= p25_hours,
    1,
    0
)

# Check outcome distribution
print("\nOutcome frequency:")
print(women["low_work_intensity"].value_counts().sort_index())

print("\nOutcome percentage:")
print((women["low_work_intensity"].value_counts(normalize=True).sort_index() * 100).round(2))

# Create class weights for imbalance
classes = np.array([0, 1])

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=women["low_work_intensity"]
)

class_weights = dict(zip(classes, class_weights_array))

print("\nClass weights:")
print(class_weights)




# ==========================================
# Step 5.1B: Build Relevant Candidate Feature Set
# ==========================================

candidate_features = [

    # -------------------------
    # 1. Demographic variables
    # -------------------------
    "age",
    "relation_head",
    "marital_status",

    # -------------------------
    # 2. Digital and financial access
    # -------------------------
    "has_mobile",
    "has_bank_account",
    "has_mfs_account",
    "internet_used",

    # -------------------------
    # 3. Education variables
    # -------------------------
    "literate_basic",
    "can_read_letter",
    "can_write_letter",
    "ever_attended_education",
    "highest_class_completed",
    "currently_attending_school",
    "current_class",
    "receives_scholarship_stipend",
    "education_place",
    "school_type_last",
    "curriculum",
    "father_education",
    "mother_education",
    "guardian_education",
    "reason_not_attending_school",
    "stipend_type",
    "stipend_amount_12m",
    "free_study_opportunity",

    # -------------------------
    # 4. Functional difficulty
    # -------------------------
    "any_functional_difficulty",
    "severe_functional_difficulty",

    # -------------------------
    # 5. Geographic variables
    # -------------------------
    "division_code",
    "district_code",
    "rmo_code",
    "rural_urban",
    "stratum16",
    "domain16",

    # -------------------------
    # 6. Household composition
    # -------------------------
    "hh_MEMBER",

    # -------------------------
    # 7. Food security
    # -------------------------
    "fies_raw_score",
    "fies_valid_items",
    "fies_Q10_1_yes",
    "fies_Q10_2_yes",
    "fies_Q10_3_yes",
    "fies_Q10_4_yes",
    "fies_Q10_5_yes",
    "fies_Q10_6_yes",
    "fies_Q10_7_yes",
    "fies_Q10_8_yes",

    # -------------------------
    # 8. Housing and living condition
    # -------------------------
    "hh_rooms",
    "wall_material",
    "roof_material",
    "covered_area",
    "toilet_type",
    "shared_toilet",
    "drinking_water_source",
    "other_water_source",
    "water_collection_time_min",
    "handwashing_place",
    "has_handwashing_material",
    "light_source",
    "electricity_hours_24h",
    "cooking_stove_type",
    "cooking_fuel",
    "cooking_location",
    "has_landline",
    "has_computer",
    "house_use",
    "house_value",
    "slum_household",
    "all_season_road_2km",
    "transport_stop_halfkm",
    "transport_stop_time_min",
    "regular_waste_collection",

    # -------------------------
    # 9. Asset ownership and wealth
    # -------------------------
    "owns_land_property_nonoperated",
    "land_property_value",
    "purchased_land_12m",
    "land_purchase_value_12m",
    "purchased_house_flat_12m",
    "house_flat_purchase_value_12m",
    "owns_other_assets",
    "other_asset_value",
    "purchased_other_assets_12m",
    "other_asset_purchase_value_12m",
    "sold_assets_12m",
    "asset_sale_value_12m",

    # -------------------------
    # 10. Consumption and durable assets
    # -------------------------
    "sec9a1_food_value",
    "sec9a1_item_count",
    "sec9b2_value_total",
    "sec9b2_item_count",
    "sec9c_cash_credit_value",
    "sec9c_ownprod_received_value",
    "sec9c_total_value",
    "sec9c_item_count",
    "sec9d1_value_total",
    "sec9d1_amount_total",
    "sec9d1_item_count",
    "sec9d2_value_total",
    "sec9d2_amount_total",
    "sec9d2_item_count",
    "durable_owned_count",
    "durable_total_number",
    "durable_total_value",
    "durable_item_count",
    "sec9_total_observed_value",

    # -------------------------
    # 11. Employment structure variables
    # -------------------------
    "earn_occ_code",
    "earn_ind_code",
    "earn_sector",
    "earn_emp_status_ag",
    "earn_emp_status_nonag",
    "earn_org_type"
]

# Keep only variables available in your dataset
candidate_features = [var for var in candidate_features if var in women.columns]

print("Number of candidate features before leakage removal:", len(candidate_features))

# Leakage variables: these must not enter the predictor set
leakage_vars = [
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
    "population_weight"
]

# Remove leakage variables
candidate_features = [var for var in candidate_features if var not in leakage_vars]

print("Number of candidate features after leakage removal:", len(candidate_features))

# Show final selected candidate variables
candidate_features



# Check whether any leakage variable was inside candidate_features

leakage_found = [var for var in candidate_features if var in leakage_vars]

print("Leakage variables found in candidate list:")
print(leakage_found)

print("Number of leakage variables found:", len(leakage_found))





# =====================================================
# Step 5.1C: Missing Value and Variable Quality Check
# =====================================================

feature_quality = []

for var in candidate_features:
    feature_quality.append({
        "Variable": var,
        "Non_missing_N": women[var].notna().sum(),
        "Missing_N": women[var].isna().sum(),
        "Missing_percent": women[var].isna().mean() * 100,
        "Unique_values": women[var].nunique(dropna=True),
        "Data_type": str(women[var].dtype)
    })

feature_quality = pd.DataFrame(feature_quality)

# Sort by missing percentage
feature_quality = feature_quality.sort_values(
    by="Missing_percent",
    ascending=False
).reset_index(drop=True)

feature_quality.round(2)




# =====================================================
# Summary of variable quality
# =====================================================

high_missing_vars = feature_quality[
    feature_quality["Missing_percent"] > 50
]["Variable"].tolist()

constant_vars = feature_quality[
    feature_quality["Unique_values"] <= 1
]["Variable"].tolist()

high_cardinality_vars = feature_quality[
    feature_quality["Unique_values"] > 50
]["Variable"].tolist()

print("Total candidate features:", len(candidate_features))
print("Variables with more than 50% missing:", len(high_missing_vars))
print(high_missing_vars)

print("\nConstant or near-empty variables:", len(constant_vars))
print(constant_vars)

print("\nVariables with more than 50 unique values:", len(high_cardinality_vars))
print(high_cardinality_vars)






# =====================================================
# Step 5.1D: Handle Structural Missing from HIES Skip Patterns
# =====================================================

women_ml = women.copy()

# -----------------------------------------------------
# Helper function 1:
# Fill value variables with 0 when the related yes/no indicator says "No"
# This is structural missing, not real information loss.
# -----------------------------------------------------

def fill_structural_zero(df, value_var, indicator_var, no_values=[0, 2]):
    """
    If indicator says No, missing value_var means structurally zero.
    Example: if purchased_land_12m == 0/2, then land_purchase_value_12m = 0.
    """
    if value_var in df.columns and indicator_var in df.columns:
        before_missing = df[value_var].isna().sum()
        
        condition = df[value_var].isna() & df[indicator_var].isin(no_values)
        df.loc[condition, value_var] = 0
        
        after_missing = df[value_var].isna().sum()
        
        print(f"{value_var}: missing before = {before_missing}, after = {after_missing}")


# -----------------------------------------------------
# A. Asset / purchase / sale value variables
# -----------------------------------------------------

fill_structural_zero(
    women_ml,
    value_var="land_property_value",
    indicator_var="owns_land_property_nonoperated"
)

fill_structural_zero(
    women_ml,
    value_var="land_purchase_value_12m",
    indicator_var="purchased_land_12m"
)

fill_structural_zero(
    women_ml,
    value_var="house_flat_purchase_value_12m",
    indicator_var="purchased_house_flat_12m"
)

fill_structural_zero(
    women_ml,
    value_var="other_asset_value",
    indicator_var="owns_other_assets"
)

fill_structural_zero(
    women_ml,
    value_var="other_asset_purchase_value_12m",
    indicator_var="purchased_other_assets_12m"
)

fill_structural_zero(
    women_ml,
    value_var="asset_sale_value_12m",
    indicator_var="sold_assets_12m"
)


# -----------------------------------------------------
# Helper function 2:
# Fill categorical skip-pattern missing values with -99
# Meaning: Not applicable due to survey skip pattern
# -----------------------------------------------------

def fill_not_applicable(df, var, fill_value=-99):
    if var in df.columns:
        before_missing = df[var].isna().sum()
        df[var] = df[var].fillna(fill_value)
        after_missing = df[var].isna().sum()
        print(f"{var}: missing before = {before_missing}, after = {after_missing}")


# -----------------------------------------------------
# B. Education skip-pattern variables
# -----------------------------------------------------

skip_categorical_vars = [
    "current_class",
    "receives_scholarship_stipend",
    "education_place",
    "school_type_last",
    "curriculum",
    "reason_not_attending_school",
    "free_study_opportunity"
]

for var in skip_categorical_vars:
    fill_not_applicable(women_ml, var)


# -----------------------------------------------------
# C. Stipend amount
# If not receiving stipend, amount is structurally zero
# -----------------------------------------------------

if "stipend_amount_12m" in women_ml.columns and "receives_scholarship_stipend" in women_ml.columns:
    before_missing = women_ml["stipend_amount_12m"].isna().sum()
    
    condition = women_ml["stipend_amount_12m"].isna() & women_ml["receives_scholarship_stipend"].isin([0, 2, -99])
    women_ml.loc[condition, "stipend_amount_12m"] = 0
    
    after_missing = women_ml["stipend_amount_12m"].isna().sum()
    
    print(f"stipend_amount_12m: missing before = {before_missing}, after = {after_missing}")


# -----------------------------------------------------
# D. Employment skip-pattern variable
# Agricultural employment status may be not applicable for non-agricultural workers
# -----------------------------------------------------

fill_not_applicable(women_ml, "earn_emp_status_ag")


# -----------------------------------------------------
# E. Location/infrastructure skip-pattern variable
# -----------------------------------------------------

fill_not_applicable(women_ml, "all_season_road_2km")


# -----------------------------------------------------
# F. Remove truly unhelpful constant variables
# These had no useful variation in your sample
# -----------------------------------------------------

drop_due_to_constant = [
    "guardian_education",
    "stipend_type",
    "durable_item_count"
]

drop_due_to_constant = [var for var in drop_due_to_constant if var in candidate_features]

candidate_features_clean = [
    var for var in candidate_features
    if var not in drop_due_to_constant
]

print("\nDropped constant/unhelpful variables:")
print(drop_due_to_constant)

print("\nCandidate features before cleaning:", len(candidate_features))
print("Candidate features after cleaning:", len(candidate_features_clean))





# =====================================================
# Re-check missing values after skip-pattern handling
# =====================================================

feature_quality_after = []

for var in candidate_features_clean:
    feature_quality_after.append({
        "Variable": var,
        "Non_missing_N": women_ml[var].notna().sum(),
        "Missing_N": women_ml[var].isna().sum(),
        "Missing_percent": women_ml[var].isna().mean() * 100,
        "Unique_values": women_ml[var].nunique(dropna=True),
        "Data_type": str(women_ml[var].dtype)
    })

feature_quality_after = pd.DataFrame(feature_quality_after)

high_missing_after = feature_quality_after[
    feature_quality_after["Missing_percent"] > 50
]["Variable"].tolist()

constant_after = feature_quality_after[
    feature_quality_after["Unique_values"] <= 1
]["Variable"].tolist()

high_cardinality_after = feature_quality_after[
    feature_quality_after["Unique_values"] > 50
]["Variable"].tolist()

print("Variables with more than 50% missing after skip-pattern handling:", len(high_missing_after))
print(high_missing_after)

print("\nConstant variables after cleaning:", len(constant_after))
print(constant_after)

print("\nHigh-cardinality variables after cleaning:", len(high_cardinality_after))
print(high_cardinality_after)

feature_quality_after.sort_values("Missing_percent", ascending=False).head(20).round(2)






# =====================================================
# Step 5.1E: Prepare Variables for Descriptive Analysis
# =====================================================

# Continuous variables for descriptive statistics
continuous_desc_vars = [
    "age",
    "highest_class_completed",
    "hh_MEMBER",
    "hh_rooms",
    "covered_area",
    "water_collection_time_min",
    "electricity_hours_24h",
    "transport_stop_time_min",
    "fies_raw_score",
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
    "earn_annual_hours"
]

# Keep only available variables
continuous_desc_vars = [
    var for var in continuous_desc_vars 
    if var in women_ml.columns
]

# Categorical variables for descriptive statistics
categorical_desc_vars = [
    "marital_status",
    "relation_head",
    "rural_urban",
    "division_code",
    "rmo_code",
    "has_mobile",
    "has_bank_account",
    "has_mfs_account",
    "internet_used",
    "literate_basic",
    "can_read_letter",
    "can_write_letter",
    "ever_attended_education",
    "currently_attending_school",
    "any_functional_difficulty",
    "severe_functional_difficulty",
    "toilet_type",
    "shared_toilet",
    "drinking_water_source",
    "light_source",
    "cooking_stove_type",
    "cooking_fuel",
    "has_computer",
    "slum_household",
    "transport_stop_halfkm",
    "regular_waste_collection",
    "owns_land_property_nonoperated",
    "owns_other_assets",
    "purchased_land_12m",
    "purchased_house_flat_12m",
    "sold_assets_12m",
    "earn_sector",
    "earn_emp_status_ag",
    "earn_emp_status_nonag",
    "earn_org_type",
    "earn_ind_code",
    "earn_occ_code"
]

# Keep only available variables
categorical_desc_vars = [
    var for var in categorical_desc_vars 
    if var in women_ml.columns
]

print("Continuous descriptive variables:", len(continuous_desc_vars))
print(continuous_desc_vars)

print("\nCategorical descriptive variables:", len(categorical_desc_vars))
print(categorical_desc_vars)



# =====================================================
# Check unique values of categorical descriptive variables
# =====================================================

cat_unique_check = []

for var in categorical_desc_vars:
    cat_unique_check.append({
        "Variable": var,
        "Unique_values": women_ml[var].nunique(dropna=True),
        "Missing_percent": women_ml[var].isna().mean() * 100
    })

cat_unique_check = pd.DataFrame(cat_unique_check).sort_values(
    by="Unique_values", 
    ascending=False
)

cat_unique_check




# =====================================================
# Step 5.1F: Table 1A - Continuous Characteristics
# =====================================================

from scipy.stats import mannwhitneyu

continuous_desc_vars = [
    "age",
    "highest_class_completed",
    "hh_MEMBER",
    "hh_rooms",
    "covered_area",
    "water_collection_time_min",
    "electricity_hours_24h",
    "transport_stop_time_min",
    "fies_raw_score",
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
    "earn_annual_hours"
]

# Keep only available variables
continuous_desc_vars = [
    var for var in continuous_desc_vars
    if var in women_ml.columns
]

continuous_table = []

for var in continuous_desc_vars:
    
    group0 = women_ml.loc[
        women_ml["low_work_intensity"] == 0, var
    ].dropna()
    
    group1 = women_ml.loc[
        women_ml["low_work_intensity"] == 1, var
    ].dropna()
    
    # Mann-Whitney U test is safer because income/assets/consumption are skewed
    if len(group0) > 0 and len(group1) > 0:
        try:
            stat, p_value = mannwhitneyu(
                group0,
                group1,
                alternative="two-sided"
            )
        except:
            p_value = np.nan
    else:
        p_value = np.nan
    
    continuous_table.append({
        "Variable": var,
        
        "Overall_N": women_ml[var].notna().sum(),
        "Overall_Mean": women_ml[var].mean(),
        "Overall_SD": women_ml[var].std(),
        "Overall_Median": women_ml[var].median(),
        "Overall_P25": women_ml[var].quantile(0.25),
        "Overall_P75": women_ml[var].quantile(0.75),
        
        "Not_Low_N": group0.count(),
        "Not_Low_Mean": group0.mean(),
        "Not_Low_SD": group0.std(),
        "Not_Low_Median": group0.median(),
        
        "Low_N": group1.count(),
        "Low_Mean": group1.mean(),
        "Low_SD": group1.std(),
        "Low_Median": group1.median(),
        
        "P_value": p_value
    })

table1a_continuous = pd.DataFrame(continuous_table)

# Round values
table1a_continuous = table1a_continuous.round(3)

table1a_continuous



# =====================================================
# Step 5.1F Revised: PCA Wealth/Living Index + Reduced Feature Set
# =====================================================

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------------------------------------
# 1. Variables used to construct wealth/living index
# ---------------------------------------------

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
    "sec9_total_observed_value"
]

wealth_living_vars = [
    var for var in wealth_living_vars
    if var in women_ml.columns
]

print("Variables used for wealth/living PCA:")
print(wealth_living_vars)

# ---------------------------------------------
# 2. Prepare PCA input
# ---------------------------------------------

X_wealth = women_ml[wealth_living_vars].copy()

# Convert negative values to missing if any exist
X_wealth = X_wealth.mask(X_wealth < 0, np.nan)

# Log transform skewed monetary/count variables
X_wealth_log = np.log1p(X_wealth)

# Median imputation
imputer = SimpleImputer(strategy="median")
X_wealth_imputed = imputer.fit_transform(X_wealth_log)

# Standardization
scaler = StandardScaler()
X_wealth_scaled = scaler.fit_transform(X_wealth_imputed)

# PCA with one component
pca = PCA(n_components=1, random_state=42)
wealth_index = pca.fit_transform(X_wealth_scaled).flatten()

# Add PCA index to dataset
women_ml["wealth_living_index"] = wealth_index

# Make sure higher value means better household economic condition
corr_check = np.corrcoef(
    women_ml["wealth_living_index"],
    np.log1p(women_ml["sec9_total_observed_value"].fillna(women_ml["sec9_total_observed_value"].median()))
)[0, 1]

if corr_check < 0:
    women_ml["wealth_living_index"] = -women_ml["wealth_living_index"]

print("\nPCA explained variance ratio:")
print(round(pca.explained_variance_ratio_[0], 4))

print("\nWealth/living index summary:")
print(women_ml["wealth_living_index"].describe().round(3))



# =====================================================
# Step 5.1G: Create Reduced Relevant Feature Set
# =====================================================

reduced_features = [

    # -------------------------
    # Demographic
    # -------------------------
    "age",
    "marital_status",
    "relation_head",

    # -------------------------
    # Education
    # -------------------------
    "literate_basic",
    "highest_class_completed",

    # -------------------------
    # Digital and financial access
    # -------------------------
    "has_mobile",
    "has_bank_account",
    "has_mfs_account",
    "internet_used",
    "has_computer",

    # -------------------------
    # Functional difficulty
    # -------------------------
    "any_functional_difficulty",

    # -------------------------
    # Geography
    # -------------------------
    "rural_urban",
    "division_code",
    "rmo_code",
    "domain16",

    # -------------------------
    # Household and food security
    # -------------------------
    "hh_MEMBER",
    "fies_raw_score",
    "wealth_living_index",

    # -------------------------
    # Housing and infrastructure
    # -------------------------
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

    # -------------------------
    # Employment structure
    # -------------------------
    "earn_sector",
    "earn_emp_status_ag",
    "earn_emp_status_nonag",
    "earn_org_type",
    "earn_ind_code",
    "earn_occ_code"
]

# Keep only variables available in dataset
reduced_features = [
    var for var in reduced_features
    if var in women_ml.columns
]

print("Number of reduced features:", len(reduced_features))
print(reduced_features)




# =====================================================
# Check reduced feature quality
# =====================================================

reduced_quality = []

for var in reduced_features:
    reduced_quality.append({
        "Variable": var,
        "Missing_percent": women_ml[var].isna().mean() * 100,
        "Unique_values": women_ml[var].nunique(dropna=True),
        "Data_type": str(women_ml[var].dtype)
    })

reduced_quality = pd.DataFrame(reduced_quality).sort_values(
    by="Unique_values",
    ascending=False
)

reduced_quality.round(2)




# =====================================================
# Step 5.2: Distribution of Low Work Intensity
# =====================================================

import matplotlib.pyplot as plt

# Frequency table
lowwi_freq = women_ml["low_work_intensity"].value_counts().sort_index()

lowwi_percent = (
    women_ml["low_work_intensity"]
    .value_counts(normalize=True)
    .sort_index() * 100
)

lowwi_distribution = pd.DataFrame({
    "Outcome": ["Not low work intensity", "Low work intensity"],
    "Frequency": lowwi_freq.values,
    "Percent": lowwi_percent.round(2).values
})

lowwi_distribution


# =====================================================
# Figure 1: Distribution of Low Work Intensity
# =====================================================

plt.figure(figsize=(6, 4))

plt.bar(
    lowwi_distribution["Outcome"],
    lowwi_distribution["Percent"]
)

plt.ylabel("Percent")
plt.xlabel("Work intensity group")
plt.title("Distribution of Low Work Intensity among Working Women")

plt.xticks(rotation=15)
plt.tight_layout()
plt.show()


# =====================================================
# Figure 2: Annual Working Hours Distribution
# =====================================================

plt.figure(figsize=(7, 4))

plt.hist(
    women_ml["earn_annual_hours"].dropna(),
    bins=30
)

plt.axvline(
    p25_hours,
    linestyle="--",
    linewidth=2,
    label=f"25th percentile = {p25_hours:.0f} hours"
)

plt.xlabel("Annual working hours")
plt.ylabel("Number of women")
plt.title("Distribution of Annual Working Hours among Working Women")
plt.legend()

plt.tight_layout()
plt.show()



# =====================================================
# Step 5.2B: Low Work Intensity by Key Background Variables
# =====================================================

from scipy.stats import chi2_contingency

target = "low_work_intensity"

key_categorical_vars = [
    "rural_urban",
    "division_code",
    "rmo_code",
    "marital_status",
    "relation_head",
    "literate_basic",
    "has_mobile",
    "has_bank_account",
    "has_mfs_account",
    "internet_used",
    "has_computer",
    "any_functional_difficulty",
    "transport_stop_halfkm",
    "regular_waste_collection",
    "slum_household",
    "earn_sector",
    "earn_emp_status_ag",
    "earn_emp_status_nonag",
    "earn_org_type"
]

# Keep only variables available in dataset
key_categorical_vars = [
    var for var in key_categorical_vars
    if var in women_ml.columns
]

lowwi_by_category_tables = []

for var in key_categorical_vars:
    
    temp = women_ml[[var, target]].copy()
    
    # Keep missing as a separate category for descriptive analysis
    temp[var] = temp[var].fillna("Missing/Not applicable")
    
    # Frequency table
    tab = pd.crosstab(temp[var], temp[target])
    
    # Make sure both outcome columns exist
    if 0 not in tab.columns:
        tab[0] = 0
    if 1 not in tab.columns:
        tab[1] = 0
    
    tab = tab[[0, 1]]
    tab["Total"] = tab[0] + tab[1]
    tab["Low_Work_Intensity_Percent"] = (tab[1] / tab["Total"] * 100).round(2)
    
    # Chi-square test
    try:
        chi2, p_value, dof, expected = chi2_contingency(tab[[0, 1]])
    except:
        p_value = np.nan
    
    tab = tab.reset_index()
    tab.columns = [
        "Category",
        "Not_Low_Work_Intensity_N",
        "Low_Work_Intensity_N",
        "Total_N",
        "Low_Work_Intensity_Percent"
    ]
    
    tab.insert(0, "Variable", var)
    tab["P_value"] = round(p_value, 4)
    
    lowwi_by_category_tables.append(tab)

table2b_lowwi_by_category = pd.concat(
    lowwi_by_category_tables,
    ignore_index=True
)

table2b_lowwi_by_category


# =====================================================
# Variables significantly associated with low work intensity
# =====================================================

significant_vars = (
    table2b_lowwi_by_category[table2b_lowwi_by_category["P_value"] < 0.05]
    ["Variable"]
    .drop_duplicates()
    .tolist()
)

print("Variables significantly associated with low work intensity:")
print(significant_vars)



# =====================================================
# Top categories with highest low work intensity
# Only categories with at least 30 observations
# =====================================================

top_lowwi_groups = table2b_lowwi_by_category[
    table2b_lowwi_by_category["Total_N"] >= 30
].sort_values(
    by="Low_Work_Intensity_Percent",
    ascending=False
)

top_lowwi_groups.head(20)


# =====================================================
# Top groups with highest low work intensity
# Only categories with at least 30 observations
# =====================================================

top_lowwi_groups = table2b_lowwi_by_category[
    table2b_lowwi_by_category["Total_N"] >= 30
].sort_values(
    by="Low_Work_Intensity_Percent",
    ascending=False
)

top_lowwi_groups.head(25)




top_lowwi_groups.to_excel(
    "Top_Low_Work_Intensity_Groups.xlsx",
    index=False
)

print("Top low work intensity groups exported successfully.")


# =====================================================
# Clean p-value display for paper
# =====================================================

table2b_paper = table2b_lowwi_by_category.copy()

table2b_paper["P_value_display"] = table2b_paper["P_value"].apply(
    lambda x: "<0.001" if x < 0.001 else round(x, 3)
)

table2b_paper.head()




# =====================================================
# Step 5.2C: Low Work Intensity by Continuous Variables
# =====================================================

from scipy.stats import mannwhitneyu

continuous_key_vars = [
    "age",
    "highest_class_completed",
    "hh_MEMBER",
    "fies_raw_score",
    "wealth_living_index",
    "electricity_hours_24h",
    "transport_stop_time_min",
    "earn_annual_hours"
]

continuous_key_vars = [
    var for var in continuous_key_vars
    if var in women_ml.columns
]

continuous_lowwi_table = []

for var in continuous_key_vars:
    
    group0 = women_ml.loc[
        women_ml["low_work_intensity"] == 0, var
    ].dropna()
    
    group1 = women_ml.loc[
        women_ml["low_work_intensity"] == 1, var
    ].dropna()
    
    # Mann-Whitney U test because many variables may be non-normal
    if len(group0) > 0 and len(group1) > 0:
        stat, p_value = mannwhitneyu(
            group0,
            group1,
            alternative="two-sided"
        )
    else:
        p_value = np.nan
    
    continuous_lowwi_table.append({
        "Variable": var,
        
        "Not_Low_N": group0.count(),
        "Not_Low_Mean": group0.mean(),
        "Not_Low_SD": group0.std(),
        "Not_Low_Median": group0.median(),
        
        "Low_N": group1.count(),
        "Low_Mean": group1.mean(),
        "Low_SD": group1.std(),
        "Low_Median": group1.median(),
        
        "Mean_Difference_Low_minus_NotLow": group1.mean() - group0.mean(),
        "P_value": p_value
    })

table2c_continuous_lowwi = pd.DataFrame(continuous_lowwi_table)

table2c_continuous_lowwi = table2c_continuous_lowwi.round(3)

table2c_continuous_lowwi


# =====================================================
# Format p-values for paper
# =====================================================

table2c_paper = table2c_continuous_lowwi.copy()

table2c_paper["P_value_display"] = table2c_paper["P_value"].apply(
    lambda x: "<0.001" if x < 0.001 else round(x, 3)
)

table2c_paper




# =====================================================
# Step 5.3A Revised: Inspect High-Category Variables
# =====================================================

high_category_vars = [
    "earn_occ_code",
    "earn_ind_code",
    "domain16",
    "relation_head",
    "drinking_water_source",
    "toilet_type"
]

high_category_vars = [
    var for var in high_category_vars
    if var in women_ml.columns
]

for var in high_category_vars:
    print("\n==============================")
    print("Variable:", var)
    print("Unique categories:", women_ml[var].nunique(dropna=False))
    print(women_ml[var].value_counts(dropna=False).head(20))
    
    
    
    
# =====================================================
# Step 5.3B: Group Rare Categories Before Boruta
# =====================================================

women_boruta = women_ml.copy()

def group_rare_categories(df, var, min_count=30, rare_code=-88, missing_code=-99):
    """
    Groups categories with frequency below min_count into rare_code.
    Missing values are coded separately as missing_code.
    """
    if var in df.columns:
        temp = df[var].copy()
        
        # Missing category
        temp = temp.fillna(missing_code)
        
        # Frequency count
        counts = temp.value_counts()
        
        # Identify rare categories, excluding missing code
        rare_categories = counts[
            (counts < min_count) & (counts.index != missing_code)
        ].index
        
        # Replace rare categories
        temp = temp.replace(list(rare_categories), rare_code)
        
        df[var + "_grp"] = temp
        
        print("\nVariable:", var)
        print("Original categories:", df[var].nunique(dropna=False))
        print("Grouped categories:", df[var + '_grp'].nunique(dropna=False))
        print("Rare categories grouped:", len(rare_categories))

# Apply rare grouping only to high-cardinality categorical variables
group_rare_categories(women_boruta, "earn_occ_code", min_count=30)
group_rare_categories(women_boruta, "earn_ind_code", min_count=30)



# =====================================================
# Step 5.3C: Update Reduced Feature Set for Boruta
# =====================================================

reduced_features_boruta = reduced_features.copy()

# Remove original high-cardinality variables
reduced_features_boruta = [
    var for var in reduced_features_boruta
    if var not in ["earn_occ_code", "earn_ind_code"]
]

# Add grouped versions
for var in ["earn_occ_code_grp", "earn_ind_code_grp"]:
    if var in women_boruta.columns:
        reduced_features_boruta.append(var)

print("Final number of reduced Boruta features:", len(reduced_features_boruta))
print(reduced_features_boruta)



# =====================================================
# Check Grouped Category Counts
# =====================================================

for var in ["earn_occ_code_grp", "earn_ind_code_grp"]:
    print("\n==============================")
    print("Variable:", var)
    print("Unique categories:", women_boruta[var].nunique(dropna=False))
    print(women_boruta[var].value_counts(dropna=False).sort_index())