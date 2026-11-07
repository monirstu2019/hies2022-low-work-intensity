# %% STEP 1.45: RF FROZEN-BORUTA EMPLOYMENT-PREDICTOR ABLATION
# Run standalone in a restarted Spyder kernel. Uses only saved files.
# IMPORTANT: Exploratory controlled ablation, NOT fully re-tuned nested CV.

from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
root = Path(os.environ.get("HIES_OUTPUT_DIR", str(PROJECT_ROOT / "outputs")))
fp = "7244d907321a2e433a44"
folder_140 = root / f"Step1_40_Remaining_Tuning_{fp}"
folder_141 = root / f"Step1_41_Nested_Outer_Predictions_{fp}"
result_dir = root / f"Step1_45_FrozenFeature_RF_Ablation_{fp}"
result_dir.mkdir(parents=True, exist_ok=True)

best_path = folder_140 / "Step1_40_Best_Params_By_Outer_And_Model.csv"
oof_path = folder_141 / "Step1_41_All_Six_Model_OOF_Predictions.csv"
stata_path = Path(os.environ.get("HIES_MERGED_DATA", str(PROJECT_ROOT / "restricted_data" / "final_hies_women_lfp_earnings_ml_ready.dta")))
for path in (best_path, oof_path, stata_path):
    assert path.is_file(), f"Missing input: {path}"

best = pd.read_csv(best_path)
oof = pd.read_csv(oof_path)
rf_original = oof.loc[oof["Model"] == "Random Forest"].copy()
assert len(rf_original) == 2110
assert rf_original["original_row_index"].is_unique
assert rf_original["Outer_Fold"].nunique() == 5
assert (rf_original["Fingerprint"] == fp).all()
rf_original = rf_original.set_index("original_row_index", verify_integrity=True)

# Raw source is read ONLY to create binary missingness indicators.
missing_fields = ["earn_emp_status_nonag", "earn_org_type", "earn_emp_status_ag"]
raw = pd.read_stata(stata_path, columns=missing_fields, convert_categoricals=False)
assert len(raw) == 62407
miss_flags = np.column_stack([
    (raw[col].isna() | raw[col].eq(-99)).to_numpy(dtype=float)
    for col in missing_fields
])

# Two ablations are defined BEFORE examining outer results.
REMOVE_TOP2 = {"earn_emp_status_nonag", "earn_org_type_grp"}
REMOVE_ALL3 = REMOVE_TOP2 | {"earn_emp_status_ag"}
SCENARIOS = [
    "Full selected features (controlled refit)",
    "Exclude top 2 employment predictors",
    "Exclude all 3 employment predictors",
    "Three missingness indicators only"
]

# Exactly the SAME fixed forest specification in every scenario.
# The original Step 1.41 model might have a different fixed tree count/seed;
# the controlled full refit is therefore the appropriate comparator here.
TREES = 400
SEED = 20261030


def feature_parent(encoded_name, groups):
    # Example: cat__earn_org_type_grp_-99.0 -> earn_org_type_grp
    tail = str(encoded_name).split("__", 1)[-1]
    matches = [g for g in groups if tail == g or tail.startswith(g + "_")]
    if matches:
        return max(matches, key=len)
    # The PCA output can be called 'pca0' after ColumnTransformer.
    if "wealth_living_index" in groups and (
        "pca" in encoded_name.lower() or "wealth" in encoded_name.lower()
    ):
        return "wealth_living_index"
    raise ValueError(
        f"Cannot map encoded feature '{encoded_name}' to any of {groups}. "
        "Stop rather than silently drop a column."
    )


def score(rows):
    y = rows["Actual"].to_numpy(dtype=int)
    p = rows["Predicted_Probability"].to_numpy(dtype=float)
    w = rows["population_weight"].to_numpy(dtype=float)
    return {
        "N": len(rows),
        "Weighted_AP": float(average_precision_score(y, p, sample_weight=w)),
        "Weighted_ROC_AUC": float(roc_auc_score(y, p, sample_weight=w)),
        "Weighted_Brier": float(np.average((p - y) ** 2, weights=w)),
        "Mean_Predicted_Risk": float(np.average(p, weights=w))
    }


out = []
audit = []
print("========== STEP 1.45: CONTROLLED RF ABLATION ==========")
print("Same outer PSU splits, same training-only fitted feature matrices.")
print("Feature sets and RF tree count held fixed; no retuning/re-Boruta.")

for outer in range(1, 6):
    cache = folder_141 / "outer_train_fit_cache" / f"Outer{outer}_Preprocessing_Boruta.npz"
    assert cache.is_file(), cache
    with np.load(cache, allow_pickle=False) as z:
        assert z["fingerprint"].item() == fp
        tr_idx = z["train_idx"].copy()
        val_idx = z["val_idx"].copy()
        X_train = z["X_train"].copy()
        X_val = z["X_validation"].copy()
        groups = z["selected_groups"].tolist()
        names = z["feature_names"].tolist()

    assert len(set(tr_idx).intersection(val_idx)) == 0
    assert len(tr_idx) + len(val_idx) == 2110
    assert X_train.shape == (len(tr_idx), len(names))
    assert X_val.shape == (len(val_idx), len(names))
    assert np.isfinite(X_train).all() and np.isfinite(X_val).all()
    assert set(tr_idx).union(val_idx) == set(rf_original.index)

    train_rows = rf_original.loc[tr_idx]
    val_rows = rf_original.loc[val_idx]
    assert (train_rows["Outer_Fold"] != outer).all()
    assert (val_rows["Outer_Fold"] == outer).all()
    train_psus = set(zip(train_rows["stratum16"], train_rows["psu"]))
    val_psus = set(zip(val_rows["stratum16"], val_rows["psu"]))
    assert not train_psus.intersection(val_psus)

    y_tr = train_rows["Actual"].to_numpy(dtype=int)
    y_va = val_rows["Actual"].to_numpy(dtype=int)
    w_tr = train_rows["population_weight"].to_numpy(dtype=float)
    w_va = val_rows["population_weight"].to_numpy(dtype=float)
    assert set(y_tr) == {0, 1} and set(y_va) == {0, 1}
    assert (w_tr > 0).all() and (w_va > 0).all()

    parent_names = [feature_parent(n, groups) for n in names]
    assert set(parent_names) == set(groups)

    match = best.loc[
        (best["Outer_Fold"] == outer) & (best["Model"] == "Random Forest")
    ]
    assert len(match) == 1
    params = json.loads(match.iloc[0]["Parameters_JSON"])
    assert set(params) == {"max_depth", "max_features", "min_samples_leaf"}

    print(f"\nOuter {outer}: train={len(tr_idx)}, validation={len(val_idx)}, "
          f"Boruta groups={groups}", flush=True)

    for scenario in SCENARIOS:
        if scenario == SCENARIOS[0]:
            Xtr, Xva = X_train, X_val
        elif scenario == SCENARIOS[1]:
            retain = [i for i, g in enumerate(parent_names) if g not in REMOVE_TOP2]
            Xtr, Xva = X_train[:, retain], X_val[:, retain]
        elif scenario == SCENARIOS[2]:
            retain = [i for i, g in enumerate(parent_names) if g not in REMOVE_ALL3]
            Xtr, Xva = X_train[:, retain], X_val[:, retain]
        else:
            # Deliberately fit only 3 survey-routing missingness flags.
            Xtr, Xva = miss_flags[tr_idx], miss_flags[val_idx]

        assert Xtr.shape[1] > 0 and Xtr.shape[1] == Xva.shape[1]
        forest = RandomForestClassifier(
            n_estimators=TREES, random_state=SEED, n_jobs=-1, **params
        )
        forest.fit(Xtr, y_tr, sample_weight=w_tr / w_tr.mean())
        prob = forest.predict_proba(Xva)[:, 1]
        assert prob.shape == (len(val_idx),)
        assert np.isfinite(prob).all() and ((prob >= 0) & (prob <= 1)).all()
        out.append(pd.DataFrame({
            "Scenario": scenario, "Outer_Fold": outer,
            "original_row_index": val_idx, "Actual": y_va,
            "population_weight": w_va,
            "stratum16": val_rows["stratum16"].to_numpy(),
            "psu": val_rows["psu"].to_numpy(),
            "Predicted_Probability": prob
        }))
        audit.append({
            "Outer_Fold": outer, "Scenario": scenario,
            "Retained_Columns": Xtr.shape[1],
            "Train_N": len(tr_idx), "Validation_N": len(val_idx),
            "PSU_Overlap": 0
        })
        print(f"  {scenario}: {Xtr.shape[1]} columns; predictions saved in memory.",
              flush=True)

preds = pd.concat(out, ignore_index=True)
assert len(preds) == 2110 * 4
assert not preds.duplicated(["Scenario", "original_row_index"]).any()
assert preds.groupby("Scenario").size().eq(2110).all()

summary = pd.DataFrame([
    {"Scenario": s, **score(preds.loc[preds["Scenario"] == s])}
    for s in SCENARIOS
])
base = summary.iloc[0]
for metric in ("Weighted_AP", "Weighted_ROC_AUC", "Weighted_Brier"):
    summary["Difference_vs_Controlled_Full_" + metric] = summary[metric] - base[metric]

original_metrics = score(rf_original.reset_index())
refit_full = preds.loc[preds["Scenario"] == SCENARIOS[0]].set_index(
    "original_row_index"
).loc[rf_original.index]
mean_abs_prob_diff = float(np.average(
    np.abs(refit_full["Predicted_Probability"].to_numpy()
           - rf_original["Predicted_Probability"].to_numpy()),
    weights=rf_original["population_weight"].to_numpy()
))

audit_df = pd.DataFrame(audit)
preds.to_csv(result_dir / "Step1_45_All_Controlled_OOF_Predictions.csv", index=False)
summary.to_csv(result_dir / "Step1_45_Controlled_Ablation_Performance.csv", index=False)
audit_df.to_csv(result_dir / "Step1_45_Outer_Fold_Feature_Audit.csv", index=False)

print("\n========== ORIGINAL STEP 1.41 RANDOM FOREST ==========")
print({k: round(original_metrics[k], 5)
       for k in ("Weighted_AP", "Weighted_ROC_AUC", "Weighted_Brier")})
print("Weighted mean absolute probability difference (controlled-full vs Step 1.41):",
      round(mean_abs_prob_diff, 5))
print("\n========== CONTROLLED ABLATION RESULTS ==========")
print(summary.round(5).to_string(index=False))
print("\n========== STEP 1.45 VERIFICATION ==========")
print("Nested outer validation splits: 5; PSU overlap: 0")
print("Held-out prediction rows: 8440 (2110 x 4 scenarios)")
print("Reused training-only preprocessing and Boruta masks: YES")
print("Retuned or reselected features after ablation: NO")
print("Interpretation: exploratory frozen-feature sensitivity, not fully nested ablation")
print("Files saved:", result_dir)
print("STEP 1.45 VERIFICATION SUCCESSFUL.")
