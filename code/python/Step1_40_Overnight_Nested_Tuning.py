# %% STEP 1.40: COMPLETE REMAINING NESTED-CV TUNING (ALL SIX MODELS)
# Paste as ONE Spyder cell into the running project script.
# Requires project variables created in Steps 1.25-1.39.

import os
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import average_precision_score, roc_auc_score

# ------------------------------------------------------------------
# 1. Preconditions: never silently change data, folds or model grids
# ------------------------------------------------------------------
assert len(women_analysis) == 2110
assert X_raw.shape == (2110, 48)
assert len(cv_manifest) == 2110
assert len(inner_cv_manifest) == 8440
assert len(boruta_all_details) == 525
assert len(boruta_all_folds) == 15
assert len(model_feature_names) == 35
assert len(tuned_model_templates) == 6
assert len(tuning_grids) == 6
assert set(tuning_grids) == set(tuned_model_templates)

model_order_140 = [
    "Logistic Regression", "Support Vector Machine",
    "Random Forest", "Extra Trees", "Gradient Boosting", "AdaBoost"
]
assert set(model_order_140) == set(tuning_grids)

candidate_lists_140 = {
    m: list(ParameterGrid(tuning_grids[m])) for m in model_order_140
}
assert {m: len(v) for m, v in candidate_lists_140.items()} == {
    "Logistic Regression": 8, "Support Vector Machine": 9,
    "Random Forest": 12, "Extra Trees": 8,
    "Gradient Boosting": 8, "AdaBoost": 9
}

# Make sure AdaBoost SAMME is retained, not deprecated SAMME.R.
assert tuned_model_templates["AdaBoost"].get_params()["algorithm"] == "SAMME"

# ------------------------------------------------------------------
# 2. Fingerprint the inputs so checkpoints cannot silently mix data
# ------------------------------------------------------------------
h_140 = hashlib.sha256()
for frame_140 in (
    X_raw,
    women_analysis[["low_work_intensity", "population_weight"]],
    cv_manifest,
    inner_cv_manifest
):
    digest_values = pd.util.hash_pandas_object(
        frame_140, index=True
    ).to_numpy(dtype=np.uint64)
    h_140.update(digest_values.tobytes())
    h_140.update(json.dumps(
        list(map(str, frame_140.columns))
    ).encode("utf-8"))

h_140.update(json.dumps(
    tuning_grids, sort_keys=True
).encode("utf-8"))
h_140.update(json.dumps(
    list(model_feature_names)
).encode("utf-8"))
h_140.update(json.dumps(
    list(wealth_pca_inputs)
).encode("utf-8"))
h_140.update(repr(preprocessor_template).encode("utf-8"))
h_140.update(repr(boruta_group_template).encode("utf-8"))

fingerprint_140 = h_140.hexdigest()[:20]
run_version_140 = "HIES22_NestedV1_Boruta160_Iter40"

run_dir_140 = os.path.join(
    output_dir, "Step1_40_Remaining_Tuning_" + fingerprint_140
)
cache_dir_140 = os.path.join(run_dir_140, "inner_fold_matrices")
os.makedirs(cache_dir_140, exist_ok=True)

checkpoint_140 = os.path.join(
    run_dir_140, "tuning_checkpoint.csv"
)

# Preexisting Outer 1 results for five completed classifiers.
previous_files_140 = {
    "Logistic Regression": "Step1_35_LR_Outer1_Inner_Scores.csv",
    "Support Vector Machine": "Step1_36_SVM_Outer1_Inner_Scores.csv",
    "Random Forest": "Step1_37_RF_Outer1_Inner_Scores.csv",
    "Extra Trees": "Step1_38_ET_Outer1_Inner_Scores.csv",
    "Gradient Boosting": "Step1_39_GB_Outer1_Inner_Scores.csv",
}
for m_140, filename_140 in previous_files_140.items():
    assert os.path.isfile(os.path.join(output_dir, filename_140)), (
        "Missing previous tuning output for " + m_140
    )

# ------------------------------------------------------------------
# 3. Load restartable checkpoints and validate keys
# ------------------------------------------------------------------
required_cols_140 = [
    "Run_Version", "Fingerprint", "Outer_Fold", "Inner_Fold",
    "Model", "Candidate_ID", "Parameters_JSON",
    "Weighted_AP", "Weighted_ROC_AUC", "Training_N",
    "Validation_N", "Retained_Columns"
]
if os.path.exists(checkpoint_140):
    done_140 = pd.read_csv(checkpoint_140)
    assert set(required_cols_140).issubset(done_140.columns)
    assert (done_140.Run_Version == run_version_140).all()
    assert (done_140.Fingerprint == fingerprint_140).all()
    assert not done_140.duplicated(
        ["Outer_Fold", "Inner_Fold", "Model", "Candidate_ID"]
    ).any()
    print("Recovered", len(done_140), "candidate fits from checkpoint.", flush=True)
else:
    done_140 = pd.DataFrame(columns=required_cols_140)

completed_keys_140 = set()
for record_140 in done_140.itertuples(index=False):
    key_140 = (
        int(record_140.Outer_Fold), int(record_140.Inner_Fold),
        record_140.Model, int(record_140.Candidate_ID)
    )
    assert record_140.Model in candidate_lists_140
    assert 1 <= key_140[0] <= 5
    assert 1 <= key_140[1] <= 3
    assert (key_140[0] > 1) or (record_140.Model == "AdaBoost")
    assert 1 <= key_140[3] <= len(candidate_lists_140[record_140.Model])
    expected_json_140 = json.dumps(
        candidate_lists_140[record_140.Model][key_140[3]-1],
        sort_keys=True
    )
    assert record_140.Parameters_JSON == expected_json_140
    assert np.isfinite(record_140.Weighted_AP)
    assert np.isfinite(record_140.Weighted_ROC_AUC)
    completed_keys_140.add(key_140)

expected_new_total_140 = 27 + 4 * 3 * 54
assert expected_new_total_140 == 675

# ------------------------------------------------------------------
# 4. Atomic checkpoint writer (safe if execution is interrupted)
# ------------------------------------------------------------------
def save_checkpoint_140(df, filename):
    tmp_name = filename + ".tmp"
    df.to_csv(tmp_name, index=False, float_format="%.17g")
    os.replace(tmp_name, filename)

# ------------------------------------------------------------------
# 5. Inner-fold cache: fitted on training rows ONLY
# ------------------------------------------------------------------
def load_or_create_inner_matrices_140(outer0, inner0, train_idx, val_idx,
                                      y_tr, w_tr):
    outer_num, inner_num = outer0 + 1, inner0 + 1
    if outer0 == 0:
        # Reuse the already audited caches from Step 1.35.
        path = os.path.join(
            cache_folder_135,
            f"Outer1_Inner{inner_num}_Boruta_Matrices.npz"
        )
    else:
        path = os.path.join(
            cache_dir_140,
            f"Outer{outer_num}_Inner{inner_num}_Boruta.npz"
        )

    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as cache:
            assert cache["version"].item() == (
                "HIES2022_Boruta_InnerCache_v1" if outer0 == 0
                else run_version_140
            )
            if outer0 != 0:
                assert cache["fingerprint"].item() == fingerprint_140
            assert np.array_equal(cache["train_idx"], train_idx)
            assert np.array_equal(cache["val_idx"], val_idx)
            X_tr = cache["X_train"].copy()
            X_val = cache["X_validation"].copy()
            groups = cache["selected_groups"].tolist()
            names = cache["feature_names"].tolist()
            print("  Cached preprocessing/Boruta matrices loaded.", flush=True)
    else:
        assert outer0 != 0, "Step 1.35 Outer1 cache is missing."
        selector = clone(boruta_group_template)
        selector.fit(
            X_raw.loc[train_idx], y_tr,
            sample_weight=w_tr
        )
        X_tr = selector.transform(X_raw.loc[train_idx])
        X_val = selector.transform(X_raw.loc[val_idx])
        groups = list(selector.selected_groups_)
        names = list(selector.get_feature_names_out())

        # Write complete compressed file, then rename atomically.
        tmp_path = path + ".tmp"
        with open(tmp_path, "wb") as handle:
            np.savez_compressed(
                handle,
                version=np.array(run_version_140),
                fingerprint=np.array(fingerprint_140),
                train_idx=train_idx,
                val_idx=val_idx,
                X_train=X_tr,
                X_validation=X_val,
                selected_groups=np.asarray(groups, dtype=str),
                feature_names=np.asarray(names, dtype=str)
            )
        os.replace(tmp_path, path)
        print("  Training-only preprocessing/Boruta cache saved.", flush=True)

    # Compare newly fitted selection with the frozen Step 1.32 audit.
    recorded = boruta_all_details.loc[
        (boruta_all_details["Outer_Fold"] == outer_num)
        & (boruta_all_details["Inner_Fold"] == inner_num)
    ]
    assert len(recorded) == 35
    expected_groups = set(recorded.loc[
        recorded["Selected"].astype(bool), "Original_Predictor"
    ])
    assert set(groups) == expected_groups, (
        f"Boruta reproducibility mismatch at outer {outer_num}, inner {inner_num}"
    )
    expected_columns = int(boruta_all_folds.loc[
        (boruta_all_folds["Outer_Fold"] == outer_num)
        & (boruta_all_folds["Inner_Fold"] == inner_num),
        "Retained_Model_Columns"
    ].iloc[0])
    assert X_tr.shape == (len(train_idx), expected_columns)
    assert X_val.shape == (len(val_idx), expected_columns)
    assert len(names) == expected_columns
    assert np.isfinite(X_tr).all() and np.isfinite(X_val).all()
    return X_tr, X_val, groups

# ------------------------------------------------------------------
# 6. Train and score ALL remaining models, saving every candidate
# ------------------------------------------------------------------
print("\n========== STEP 1.40: REMAINING NESTED TUNING ==========", flush=True)
print("Run fingerprint:", fingerprint_140, flush=True)
print("Results folder:", run_dir_140, flush=True)
print("Remaining fits in plan:", expected_new_total_140, flush=True)
print("Already checkpointed:", len(completed_keys_140), flush=True)
print("No outer-validation rows will be used.", flush=True)

for outer0_140 in range(5):
    for inner0_140 in range(3):
        outer_num_140 = outer0_140 + 1
        inner_num_140 = inner0_140 + 1
        relevant_models_140 = (
            ["AdaBoost"] if outer0_140 == 0 else model_order_140
        )
        all_keys_for_fold_140 = [
            (outer_num_140, inner_num_140, m, c)
            for m in relevant_models_140
            for c in range(1, len(candidate_lists_140[m])+1)
        ]
        if all(k in completed_keys_140 for k in all_keys_for_fold_140):
            print(f"Outer {outer_num_140} / Inner {inner_num_140}: complete; skipped.", flush=True)
            continue

        print("\n" + "="*64, flush=True)
        print(f"OUTER {outer_num_140} | INNER {inner_num_140}", flush=True)
        print("="*64, flush=True)
        manifest = inner_cv_manifest.loc[
            inner_cv_manifest["outer_fold"] == outer0_140
        ]
        train_rows = manifest.loc[manifest["inner_fold"] != inner0_140]
        val_rows = manifest.loc[manifest["inner_fold"] == inner0_140]
        train_idx = train_rows["original_row_index"].to_numpy()
        val_idx = val_rows["original_row_index"].to_numpy()
        assert len(set(train_idx) & set(val_idx)) == 0

        # Check PSU isolation against inner and outer validation sets.
        tr_psus = set(zip(train_rows["stratum16"], train_rows["psu"]))
        va_psus = set(zip(val_rows["stratum16"], val_rows["psu"]))
        outer_val = cv_manifest.loc[
            cv_manifest["outer_fold"] == outer0_140
        ]
        outer_psus = set(zip(outer_val["stratum16"], outer_val["psu"]))
        assert not (tr_psus & va_psus)
        assert not (tr_psus & outer_psus)
        assert not (va_psus & outer_psus)
        assert val_rows["stratum16"].nunique() == 16
        assert len(set(train_idx) & set(outer_val.index)) == 0
        assert len(set(val_idx) & set(outer_val.index)) == 0

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
        assert np.isfinite(w_tr).all() and (w_tr > 0).all()
        assert np.isfinite(w_val).all() and (w_val > 0).all()
        assert set(np.unique(y_tr)) == {0, 1}
        assert set(np.unique(y_val)) == {0, 1}

        X_tr, X_val, selected_groups = load_or_create_inner_matrices_140(
            outer0_140, inner0_140, train_idx, val_idx, y_tr, w_tr
        )
        fit_w = w_tr / w_tr.mean()
        print(f"  Train N={len(y_tr)}, validation N={len(y_val)}, "
              f"model columns={X_tr.shape[1]}", flush=True)
        print("  Selected original predictors:", selected_groups, flush=True)

        for model_name in relevant_models_140:
            params_list = candidate_lists_140[model_name]
            needed = [
                (i, params) for i, params in enumerate(params_list, start=1)
                if (outer_num_140, inner_num_140, model_name, i)
                not in completed_keys_140
            ]
            if not needed:
                print("  ", model_name, ": already completed.", flush=True)
                continue
            print(f"  {model_name}: {len(needed)} fits", flush=True)

            for candidate_id, params in needed:
                estimator = clone(tuned_model_templates[model_name])
                estimator.set_params(**params)
                if model_name == "Support Vector Machine":
                    # Ranking metrics do not require probability calibration.
                    estimator.set_params(probability=False)
                estimator.fit(X_tr, y_tr, sample_weight=fit_w)
                assert np.array_equal(estimator.classes_, [0, 1])

                if model_name == "Support Vector Machine":
                    score = estimator.decision_function(X_val)
                else:
                    score = estimator.predict_proba(X_val)[:, 1]
                assert score.shape == (len(y_val),)
                assert np.isfinite(score).all()

                ap = float(average_precision_score(
                    y_val, score, sample_weight=w_val
                ))
                auc = float(roc_auc_score(
                    y_val, score, sample_weight=w_val
                ))
                assert 0.0 <= ap <= 1.0
                assert 0.0 <= auc <= 1.0

                row = {
                    "Run_Version": run_version_140,
                    "Fingerprint": fingerprint_140,
                    "Outer_Fold": outer_num_140,
                    "Inner_Fold": inner_num_140,
                    "Model": model_name,
                    "Candidate_ID": candidate_id,
                    "Parameters_JSON": json.dumps(params, sort_keys=True),
                    "Weighted_AP": ap,
                    "Weighted_ROC_AUC": auc,
                    "Training_N": len(y_tr),
                    "Validation_N": len(y_val),
                    "Retained_Columns": X_tr.shape[1]
                }
                done_140 = pd.concat(
                    [done_140, pd.DataFrame([row])], ignore_index=True
                )
                save_checkpoint_140(done_140, checkpoint_140)
                completed_keys_140.add(
                    (outer_num_140, inner_num_140, model_name, candidate_id)
                )
            print(f"    Completed {model_name}; checkpoint saved.", flush=True)

# ------------------------------------------------------------------
# 7. Verify complete 675 new fits, then merge prior 135 fits
# ------------------------------------------------------------------
assert len(done_140) == 675
assert len(completed_keys_140) == 675
assert not done_140.duplicated(
    ["Outer_Fold", "Inner_Fold", "Model", "Candidate_ID"]
).any()
assert done_140[["Weighted_AP", "Weighted_ROC_AUC"]].notna().all().all()

old_score_frames_140 = []
for model_name, filename in previous_files_140.items():
    old = pd.read_csv(os.path.join(output_dir, filename))
    assert (old["Outer_Fold"] == 1).all()
    assert (old["Model"] == model_name).all()
    assert len(old) == 3 * len(candidate_lists_140[model_name])
    old_score_frames_140.append(old[
        ["Outer_Fold", "Inner_Fold", "Model", "Candidate_ID",
         "Parameters_JSON", "Weighted_AP", "Weighted_ROC_AUC"]
    ])

all_scores_140 = pd.concat(
    old_score_frames_140 + [done_140[
        ["Outer_Fold", "Inner_Fold", "Model", "Candidate_ID",
         "Parameters_JSON", "Weighted_AP", "Weighted_ROC_AUC"]
    ]], ignore_index=True
)
assert len(all_scores_140) == 810
assert not all_scores_140.duplicated(
    ["Outer_Fold", "Inner_Fold", "Model", "Candidate_ID"]
).any()

# ------------------------------------------------------------------
# 8. Rank candidates WITHIN each outer fold and model (NOT across models)
# ------------------------------------------------------------------
ranked_140 = (
    all_scores_140
    .groupby(["Outer_Fold", "Model", "Candidate_ID", "Parameters_JSON"],
             as_index=False)
    .agg(
        Inner_Folds=("Inner_Fold", "nunique"),
        Mean_Weighted_AP=("Weighted_AP", "mean"),
        SD_Weighted_AP=("Weighted_AP", "std"),
        Mean_Weighted_ROC_AUC=("Weighted_ROC_AUC", "mean")
    )
)
assert len(ranked_140) == 270
assert (ranked_140["Inner_Folds"] == 3).all()
ranked_140 = ranked_140.sort_values(
    ["Outer_Fold", "Model", "Mean_Weighted_AP",
     "Mean_Weighted_ROC_AUC", "Candidate_ID"],
    ascending=[True, True, False, False, True]
).reset_index(drop=True)
ranked_140.insert(
    0, "Rank", ranked_140.groupby(["Outer_Fold", "Model"]).cumcount()+1
)
best_140 = ranked_140.loc[ranked_140["Rank"] == 1].copy()
assert len(best_140) == 30

# Save and print compact results.
all_scores_140.to_csv(os.path.join(
    run_dir_140, "Step1_40_All_810_Inner_CV_Fits.csv"
), index=False)
ranked_140.to_csv(os.path.join(
    run_dir_140, "Step1_40_All_Outer_Model_Candidate_Rankings.csv"
), index=False)
best_140.to_csv(os.path.join(
    run_dir_140, "Step1_40_Best_Params_By_Outer_And_Model.csv"
), index=False)

print("\n========== STEP 1.40 FINAL VERIFICATION ==========", flush=True)
print("New tuning fits:", len(done_140))
print("Previous tuning fits:", sum(len(f) for f in old_score_frames_140))
print("Total tuning fits:", len(all_scores_140))
print("Total candidate rankings:", len(ranked_140))
print("Best configurations (5 outer folds x 6 models):", len(best_140))
print("\nBEST PARAMS PER OUTER FOLD AND MODEL:")
print(best_140[[
    "Outer_Fold", "Model", "Parameters_JSON",
    "Mean_Weighted_AP", "Mean_Weighted_ROC_AUC"
]].round(5).to_string(index=False))
print("\nCheckpoint/output folder:", run_dir_140)
print("Outer-validation observations scored: 0")
print("STEP 1.40 VERIFICATION SUCCESSFUL.", flush=True)
