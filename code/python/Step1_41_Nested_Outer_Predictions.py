# %% STEP 1.41: NESTED-CV OUTER-FOLD REFITTING AND HELD-OUT PREDICTIONS
# Run in the SAME Spyder console as Step 1.40, after all earlier steps.
# Written for the frozen HIES2022 analysis (5 outer folds x 6 models).
# All preprocessing, wealth PCA, Boruta, hyperparameter tuning, and SVM
# calibration are learned without accessing the corresponding outer holdout.

import os
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

# -------------------------------------------------------------------
# 1. Check frozen run: fail early if datasets, tuning or folds changed
# -------------------------------------------------------------------
assert len(women_analysis) == 2110
assert X_raw.shape == (2110, 48)
assert len(cv_manifest) == 2110
assert len(inner_cv_manifest) == 8440
assert len(best_140) == 30 and len(all_scores_140) == 810
assert len(model_order_140) == 6 and len(model_feature_names) == 35
assert best_140.groupby(['Outer_Fold', 'Model']).size().eq(1).all()
assert os.path.isdir(run_dir_140)
assert 'fingerprint_140' in globals()
assert 'load_or_create_inner_matrices_140' in globals()
assert tuned_model_templates['AdaBoost'].algorithm == 'SAMME'

run_version_141 = 'HIES2022_OuterOOF_v1_PSU_SVMPlatt'
run_dir_141 = os.path.join(
    output_dir, f'Step1_41_Nested_Outer_Predictions_{fingerprint_140}'
)
outer_cache_dir_141 = os.path.join(run_dir_141, 'outer_train_fit_cache')
model_dir_141 = os.path.join(run_dir_141, 'model_prediction_checkpoints')
os.makedirs(outer_cache_dir_141, exist_ok=True)
os.makedirs(model_dir_141, exist_ok=True)

# Canonical output order, no optimization or selection using outer scores.
models_141 = list(model_order_140)
assert set(models_141) == set(tuned_model_templates)

# -------------------------------------------------------------------
# 2. Reusable atomic output utilities
# -------------------------------------------------------------------
def atomic_csv_141(df, path):
    tmp = path + '.tmp'
    df.to_csv(tmp, index=False, float_format='%.17g')
    os.replace(tmp, path)


def atomic_npz_141(path, **arrays):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def expected_params_141(outer_number, model_name):
    match = best_140.loc[
        (best_140['Outer_Fold'] == outer_number)
        & (best_140['Model'] == model_name)
    ]
    assert len(match) == 1
    params_json = str(match.iloc[0]['Parameters_JSON'])
    params = json.loads(params_json)
    assert params in candidate_lists_140[model_name]
    return params, params_json


# -------------------------------------------------------------------
# 3. Calibration for SVM: weighted Platt regression trained only on
#    outer-training OOF scores from three PSU-disjoint INNER folds.
#    Each inner-fold preprocessing / Boruta was fitted on inner-train.
#    This is a training-only probability mapping, not test calibration.
# -------------------------------------------------------------------
def fit_svm_platt_141(outer0, params, outer_training_indices):
    all_margin_parts, all_y_parts, all_weight_parts, all_idx_parts = [], [], [], []
    inner_manifest = inner_cv_manifest.loc[
        inner_cv_manifest['outer_fold'] == outer0
    ]
    for inner0 in range(3):
        tr_rows = inner_manifest.loc[inner_manifest['inner_fold'] != inner0]
        va_rows = inner_manifest.loc[inner_manifest['inner_fold'] == inner0]
        inner_train_idx = tr_rows['original_row_index'].to_numpy()
        inner_val_idx = va_rows['original_row_index'].to_numpy()
        tr_groups = set(zip(tr_rows['stratum16'], tr_rows['psu']))
        va_groups = set(zip(va_rows['stratum16'], va_rows['psu']))
        assert not tr_groups.intersection(va_groups)
        assert not set(inner_train_idx).intersection(inner_val_idx)
        assert not set(inner_val_idx).difference(set(outer_training_indices))

        inner_y = women_analysis.loc[
            inner_train_idx, 'low_work_intensity'
        ].to_numpy(dtype=int)
        inner_w = women_analysis.loc[
            inner_train_idx, 'population_weight'
        ].to_numpy(dtype=float)
        X_inner_train, X_inner_val, _ = load_or_create_inner_matrices_140(
            outer0, inner0, inner_train_idx, inner_val_idx, inner_y, inner_w
        )
        svc = clone(tuned_model_templates['Support Vector Machine'])
        svc.set_params(**params, probability=False)
        svc.fit(
            X_inner_train, inner_y,
            sample_weight=inner_w / inner_w.mean()
        )
        assert np.array_equal(svc.classes_, [0, 1])
        margins = np.asarray(svc.decision_function(X_inner_val), dtype=float)
        assert len(margins) == len(inner_val_idx) and np.isfinite(margins).all()
        all_margin_parts.append(margins)
        all_idx_parts.append(inner_val_idx)
        all_y_parts.append(women_analysis.loc[
            inner_val_idx, 'low_work_intensity'
        ].to_numpy(dtype=int))
        all_weight_parts.append(women_analysis.loc[
            inner_val_idx, 'population_weight'
        ].to_numpy(dtype=float))

    # Each OUTER-training observation has exactly one cross-fitted margin.
    calib_idx = np.concatenate(all_idx_parts)
    assert len(calib_idx) == len(outer_training_indices)
    assert len(np.unique(calib_idx)) == len(outer_training_indices)
    assert set(calib_idx) == set(outer_training_indices)
    margins_all = np.concatenate(all_margin_parts)
    labels_all = np.concatenate(all_y_parts)
    weights_all = np.concatenate(all_weight_parts)
    assert np.isfinite(weights_all).all() and (weights_all > 0).all()
    assert set(np.unique(labels_all)) == {0, 1}

    # Fixed method, not chosen based on outer-fold validation data.
    calibrator = LogisticRegression(
        penalty='l2', C=1000.0, solver='lbfgs', max_iter=2000
    )
    calibrator.fit(
        margins_all.reshape(-1, 1), labels_all,
        sample_weight=weights_all / weights_all.mean()
    )
    assert np.array_equal(calibrator.classes_, [0, 1])
    slope = float(calibrator.coef_[0, 0])
    intercept = float(calibrator.intercept_[0])
    if not np.isfinite(slope) or slope <= 0:
        raise RuntimeError(
            f'Outer {outer0+1}: SVM calibration is non-monotone; investigate.'
        )
    return calibrator, {
        'Outer_Fold': outer0 + 1,
        'Calibration_Method': 'PSU-disjoint 3-inner-fold weighted Platt',
        'Training_Cases': len(calib_idx),
        'Positive_Cases': int(labels_all.sum()),
        'Platt_Slope': slope,
        'Platt_Intercept': intercept,
        'Margin_SD': float(margins_all.std()),
    }


# -------------------------------------------------------------------
# 4. Outer refit and held-out predictions. Save after EACH model.
# -------------------------------------------------------------------
print('========== STEP 1.41: NESTED OUTER PREDICTIONS ==========', flush=True)
print('Run fingerprint:', fingerprint_140, flush=True)
print('Results folder:', run_dir_141, flush=True)
print('Models:', ', '.join(models_141), flush=True)
print('This step scores outer folds for the FIRST time.', flush=True)

calibration_meta_rows_141 = []

for outer0_141 in range(5):
    outer_number = outer0_141 + 1
    outer_train = cv_manifest.loc[cv_manifest['outer_fold'] != outer0_141]
    outer_val = cv_manifest.loc[cv_manifest['outer_fold'] == outer0_141]
    train_idx = outer_train.index.to_numpy()
    val_idx = outer_val.index.to_numpy()
    assert len(set(train_idx) & set(val_idx)) == 0
    tr_psus = set(zip(outer_train['stratum16'], outer_train['psu']))
    va_psus = set(zip(outer_val['stratum16'], outer_val['psu']))
    assert len(tr_psus & va_psus) == 0
    assert len(train_idx) + len(val_idx) == 2110
    assert outer_val['stratum16'].nunique() == 16
    assert len(outer_train) >= 1680 and 419 <= len(outer_val) <= 430

    y_train = women_analysis.loc[
        train_idx, 'low_work_intensity'
    ].to_numpy(dtype=int)
    y_val = women_analysis.loc[
        val_idx, 'low_work_intensity'
    ].to_numpy(dtype=int)
    w_train = women_analysis.loc[
        train_idx, 'population_weight'
    ].to_numpy(dtype=float)
    w_val = women_analysis.loc[
        val_idx, 'population_weight'
    ].to_numpy(dtype=float)
    assert np.isfinite(w_train).all() and (w_train > 0).all()
    assert np.isfinite(w_val).all() and (w_val > 0).all()

    # Checkpoint full-outer-training fitted preprocessing + Boruta matrices.
    outer_cache_path = os.path.join(
        outer_cache_dir_141, f'Outer{outer_number}_Preprocessing_Boruta.npz'
    )
    if os.path.exists(outer_cache_path):
        with np.load(outer_cache_path, allow_pickle=False) as d:
            assert d['version'].item() == run_version_141
            assert d['fingerprint'].item() == fingerprint_140
            assert np.array_equal(d['train_idx'], train_idx)
            assert np.array_equal(d['val_idx'], val_idx)
            X_tr = d['X_train'].copy()
            X_va = d['X_validation'].copy()
            selected_groups = d['selected_groups'].tolist()
            names = d['feature_names'].tolist()
        cache_status = 'reused full-outer-training fit'
    else:
        selector = clone(boruta_group_template)
        selector.fit(
            X_raw.loc[train_idx], y_train,
            sample_weight=w_train
        )
        X_tr = selector.transform(X_raw.loc[train_idx])
        X_va = selector.transform(X_raw.loc[val_idx])
        selected_groups = list(selector.selected_groups_)
        names = list(selector.get_feature_names_out())
        atomic_npz_141(
            outer_cache_path,
            version=np.array(run_version_141),
            fingerprint=np.array(fingerprint_140),
            train_idx=train_idx,
            val_idx=val_idx,
            X_train=X_tr,
            X_validation=X_va,
            selected_groups=np.asarray(selected_groups, dtype=str),
            feature_names=np.asarray(names, dtype=str)
        )
        cache_status = 'new full-outer-training fit'

    assert X_tr.shape[0] == len(y_train)
    assert X_va.shape[0] == len(y_val)
    assert X_tr.shape[1] == X_va.shape[1] == len(names)
    assert np.isfinite(X_tr).all() and np.isfinite(X_va).all()
    assert len(selected_groups) >= 1
    assert set(selected_groups).issubset(set(model_feature_names))

    print('\n' + '=' * 68, flush=True)
    print(f'OUTER {outer_number}: train={len(train_idx)}, '
          f'holdout={len(val_idx)}, encoded={len(names)}', flush=True)
    print('Preprocessing + Boruta:', cache_status, flush=True)
    print('Selected original predictors:', selected_groups, flush=True)

    for model_name in models_141:
        params, params_json = expected_params_141(outer_number, model_name)
        safe_name = model_name.replace(' ', '_')
        checkpoint_path = os.path.join(
            model_dir_141, f'Outer{outer_number}_{safe_name}_Predictions.csv'
        )
        if os.path.exists(checkpoint_path):
            chk = pd.read_csv(checkpoint_path)
            assert len(chk) == len(val_idx)
            assert (chk['Run_Version'] == run_version_141).all()
            assert (chk['Fingerprint'] == fingerprint_140).all()
            assert (chk['Outer_Fold'] == outer_number).all()
            assert (chk['Model'] == model_name).all()
            assert (chk['Parameters_JSON'] == params_json).all()
            assert np.array_equal(chk['original_row_index'].to_numpy(), val_idx)
            assert np.array_equal(chk['Actual'].to_numpy(dtype=int), y_val)
            assert np.isfinite(chk['Predicted_Probability']).all()
            print(f'  {model_name}: already saved; skipped.', flush=True)
            continue

        model = clone(tuned_model_templates[model_name])
        model.set_params(**params)
        if model_name == 'Support Vector Machine':
            model.set_params(probability=False)
        model.fit(X_tr, y_train, sample_weight=w_train / w_train.mean())
        assert np.array_equal(model.classes_, [0, 1])

        if model_name == 'Support Vector Machine':
            # Refit best SVM to all outer train, apply training-only
            # calibration learned from 3 group-disjoint inner-fold margins.
            raw_score = np.asarray(model.decision_function(X_va), dtype=float)
            platt, meta = fit_svm_platt_141(
                outer0_141, params, train_idx
            )
            p = np.asarray(
                platt.predict_proba(raw_score.reshape(-1, 1))[:, 1],
                dtype=float
            )
            calibration_meta_rows_141.append(meta)
            scoring_method = 'PSU cross-fitted Platt-calibrated SVM'
        else:
            p = np.asarray(model.predict_proba(X_va)[:, 1], dtype=float)
            raw_score = p.copy()
            scoring_method = 'Native predicted probability'

        assert p.shape == raw_score.shape == (len(val_idx),)
        assert np.isfinite(p).all() and np.isfinite(raw_score).all()
        assert ((p >= 0) & (p <= 1)).all()

        prediction_data = pd.DataFrame({
            'Run_Version': run_version_141,
            'Fingerprint': fingerprint_140,
            'Model': model_name,
            'Outer_Fold': outer_number,
            'original_row_index': val_idx,
            'Actual': y_val,
            'population_weight': w_val,
            'stratum16': outer_val['stratum16'].to_numpy(),
            'psu': outer_val['psu'].to_numpy(),
            'Raw_Score': raw_score,
            'Predicted_Probability': p,
            'Scoring_Method': scoring_method,
            'Parameters_JSON': params_json,
            'Selected_Original_Predictors': ', '.join(selected_groups),
            'Retained_Encoded_Columns': len(names)
        })
        atomic_csv_141(prediction_data, checkpoint_path)
        print(f'  {model_name}: {len(val_idx)} held-out predictions saved.', flush=True)

# -------------------------------------------------------------------
# 5. Combine all 30 held-out prediction files; verify OOF integrity.
# -------------------------------------------------------------------
all_prediction_parts_141 = []
for outer_number in range(1, 6):
    for model_name in models_141:
        path = os.path.join(
            model_dir_141,
            f'Outer{outer_number}_{model_name.replace(" ", "_")}_Predictions.csv'
        )
        assert os.path.isfile(path)
        part = pd.read_csv(path)
        assert (part['Run_Version'] == run_version_141).all()
        assert (part['Fingerprint'] == fingerprint_140).all()
        assert (part['Outer_Fold'] == outer_number).all()
        assert (part['Model'] == model_name).all()
        assert part['original_row_index'].is_unique
        expected_val = cv_manifest.index[
            cv_manifest['outer_fold'] == outer_number - 1
        ].to_numpy()
        assert np.array_equal(part['original_row_index'].to_numpy(), expected_val)
        all_prediction_parts_141.append(part)

oof_predictions_141 = pd.concat(all_prediction_parts_141, ignore_index=True)
assert len(oof_predictions_141) == 2110 * 6
assert not oof_predictions_141.duplicated(
    ['Model', 'original_row_index']
).any()
assert oof_predictions_141.groupby('Model').size().eq(2110).all()
assert oof_predictions_141.groupby('Model')['Outer_Fold'].nunique().eq(5).all()
assert oof_predictions_141['Predicted_Probability'].between(0, 1).all()
assert np.isfinite(oof_predictions_141['Predicted_Probability']).all()

# -------------------------------------------------------------------
# 6. Describe held-out performance without using it for tuning.
#    Survey-weighted AP, AUC, Brier, and 0.50-threshold statistics.
# -------------------------------------------------------------------
def metrics_141(rows):
    yy = rows['Actual'].to_numpy(dtype=int)
    pp = rows['Predicted_Probability'].to_numpy(dtype=float)
    ww = rows['population_weight'].to_numpy(dtype=float)
    pred = pp >= 0.5
    tp = ww[(yy == 1) & pred].sum()
    fp = ww[(yy == 0) & pred].sum()
    tn = ww[(yy == 0) & (~pred)].sum()
    fn = ww[(yy == 1) & (~pred)].sum()
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    specific = tn / (tn + fp) if (tn + fp) else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    return {
        'N': len(rows), 'Low_N': int(yy.sum()),
        'Weighted_Outcome_Prevalence': float(np.average(yy, weights=ww)),
        'Weighted_AP': float(average_precision_score(yy, pp, sample_weight=ww)),
        'Weighted_ROC_AUC': float(roc_auc_score(yy, pp, sample_weight=ww)),
        'Weighted_Brier': float(np.average((pp - yy) ** 2, weights=ww)),
        'Weighted_Accuracy_at_0.50': float((tp + tn) / ww.sum()),
        'Weighted_Precision_at_0.50': float(prec),
        'Weighted_Recall_at_0.50': float(recall),
        'Weighted_Specificity_at_0.50': float(specific),
        'Weighted_F1_at_0.50': float(f1)
    }

fold_metric_rows_141, pooled_metric_rows_141 = [], []
for model_name in models_141:
    rows_model = oof_predictions_141.loc[oof_predictions_141['Model'] == model_name]
    assert len(rows_model) == 2110
    pooled_metric_rows_141.append({'Model': model_name, **metrics_141(rows_model)})
    for outer_number in range(1, 6):
        rows_fold = rows_model.loc[rows_model['Outer_Fold'] == outer_number]
        fold_metric_rows_141.append({
            'Model': model_name, 'Outer_Fold': outer_number,
            **metrics_141(rows_fold)
        })

pooled_oof_metrics_141 = pd.DataFrame(pooled_metric_rows_141)
outer_fold_metrics_141 = pd.DataFrame(fold_metric_rows_141)
assert len(pooled_oof_metrics_141) == 6 and len(outer_fold_metrics_141) == 30
assert np.isfinite(pooled_oof_metrics_141[
    ['Weighted_AP', 'Weighted_ROC_AUC', 'Weighted_Brier']
].to_numpy(dtype=float)).all()

atomic_csv_141(oof_predictions_141, os.path.join(
    run_dir_141, 'Step1_41_All_Six_Model_OOF_Predictions.csv'
))
atomic_csv_141(pooled_oof_metrics_141, os.path.join(
    run_dir_141, 'Step1_41_Preliminary_Pooled_OOF_Metrics.csv'
))
atomic_csv_141(outer_fold_metrics_141, os.path.join(
    run_dir_141, 'Step1_41_Preliminary_Outer_Fold_Metrics.csv'
))
if calibration_meta_rows_141:
    atomic_csv_141(pd.DataFrame(calibration_meta_rows_141), os.path.join(
        run_dir_141, 'Step1_41_SVM_Training_Only_Calibration_Audit.csv'
    ))

print('\n========== STEP 1.41 FINAL VERIFICATION ==========', flush=True)
print('Outer folds evaluated: 5', flush=True)
print('Models evaluated: 6', flush=True)
print('Independent held-out predictions:', len(oof_predictions_141), flush=True)
print('Predictions per model: 2110', flush=True)
print('Total per-fold metrics:', len(outer_fold_metrics_141), flush=True)
print('PSU overlap between outer training and outer validation: 0', flush=True)
print('\nPOOLED OUTER-OF-FOLD METRICS (DESCRIPTIVE; NO 95% CI YET):')
print(pooled_oof_metrics_141[
    ['Model', 'Weighted_AP', 'Weighted_ROC_AUC', 'Weighted_Brier',
     'Weighted_Recall_at_0.50', 'Weighted_F1_at_0.50']
].round(5).to_string(index=False))
print('\nCheckpoints:', run_dir_141)
print('Calibration curves/CI and full statistical evaluation: NEXT STEP.')
print('STEP 1.41 VERIFICATION SUCCESSFUL.', flush=True)
