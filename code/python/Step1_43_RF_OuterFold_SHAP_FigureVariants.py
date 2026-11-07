# %% STEP 1.43: HELD-OUT RANDOM FOREST SHAP (ORIGINAL-PREDICTOR GROUPS)
# Run in SAME Spyder console after Steps 1.41 and 1.42.
# Refit ONLY the preselected Random Forest in each outer-training split,
# using exact cached training-only PCA/Boruta matrices and tuned settings.
# Never change/refit preprocessing, Boruta, or hyperparameters using holdouts.
# The SHAP estimand is contribution to Random Forest predicted P(Y=1),
# not a causal effect. Whole one-hot groups are added back together.

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.base import clone

STEP143_VERSION = 'HIES2022_RF_OuterHoldout_SHAP_v1'
assert len(women_analysis) == 2110 and len(cv_manifest) == 2110
assert len(best_140) == 30 and len(model_feature_names) == 35
assert 'fingerprint_140' in globals() and 'run_dir_141' in globals()
assert callable(boruta_group_demo._identify_parent)

shap_folder_143 = os.path.join(
    output_dir, f'Step1_43_RF_OuterHoldout_SHAP_{fingerprint_140}'
)
os.makedirs(shap_folder_143, exist_ok=True)
checkpoint_dir_143 = os.path.join(shap_folder_143, 'outer_fold_checkpoints')
os.makedirs(checkpoint_dir_143, exist_ok=True)

all_oof_path_143 = os.path.join(
    run_dir_141, 'Step1_41_All_Six_Model_OOF_Predictions.csv'
)
assert os.path.isfile(all_oof_path_143)
rf_oof_143 = pd.read_csv(all_oof_path_143)
rf_oof_143 = rf_oof_143.loc[rf_oof_143['Model'] == 'Random Forest'].copy()
assert len(rf_oof_143) == 2110
assert rf_oof_143['original_row_index'].is_unique
assert (rf_oof_143['Fingerprint'] == fingerprint_140).all()

parent_order_143 = list(model_feature_names)
parent_to_number_143 = {name: i for i, name in enumerate(parent_order_143)}
assert len(parent_to_number_143) == 35
all_fold_frames_143 = []
outer_feature_rows_143 = []
max_additivity_error_143 = 0.0


def save_csv_143(frame, path):
    tmp = path + '.tmp'
    frame.to_csv(tmp, index=False, float_format='%.17g')
    os.replace(tmp, path)


def save_npz_143(path, **arrays):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(tmp, path)


def positive_class_shap_143(values, n_rows, n_columns):
    # Boruta uses sklearn 1.5.2 / SHAP 0.50; handle both common SHAP APIs.
    if isinstance(values, list):
        assert len(values) == 2
        positive_values = np.asarray(values[1], dtype=float)
    else:
        arr = np.asarray(values, dtype=float)
        if arr.shape == (n_rows, n_columns, 2):
            positive_values = arr[:, :, 1]
        elif arr.shape == (2, n_rows, n_columns):
            positive_values = arr[1]
        else:
            raise RuntimeError(f'Unexpected SHAP value array shape: {arr.shape}')
    assert positive_values.shape == (n_rows, n_columns)
    assert np.isfinite(positive_values).all()
    return positive_values


print('========== STEP 1.43: HELD-OUT RF SHAP ==========', flush=True)
print('Run fingerprint:', fingerprint_140, flush=True)
print('SHAP version:', shap.__version__, flush=True)
print('Original predictors:', len(parent_order_143), flush=True)

for outer0_143 in range(5):
    outer_number_143 = outer0_143 + 1
    train_rows_143 = cv_manifest.loc[
        cv_manifest['outer_fold'] != outer0_143
    ]
    valid_rows_143 = cv_manifest.loc[
        cv_manifest['outer_fold'] == outer0_143
    ]
    train_idx_143 = train_rows_143.index.to_numpy()
    valid_idx_143 = valid_rows_143.index.to_numpy()
    assert not set(train_idx_143) & set(valid_idx_143)
    assert not set(zip(train_rows_143['stratum16'], train_rows_143['psu'])) & set(
        zip(valid_rows_143['stratum16'], valid_rows_143['psu'])
    )

    outer_ref_143 = rf_oof_143.loc[
        rf_oof_143['Outer_Fold'] == outer_number_143
    ].set_index('original_row_index').loc[valid_idx_143].reset_index()
    target_prob_143 = outer_ref_143['Predicted_Probability'].to_numpy(dtype=float)
    assert np.array_equal(
        outer_ref_143['Actual'].to_numpy(dtype=int),
        women_analysis.loc[valid_idx_143, 'low_work_intensity'].to_numpy(dtype=int)
    )
    assert np.allclose(
        outer_ref_143['population_weight'].to_numpy(dtype=float),
        women_analysis.loc[valid_idx_143, 'population_weight'].to_numpy(dtype=float)
    )
    saved_outer_cache_143 = os.path.join(
        run_dir_141, 'outer_train_fit_cache',
        f'Outer{outer_number_143}_Preprocessing_Boruta.npz'
    )
    assert os.path.isfile(saved_outer_cache_143)
    with np.load(saved_outer_cache_143, allow_pickle=False) as d:
        assert d['fingerprint'].item() == fingerprint_140
        assert np.array_equal(d['train_idx'], train_idx_143)
        assert np.array_equal(d['val_idx'], valid_idx_143)
        X_train_143 = d['X_train'].copy()
        X_holdout_143 = d['X_validation'].copy()
        selected_names_143 = d['feature_names'].tolist()
        selected_groups_143 = d['selected_groups'].tolist()
    assert X_train_143.shape[0] == len(train_idx_143)
    assert X_holdout_143.shape == (len(valid_idx_143), len(selected_names_143))
    assert np.isfinite(X_train_143).all() and np.isfinite(X_holdout_143).all()
    assert set(selected_groups_143).issubset(parent_to_number_143)
    parents_143 = [boruta_group_demo._identify_parent(s) for s in selected_names_143]
    assert set(parents_143) == set(selected_groups_143)

    selected_params_143 = best_140.loc[
        (best_140['Outer_Fold'] == outer_number_143)
        & (best_140['Model'] == 'Random Forest')
    ]
    assert len(selected_params_143) == 1
    params_json_143 = str(selected_params_143.iloc[0]['Parameters_JSON'])
    params_143 = json.loads(params_json_143)

    ckpt_path_143 = os.path.join(
        checkpoint_dir_143, f'Outer{outer_number_143}_RF_GroupSHAP.npz'
    )
    if os.path.isfile(ckpt_path_143):
        with np.load(ckpt_path_143, allow_pickle=False) as d:
            assert d['version'].item() == STEP143_VERSION
            assert d['fingerprint'].item() == fingerprint_140
            assert d['parameters'].item() == params_json_143
            assert np.array_equal(d['row_indices'], valid_idx_143)
            assert np.array_equal(d['parent_names'], np.asarray(parent_order_143))
            group_shap_143 = d['group_shap'].copy()
            base_values_143 = d['baseline'].copy()
            computed_p_143 = d['probabilities'].copy()
        print(f'  Outer {outer_number_143}: checked SHAP checkpoint reused.', flush=True)
    else:
        y_train_143 = women_analysis.loc[
            train_idx_143, 'low_work_intensity'
        ].to_numpy(dtype=int)
        weights_train_143 = women_analysis.loc[
            train_idx_143, 'population_weight'
        ].to_numpy(dtype=float)
        assert np.isfinite(weights_train_143).all() and (weights_train_143 > 0).all()

        rf_143 = clone(tuned_model_templates['Random Forest'])
        rf_143.set_params(**params_143)
        rf_143.fit(
            X_train_143, y_train_143,
            sample_weight=weights_train_143 / weights_train_143.mean()
        )
        assert np.array_equal(rf_143.classes_, [0, 1])
        computed_p_143 = rf_143.predict_proba(X_holdout_143)[:, 1]
        if not np.allclose(computed_p_143, target_prob_143, rtol=1e-8, atol=1e-8):
            raise RuntimeError(
                f'Outer {outer_number_143}: refitted Random Forest DOES NOT reproduce '
                'the exact Step 1.41 held-out probabilities. Stop and investigate.'
            )
        print(f'  Outer {outer_number_143}: RF OOF predictions reproduced; calculating SHAP...', flush=True)
        explainer_143 = shap.TreeExplainer(
            rf_143, feature_perturbation='tree_path_dependent',
            model_output='raw'
        )
        shap_columns_143 = positive_class_shap_143(
            explainer_143.shap_values(X_holdout_143),
            len(valid_idx_143), len(selected_names_143)
        )
        baseline_143 = np.asarray(explainer_143.expected_value, dtype=float)
        assert baseline_143.shape == (2,) and np.isfinite(baseline_143).all()
        base_values_143 = np.full(len(valid_idx_143), baseline_143[1])
        group_shap_143 = np.zeros((len(valid_idx_143), len(parent_order_143)), dtype=float)
        for feature_index_143, parent_name_143 in enumerate(parents_143):
            group_shap_143[:, parent_to_number_143[parent_name_143]] += (
                shap_columns_143[:, feature_index_143]
            )
        save_npz_143(
            ckpt_path_143,
            version=np.array(STEP143_VERSION),
            fingerprint=np.array(fingerprint_140),
            parameters=np.array(params_json_143),
            row_indices=valid_idx_143,
            parent_names=np.asarray(parent_order_143, dtype=str),
            group_shap=group_shap_143,
            baseline=base_values_143,
            probabilities=computed_p_143
        )
        print(f'  Outer {outer_number_143}: SHAP checkpoint saved.', flush=True)

    assert group_shap_143.shape == (len(valid_idx_143), len(parent_order_143))
    assert np.isfinite(group_shap_143).all() and np.isfinite(base_values_143).all()
    assert np.allclose(computed_p_143, target_prob_143, rtol=1e-8, atol=1e-8)
    additivity_error_143 = float(np.max(np.abs(
        base_values_143 + group_shap_143.sum(axis=1) - target_prob_143
    )))
    max_additivity_error_143 = max(max_additivity_error_143, additivity_error_143)
    assert additivity_error_143 < 1e-5, (
        f'Outer {outer_number_143}: SHAP additivity failed: {additivity_error_143}'
    )

    case_data_143 = pd.DataFrame({
        'original_row_index': valid_idx_143,
        'Outer_Fold': outer_number_143,
        'Actual': outer_ref_143['Actual'].to_numpy(dtype=int),
        'population_weight': outer_ref_143['population_weight'].to_numpy(dtype=float),
        'stratum16': outer_ref_143['stratum16'].to_numpy(),
        'psu': outer_ref_143['psu'].to_numpy(),
        'Predicted_Probability': target_prob_143,
        'SHAP_Baseline_Probability': base_values_143,
    })
    for feat_no_143, feat_143 in enumerate(parent_order_143):
        case_data_143['SHAP__' + feat_143] = group_shap_143[:, feat_no_143]
    all_fold_frames_143.append(case_data_143)
    for group_143 in selected_groups_143:
        outer_feature_rows_143.append({
            'Outer_Fold': outer_number_143, 'Original_Predictor': group_143,
            'Encoded_Columns': int(sum(x == group_143 for x in parents_143)),
            'Parameters_JSON': params_json_143,
        })
    print(f'  Outer {outer_number_143}: n={len(valid_idx_143)}, features={len(selected_groups_143)}, '
          f'additivity error={additivity_error_143:.3g}', flush=True)

# --------------------------------------------------------------------
# Combine held-out SHAP for 2,110 unique women; all 35 original predictor
# groups appear in each row (zero attribution when absent from a fold).
# --------------------------------------------------------------------
oof_group_shap_143 = pd.concat(all_fold_frames_143, ignore_index=True)
assert len(oof_group_shap_143) == 2110
assert oof_group_shap_143['original_row_index'].is_unique
assert oof_group_shap_143['Outer_Fold'].nunique() == 5
assert np.array_equal(
    np.sort(oof_group_shap_143['original_row_index'].to_numpy()),
    np.sort(cv_manifest.index.to_numpy())
)
shap_arr_143 = oof_group_shap_143[
    ['SHAP__' + x for x in parent_order_143]
].to_numpy(dtype=float)
weights_143 = oof_group_shap_143['population_weight'].to_numpy(dtype=float)
assert np.isfinite(shap_arr_143).all() and np.isfinite(weights_143).all()
assert (weights_143 > 0).all()
assert np.allclose(
    oof_group_shap_143['SHAP_Baseline_Probability'].to_numpy()
    + shap_arr_143.sum(axis=1),
    oof_group_shap_143['Predicted_Probability'].to_numpy(), atol=1e-5
)

selection_table_143 = pd.DataFrame(outer_feature_rows_143)
assert selection_table_143['Outer_Fold'].nunique() == 5
n_selections_143 = selection_table_143.groupby('Original_Predictor').size().to_dict()
global_rows_143 = []
for i, parent_143 in enumerate(parent_order_143):
    sv = shap_arr_143[:, i]
    global_rows_143.append({
        'Original_Predictor': parent_143,
        'Selected_Outer_Folds': int(n_selections_143.get(parent_143, 0)),
        'Weighted_Mean_Abs_SHAP_Probability': float(np.average(np.abs(sv), weights=weights_143)),
        'Weighted_Mean_Signed_SHAP_Probability': float(np.average(sv, weights=weights_143)),
        'Observations_With_Nonzero_SHAP': int(np.sum(np.abs(sv) > 1e-12)),
    })
global_shap_143 = pd.DataFrame(global_rows_143).sort_values(
    ['Weighted_Mean_Abs_SHAP_Probability', 'Original_Predictor'],
    ascending=[False, True]
).reset_index(drop=True)
assert len(global_shap_143) == 35
assert len(selection_table_143) >= 5

save_csv_143(oof_group_shap_143, os.path.join(
    shap_folder_143, 'Step1_43_RF_HeldOut_GroupSHAP_2110.csv'
))
save_csv_143(global_shap_143, os.path.join(
    shap_folder_143, 'Step1_43_RF_Weighted_GroupSHAP_Importance.csv'
))
save_csv_143(selection_table_143, os.path.join(
    shap_folder_143, 'Step1_43_RF_Outer_Fold_Feature_Selections.csv'
))

# ------------------------------------------------------------
# Publication-quality figure exports: BOTH title versions.
# Every figure: PNG (300 dpi), TIFF (600 dpi, LZW), PDF (vector).
# The "NoTitle" version omits the title INSIDE the image only;
# labels and legend remain for interpretation. Captions should be
# placed below the figure in the manuscript.
# ------------------------------------------------------------
from sklearn.metrics import roc_curve, precision_recall_curve

FIGURE_EXPORT_VERSION_143 = 'HIES2022_FigureFormats_v2'
figure_export_log_143 = []


def save_figure_versions_143(fig, ax, folder, basename, title):
    for label, displayed_title in [('WithTitle', title), ('NoTitle', '')]:
        ax.set_title(displayed_title, fontsize=12 if displayed_title else None)
        fig.tight_layout()
        for ext, dpi, extras in [
            ('png', 300, {}),
            ('tiff', 600, {'pil_kwargs': {'compression': 'tiff_lzw'}}),
            ('pdf', 300, {}),
        ]:
            path = os.path.join(folder, f'{basename}_{label}.{ext}')
            fig.savefig(
                path, dpi=dpi, bbox_inches='tight',
                facecolor='white', **extras
            )
            assert os.path.isfile(path) and os.path.getsize(path) > 0
            figure_export_log_143.append({
                'Figure': basename, 'Version': label,
                'Format': ext.upper(), 'DPI': dpi if ext != 'pdf' else 'vector',
                'File': path
            })
    # Also keep the pre-existing filenames with titles, for compatibility.
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, basename + '.png'), dpi=300,
                bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(folder, basename + '.pdf'),
                bbox_inches='tight', facecolor='white')
    plt.close(fig)


# 1. Weighted global importance bars
# ------------------------------------------------------------
top_143 = global_shap_143.head(12).iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 5.6))
ax.barh(top_143['Original_Predictor'], top_143['Weighted_Mean_Abs_SHAP_Probability'])
ax.set_xlabel('Survey-weighted mean absolute grouped SHAP (predicted probability)')
save_figure_versions_143(
    fig, ax, shap_folder_143, 'Step1_43_RF_GroupSHAP_Importance',
    'Random Forest: held-out SHAP importance across five outer folds'
)

# 2. SHAP dot distribution (unweighted observation dots)
# ------------------------------------------------------------
plot_names_143 = global_shap_143.head(10)['Original_Predictor'].tolist()
rng_143 = np.random.default_rng(20261030)
fig, ax = plt.subplots(figsize=(9, 6))
for j, name_143 in enumerate(plot_names_143[::-1]):
    values_143 = oof_group_shap_143['SHAP__' + name_143].to_numpy(dtype=float)
    jitter_143 = rng_143.normal(0, 0.08, size=len(values_143))
    ax.scatter(values_143, j + jitter_143, s=5, alpha=0.14, linewidths=0)
ax.axvline(0, linestyle='--', color='gray', linewidth=1)
ax.set_yticks(np.arange(len(plot_names_143)))
ax.set_yticklabels(plot_names_143[::-1])
ax.set_xlabel('Group SHAP contribution to predicted probability of low work intensity')
save_figure_versions_143(
    fig, ax, shap_folder_143, 'Step1_43_RF_GroupSHAP_Distribution',
    'Held-out RF SHAP distribution (unweighted case dots)'
)

# 3-5. Re-export Step 1.42 ROC, PR and calibration plots.
# Rebuild from SAVED outer-of-fold predictions and calibration bins;
# no model refitting, bootstrap reruns or new inference required.
# ------------------------------------------------------------
step142_fig_dir_143 = os.path.join(
    output_dir, f'Step1_42_Calibration_Uncertainty_{fingerprint_140}'
)
step142_metric_file_143 = os.path.join(
    step142_fig_dir_143, 'Step1_42_Point_Metrics_Calibration.csv'
)
step142_bin_file_143 = os.path.join(
    step142_fig_dir_143, 'Step1_42_Weighted_Calibration_Bins.csv'
)
assert os.path.isfile(step142_metric_file_143)
assert os.path.isfile(step142_bin_file_143)

point_metrics_143 = pd.read_csv(step142_metric_file_143)
calibration_bins_saved_143 = pd.read_csv(step142_bin_file_143)
all_oof_for_figures_143 = pd.read_csv(all_oof_path_143)
model_names_for_figures_143 = [
    'Logistic Regression', 'Support Vector Machine', 'Random Forest',
    'Extra Trees', 'Gradient Boosting', 'AdaBoost'
]
assert set(point_metrics_143['Model']) == set(model_names_for_figures_143)
assert len(all_oof_for_figures_143) == 12660

ref_oof_fig_143 = (
    all_oof_for_figures_143.loc[
        all_oof_for_figures_143['Model'] == 'Random Forest'
    ].sort_values('original_row_index').reset_index(drop=True)
)
assert len(ref_oof_fig_143) == 2110
ids_fig_143 = ref_oof_fig_143['original_row_index'].to_numpy()
y_fig_143 = ref_oof_fig_143['Actual'].to_numpy(dtype=int)
w_fig_143 = ref_oof_fig_143['population_weight'].to_numpy(dtype=float)
assert np.isfinite(w_fig_143).all() and (w_fig_143 > 0).all()

p_fig_143 = {}
for model_143 in model_names_for_figures_143:
    part_143 = (
        all_oof_for_figures_143.loc[
            all_oof_for_figures_143['Model'] == model_143
        ].sort_values('original_row_index').reset_index(drop=True)
    )
    assert np.array_equal(part_143['original_row_index'].to_numpy(), ids_fig_143)
    assert np.array_equal(part_143['Actual'].to_numpy(dtype=int), y_fig_143)
    assert np.allclose(part_143['population_weight'].to_numpy(dtype=float),
                       w_fig_143, rtol=1e-11)
    p_fig_143[model_143] = part_143['Predicted_Probability'].to_numpy(dtype=float)
    assert np.isfinite(p_fig_143[model_143]).all()

# 3. ROC curves
fig, ax = plt.subplots(figsize=(8, 6))
for model_143 in model_names_for_figures_143:
    fpr_143, tpr_143, _ = roc_curve(
        y_fig_143, p_fig_143[model_143], sample_weight=w_fig_143
    )
    auc_143 = float(point_metrics_143.loc[
        point_metrics_143['Model'] == model_143, 'Weighted_ROC_AUC'
    ].iloc[0])
    ax.plot(fpr_143, tpr_143, label=f'{model_143} (AUC={auc_143:.3f})')
ax.plot([0, 1], [0, 1], linestyle='--', linewidth=1, color='gray')
ax.set_xlabel('False-positive rate (1 − specificity)')
ax.set_ylabel('True-positive rate (sensitivity)')
ax.legend(loc='lower right', fontsize=8)
save_figure_versions_143(
    fig, ax, step142_fig_dir_143, 'Step1_42_ROC_Curves',
    'Survey-weighted outer-of-fold ROC curves'
)

# 4. Precision-recall curves
fig, ax = plt.subplots(figsize=(8, 6))
for model_143 in model_names_for_figures_143:
    precision_143, recall_143, _ = precision_recall_curve(
        y_fig_143, p_fig_143[model_143], sample_weight=w_fig_143
    )
    ap_143 = float(point_metrics_143.loc[
        point_metrics_143['Model'] == model_143, 'Weighted_AP'
    ].iloc[0])
    ax.plot(recall_143, precision_143, label=f'{model_143} (AP={ap_143:.3f})')
prevalence_143 = float(np.average(y_fig_143, weights=w_fig_143))
ax.axhline(prevalence_143, linestyle='--', linewidth=1, color='gray',
           label=f'Weighted prevalence ({prevalence_143:.3f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.legend(loc='best', fontsize=8)
save_figure_versions_143(
    fig, ax, step142_fig_dir_143, 'Step1_42_PR_Curves',
    'Survey-weighted outer-of-fold precision–recall curves'
)

# 5. Binned calibration curves
fig, ax = plt.subplots(figsize=(8, 6))
for model_143 in model_names_for_figures_143:
    frame_143 = calibration_bins_saved_143.loc[
        calibration_bins_saved_143['Model'] == model_143
    ].sort_values('Bin')
    ax.plot(frame_143['Mean_Predicted'], frame_143['Observed_Frequency'],
            marker='o', markersize=4, label=model_143)
ax.plot([0, 1], [0, 1], linestyle='--', linewidth=1,
        color='gray', label='Perfect calibration')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('Mean predicted risk (survey weighted)')
ax.set_ylabel('Observed low-work-intensity frequency (survey weighted)')
ax.legend(loc='best', fontsize=8)
save_figure_versions_143(
    fig, ax, step142_fig_dir_143, 'Step1_42_Calibration_Curves',
    'Outer-of-fold calibration by predicted-risk quantiles'
)

figure_export_audit_143 = pd.DataFrame(figure_export_log_143)
assert len(figure_export_audit_143) == 5 * 2 * 3
assert figure_export_audit_143.groupby('Figure').size().eq(6).all()
figure_export_audit_143.to_csv(
    os.path.join(shap_folder_143, 'Step1_43_Figure_Export_Audit.csv'),
    index=False
)

print('\n========== STEP 1.43 VERIFICATION ==========', flush=True)
print('Refitted Random Forest models:', 5, flush=True)
print('Held-out RF SHAP cases:', len(oof_group_shap_143), flush=True)
print('Original predictor groups:', len(parent_order_143), flush=True)
print('Maximum SHAP additivity error:', max_additivity_error_143, flush=True)
print('Weighted global feature importance (top 12):', flush=True)
print(global_shap_143.head(12).round(6).to_string(index=False), flush=True)
print('\nFigure export variants:', len(figure_export_audit_143), 'files across 5 figures', flush=True)
print('Step 1.42 ROC/PR/calibration folders:', step142_fig_dir_143, flush=True)
print('Step 1.43 SHAP figures folder:', shap_folder_143, flush=True)
print('Formats: titled and untitled PNG (300 dpi), TIFF (600 dpi), PDF (vector).', flush=True)
print('\nOutputs:', shap_folder_143, flush=True)
print('STEP 1.43 VERIFICATION SUCCESSFUL.', flush=True)
