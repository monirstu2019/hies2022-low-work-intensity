# %% STEP 1.42: NESTED-CV OUTER-OF-FOLD CALIBRATION AND PSU BOOTSTRAP
# Run after successful Step 1.41, in the SAME Spyder console.
# Does NOT refit models or choose hyperparameters using held-out predictions.
# Confidence intervals: stratified PSU resampling from the FULL 720-PSU survey
# frame with OOF predictions held fixed (conditional/approximate intervals).

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, brentq
from scipy.special import expit, logit
from sklearn.metrics import (
    average_precision_score, roc_auc_score, roc_curve,
    precision_recall_curve
)

STEP142_VERSION = 'HIES2022_OOF_Calibration_Bootstrap_v1'
BOOTSTRAP_REPS_142 = 1000
BOOTSTRAP_SEED_142 = 20261030
EPS_142 = 1e-6
EXPECTED_MODELS_142 = [
    'Logistic Regression', 'Support Vector Machine',
    'Random Forest', 'Extra Trees', 'Gradient Boosting', 'AdaBoost'
]

assert 'run_dir_141' in globals() and 'fingerprint_140' in globals()
assert 'df' in globals(), 'Full HIES 2022 dataframe df is required for all 720 survey PSUs.'
assert len(women_analysis) == 2110
assert {'stratum16', 'psu'}.issubset(df.columns)
assert set(EXPECTED_MODELS_142) == set(models_141)

PREDICTIONS_CSV_142 = os.path.join(
    run_dir_141, 'Step1_41_All_Six_Model_OOF_Predictions.csv'
)
assert os.path.isfile(PREDICTIONS_CSV_142), PREDICTIONS_CSV_142
out_dir_142 = os.path.join(
    output_dir, f'Step1_42_Calibration_Uncertainty_{fingerprint_140}'
)
os.makedirs(out_dir_142, exist_ok=True)
bootstrap_checkpoint_142 = os.path.join(
    out_dir_142, 'Step1_42_PSU_Bootstrap_Checkpoint.csv'
)


def save_csv_atomic_142(data, destination):
    temporary = destination + '.temporary'
    data.to_csv(temporary, index=False, float_format='%.17g')
    os.replace(temporary, destination)


# ------------------------------------------------------------
# 1. Read and cross-check all 12,660 held-out predictions.
# ------------------------------------------------------------
oof_142 = pd.read_csv(PREDICTIONS_CSV_142)
assert len(oof_142) == 2110 * 6
assert oof_142.groupby('Model').size().eq(2110).all()
assert not oof_142.duplicated(['Model', 'original_row_index']).any()
assert (oof_142['Fingerprint'] == fingerprint_140).all()
assert set(oof_142['Model']) == set(EXPECTED_MODELS_142)

base_142 = (
    oof_142.loc[oof_142['Model'] == EXPECTED_MODELS_142[0]]
    .sort_values('original_row_index').reset_index(drop=True)
)
row_ids_142 = base_142['original_row_index'].to_numpy()
y_142 = base_142['Actual'].to_numpy(dtype=int)
w_142 = base_142['population_weight'].to_numpy(dtype=float)
assert len(row_ids_142) == 2110 and len(np.unique(row_ids_142)) == 2110
assert np.array_equal(np.unique(y_142), [0, 1])
assert np.isfinite(w_142).all() and (w_142 > 0).all()
assert np.array_equal(
    base_142['Outer_Fold'].to_numpy(dtype=int),
    cv_manifest.loc[row_ids_142, 'outer_fold'].to_numpy(dtype=int) + 1
)

pred_142 = {}
for model_name_142 in EXPECTED_MODELS_142:
    part = (
        oof_142.loc[oof_142['Model'] == model_name_142]
        .sort_values('original_row_index').reset_index(drop=True)
    )
    assert np.array_equal(part['original_row_index'].to_numpy(), row_ids_142)
    assert np.array_equal(part['Actual'].to_numpy(dtype=int), y_142)
    assert np.allclose(
        part['population_weight'].to_numpy(dtype=float), w_142, rtol=1e-11
    )
    assert np.array_equal(
        part['stratum16'].to_numpy(), base_142['stratum16'].to_numpy()
    )
    assert np.array_equal(part['psu'].to_numpy(), base_142['psu'].to_numpy())
    assert np.array_equal(part['Outer_Fold'].to_numpy(), base_142['Outer_Fold'].to_numpy())
    p = part['Predicted_Probability'].to_numpy(dtype=float)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    pred_142[model_name_142] = p

# ------------------------------------------------------------
# 2. Survey design: resample PSUs within each stratum from the
# FULL original survey PSU frame, including non-domain PSUs.
# ------------------------------------------------------------
full_psus_142 = df[['stratum16', 'psu']].drop_duplicates().reset_index(drop=True)
assert full_psus_142['stratum16'].nunique() == 16
assert len(full_psus_142) == 720, (
    f'Expected 720 full-survey PSUs, found {len(full_psus_142)}.'
)
full_group_index_142 = pd.MultiIndex.from_frame(full_psus_142)
analytic_group_index_142 = pd.MultiIndex.from_frame(
    base_142[['stratum16', 'psu']]
)
row_full_psu_142 = full_group_index_142.get_indexer(analytic_group_index_142)
assert (row_full_psu_142 >= 0).all()
analytic_psus_142 = len(np.unique(row_full_psu_142))
assert analytic_psus_142 == 589, analytic_psus_142
stratum_positions_142 = [
    np.asarray(indexes, dtype=int)
    for indexes in full_psus_142.groupby('stratum16', sort=True).indices.values()
]
assert len(stratum_positions_142) == 16
assert all(len(pos) >= 2 for pos in stratum_positions_142)
assert sum(len(pos) for pos in stratum_positions_142) == 720

# ------------------------------------------------------------
# 3. Metrics; calibration intercept/slope and calibration-in-
# the-large; all weighted by survey expansion weights.
# ------------------------------------------------------------
def metrics_142(y, p, weights):
    positive = weights > 0
    y, p, weights = y[positive], p[positive], weights[positive]
    if len(np.unique(y)) != 2:
        raise ValueError('A bootstrap replicate has only one outcome class.')
    classified = p >= 0.5
    tp = weights[(y == 1) & classified].sum()
    fp = weights[(y == 0) & classified].sum()
    tn = weights[(y == 0) & (~classified)].sum()
    fn = weights[(y == 1) & (~classified)].sum()
    total = tp + fp + tn + fn
    brier = np.average((y - p) ** 2, weights=weights)
    prevalence = np.average(y, weights=weights)
    return {
        'Weighted_AP': average_precision_score(y, p, sample_weight=weights),
        'Weighted_ROC_AUC': roc_auc_score(y, p, sample_weight=weights),
        'Weighted_Brier': brier,
        'Weighted_Brier_Skill': 1.0 - brier / (prevalence * (1 - prevalence)),
        'Weighted_Accuracy_at_0.50': (tp + tn) / total,
        'Weighted_Precision_at_0.50': tp / (tp + fp) if tp + fp else np.nan,
        'Weighted_Recall_at_0.50': tp / (tp + fn) if tp + fn else np.nan,
        'Weighted_Specificity_at_0.50': tn / (tn + fp) if tn + fp else np.nan,
        'Weighted_F1_at_0.50': (
            2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
        ),
    }


def calibration_142(y, p, weights):
    keep = weights > 0
    yy, pp, ww = y[keep], p[keep], weights[keep]
    x = logit(np.clip(pp, EPS_142, 1.0 - EPS_142))
    wn = ww / ww.sum()

    # Calibration-in-the-large: fixed slope=1, fit only logit offset.
    offset = brentq(
        lambda a: float(np.dot(wn, expit(x + a) - yy)),
        -40.0, 40.0
    )

    # Joint logistic recalibration: logit Pr(Y=1) = intercept + slope*logit(p).
    def objective(ab):
        eta = ab[0] + ab[1] * x
        residual = expit(eta) - yy
        loss = np.dot(wn, np.logaddexp(0.0, eta) - yy * eta)
        gradient = np.array([
            np.dot(wn, residual), np.dot(wn, residual * x)
        ])
        return loss, gradient

    result = minimize(
        objective, x0=np.array([float(offset), 1.0]),
        jac=True, method='L-BFGS-B',
        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    if (not np.isfinite(result.x).all()
            or np.linalg.norm(objective(result.x)[1]) > 1e-4):
        raise RuntimeError('Joint calibration model failed to converge.')
    return {
        'Calibration_In_Large': float(offset),
        'Calibration_Intercept': float(result.x[0]),
        'Calibration_Slope': float(result.x[1]),
    }


def calibration_bins_142(y, p, weights, model_name):
    # Quantiles based on unweighted prediction counts; observed/predicted
    # bin averages are survey weighted. Identical scores remain together.
    bins = pd.qcut(pd.Series(p), q=10, labels=False, duplicates='drop')
    if bins.isna().all():
        raise RuntimeError(f'No nonempty calibration bins for {model_name}')
    data = pd.DataFrame({'bin': bins, 'y': y, 'p': p, 'weight': weights})
    outputs = []
    for bin_number, grp in data.groupby('bin', sort=True):
        wt = grp['weight'].to_numpy(dtype=float)
        outputs.append({
            'Model': model_name,
            'Bin': int(bin_number) + 1,
            'N': len(grp),
            'Weighted_N': float(wt.sum()),
            'Mean_Predicted': float(np.average(grp['p'], weights=wt)),
            'Observed_Frequency': float(np.average(grp['y'], weights=wt)),
        })
    return outputs


print('========== STEP 1.42: CALIBRATION + SURVEY-PSU BOOTSTRAP ==========', flush=True)
print('Fingerprint:', fingerprint_140, flush=True)
print('Full survey PSUs:', len(full_psus_142), '| Domain PSUs:', analytic_psus_142, flush=True)
print('OOF observations per model:', len(y_142), flush=True)
print('Weighted observed prevalence:', round(np.average(y_142, weights=w_142), 6))

point_rows_142, bin_rows_142 = [], []
for name in EXPECTED_MODELS_142:
    p = pred_142[name]
    point_rows_142.append({
        'Model': name,
        'Mean_Predicted_Risk': float(np.average(p, weights=w_142)),
        'Observed_Weighted_Prevalence': float(np.average(y_142, weights=w_142)),
        'Number_Probabilities_Clipped_For_Logit': int(((p < EPS_142) | (p > 1-EPS_142)).sum()),
        **metrics_142(y_142, p, w_142),
        **calibration_142(y_142, p, w_142),
    })
    bin_rows_142.extend(calibration_bins_142(y_142, p, w_142, name))
point_table_142 = pd.DataFrame(point_rows_142)
bins_table_142 = pd.DataFrame(bin_rows_142)
save_csv_atomic_142(point_table_142, os.path.join(out_dir_142, 'Step1_42_Point_Metrics_Calibration.csv'))
save_csv_atomic_142(bins_table_142, os.path.join(out_dir_142, 'Step1_42_Weighted_Calibration_Bins.csv'))

# ------------------------------------------------------------
# 4. Resumable stratified full-frame PSU bootstrap. The SAME
# bootstrap PSU draw is used for all six models (paired CIs).
# In every replicate, hold the cross-fitted predictions fixed.
# ------------------------------------------------------------
if os.path.isfile(bootstrap_checkpoint_142):
    boot_df_142 = pd.read_csv(bootstrap_checkpoint_142)
    assert (boot_df_142['Version'] == STEP142_VERSION).all()
    assert (boot_df_142['Fingerprint'] == fingerprint_140).all()
    assert (boot_df_142['Bootstrap_Seed'] == BOOTSTRAP_SEED_142).all()
    assert boot_df_142.groupby('Replicate').size().eq(6).all()
    assert set(boot_df_142['Model']) == set(EXPECTED_MODELS_142)
    print('Existing complete bootstrap replicates:', boot_df_142['Replicate'].nunique(), flush=True)
else:
    boot_df_142 = pd.DataFrame()

completed_142 = set() if boot_df_142.empty else set(boot_df_142['Replicate'].astype(int))
assert all(1 <= b <= BOOTSTRAP_REPS_142 for b in completed_142)
new_boot_rows_142 = []
for b in range(1, BOOTSTRAP_REPS_142 + 1):
    if b in completed_142:
        continue
    rng = np.random.default_rng(BOOTSTRAP_SEED_142 + b)
    group_multiplicity = np.zeros(len(full_psus_142), dtype=int)
    for indexes in stratum_positions_142:
        n_psus = len(indexes)
        group_multiplicity[indexes] = rng.multinomial(
            n_psus, np.full(n_psus, 1.0 / n_psus)
        )
    replicate_weights = w_142 * group_multiplicity[row_full_psu_142]
    if (replicate_weights[y_142 == 1].sum() <= 0
            or replicate_weights[y_142 == 0].sum() <= 0):
        raise RuntimeError(f'Bootstrap replicate {b} has no weighted cases in a class.')
    for model_name in EXPECTED_MODELS_142:
        p = pred_142[model_name]
        result_row = {
            'Version': STEP142_VERSION,
            'Fingerprint': fingerprint_140,
            'Bootstrap_Seed': BOOTSTRAP_SEED_142,
            'Replicate': b,
            'Model': model_name,
            **metrics_142(y_142, p, replicate_weights),
            **calibration_142(y_142, p, replicate_weights),
        }
        new_boot_rows_142.append(result_row)
    if b % 50 == 0 or b == BOOTSTRAP_REPS_142:
        appended = pd.DataFrame(new_boot_rows_142)
        if boot_df_142.empty:
            boot_df_142 = appended
        elif not appended.empty:
            boot_df_142 = pd.concat([boot_df_142, appended], ignore_index=True)
        new_boot_rows_142 = []
        assert not boot_df_142.duplicated(['Replicate', 'Model']).any()
        save_csv_atomic_142(boot_df_142, bootstrap_checkpoint_142)
        print(f'  Completed and checkpointed PSU bootstrap: {b}/{BOOTSTRAP_REPS_142}', flush=True)

assert len(boot_df_142) == BOOTSTRAP_REPS_142 * 6
assert boot_df_142.groupby('Replicate').size().eq(6).all()

# ------------------------------------------------------------
# 5. Percentile 95% CIs and paired RF-minus-comparator CIs.
# ------------------------------------------------------------
metric_keys_142 = list(metrics_142(y_142, pred_142[EXPECTED_MODELS_142[0]], w_142))
metric_keys_142 += [
    'Calibration_In_Large', 'Calibration_Intercept', 'Calibration_Slope'
]
summary_rows_142 = []
for name in EXPECTED_MODELS_142:
    point = point_table_142.loc[point_table_142['Model'] == name].iloc[0]
    replicate_rows = boot_df_142.loc[boot_df_142['Model'] == name]
    result = {'Model': name, 'Bootstrap_Reps': BOOTSTRAP_REPS_142}
    for key in metric_keys_142:
        x = replicate_rows[key].to_numpy(dtype=float)
        assert len(x) == BOOTSTRAP_REPS_142 and np.isfinite(x).all()
        result[key] = float(point[key])
        result[key + '_CI_Lower'] = float(np.quantile(x, 0.025))
        result[key + '_CI_Upper'] = float(np.quantile(x, 0.975))
    summary_rows_142.append(result)
summary_142 = pd.DataFrame(summary_rows_142)
save_csv_atomic_142(summary_142, os.path.join(out_dir_142, 'Step1_42_Main_Performance_95CI.csv'))

paired_rows_142 = []
for metric_name in ['Weighted_AP', 'Weighted_ROC_AUC', 'Weighted_Brier']:
    pv = boot_df_142.pivot(index='Replicate', columns='Model', values=metric_name)
    point_rf = float(point_table_142.loc[
        point_table_142['Model'] == 'Random Forest', metric_name
    ].iloc[0])
    for name in EXPECTED_MODELS_142:
        if name == 'Random Forest':
            continue
        point_other = float(point_table_142.loc[
            point_table_142['Model'] == name, metric_name
        ].iloc[0])
        delta = (pv['Random Forest'] - pv[name]).to_numpy(dtype=float)
        paired_rows_142.append({
            'Metric': metric_name,
            'Contrast': 'Random Forest minus ' + name,
            'Point_Difference': point_rf - point_other,
            'Lower_95_CI': float(np.quantile(delta, 0.025)),
            'Upper_95_CI': float(np.quantile(delta, 0.975)),
        })
paired_142 = pd.DataFrame(paired_rows_142)
save_csv_atomic_142(paired_142, os.path.join(out_dir_142, 'Step1_42_Exploratory_Paired_RF_Differences.csv'))

# ------------------------------------------------------------
# 6. Manuscript-ready ROC, PR and calibration figures.
# Save separate 300-dpi PNGs and editable vector PDFs.
# ------------------------------------------------------------
plt.figure(figsize=(8, 6))
for name in EXPECTED_MODELS_142:
    fpr, tpr, _ = roc_curve(y_142, pred_142[name], sample_weight=w_142)
    auc = float(point_table_142.loc[point_table_142['Model'] == name, 'Weighted_ROC_AUC'].iloc[0])
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1, color='gray')
plt.xlabel('False-positive rate (1 − specificity)')
plt.ylabel('True-positive rate (sensitivity)')
plt.title('Survey-weighted outer-of-fold ROC curves')
plt.legend(loc='lower right', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(out_dir_142, 'Step1_42_ROC_Curves.png'), dpi=300)
plt.savefig(os.path.join(out_dir_142, 'Step1_42_ROC_Curves.pdf'))
plt.close()

plt.figure(figsize=(8, 6))
for name in EXPECTED_MODELS_142:
    precision, recall, _ = precision_recall_curve(
        y_142, pred_142[name], sample_weight=w_142
    )
    ap = float(point_table_142.loc[point_table_142['Model'] == name, 'Weighted_AP'].iloc[0])
    plt.plot(recall, precision, label=f'{name} (AP={ap:.3f})')
prevalence_142 = float(np.average(y_142, weights=w_142))
plt.axhline(prevalence_142, linestyle='--', linewidth=1, color='gray',
            label=f'Weighted prevalence ({prevalence_142:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Survey-weighted outer-of-fold precision–recall curves')
plt.legend(loc='best', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(out_dir_142, 'Step1_42_PR_Curves.png'), dpi=300)
plt.savefig(os.path.join(out_dir_142, 'Step1_42_PR_Curves.pdf'))
plt.close()

plt.figure(figsize=(8, 6))
for name in EXPECTED_MODELS_142:
    bd = bins_table_142.loc[bins_table_142['Model'] == name].sort_values('Bin')
    plt.plot(bd['Mean_Predicted'], bd['Observed_Frequency'], marker='o',
             markersize=4, label=name)
plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1, color='gray',
         label='Perfect calibration')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('Mean predicted risk (survey weighted)')
plt.ylabel('Observed low-work-intensity frequency (survey weighted)')
plt.title('Outer-of-fold calibration by predicted-risk quantiles')
plt.legend(loc='best', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(out_dir_142, 'Step1_42_Calibration_Curves.png'), dpi=300)
plt.savefig(os.path.join(out_dir_142, 'Step1_42_Calibration_Curves.pdf'))
plt.close()

# ------------------------------------------------------------
# 7. Console verification; no claims of model superiority yet.
# ------------------------------------------------------------
print('\n========== STEP 1.42 FINAL VERIFICATION ==========', flush=True)
print('Models:', len(EXPECTED_MODELS_142))
print('Held-out predictions:', len(oof_142))
print('Bootstrap PSU frame:', len(full_psus_142), 'across', len(stratum_positions_142), 'strata')
print('PSU bootstrap replicates:', BOOTSTRAP_REPS_142)
print('Paired RF comparisons:', len(paired_142))
print('\nWEIGHTED PERFORMANCE WITH APPROXIMATE 95% PSU-BOOTSTRAP CIs:')
report_cols = ['Model', 'Weighted_AP', 'Weighted_AP_CI_Lower', 'Weighted_AP_CI_Upper',
               'Weighted_ROC_AUC', 'Weighted_ROC_AUC_CI_Lower',
               'Weighted_ROC_AUC_CI_Upper', 'Weighted_Brier']
print(summary_142[report_cols].round(5).to_string(index=False))
print('\nCALIBRATION INTERCEPT / SLOPE / IN-THE-LARGE:')
print(point_table_142[['Model', 'Mean_Predicted_Risk', 'Calibration_Intercept',
                       'Calibration_Slope', 'Calibration_In_Large']].round(5).to_string(index=False))
print('\nExploratory RF-minus-comparator differences (95% CI):')
print(paired_142.round(5).to_string(index=False))
print('\nFigures saved: ROC, PR, calibration (PDF and PNG).')
print('Output folder:', out_dir_142)
print('CI interpretation: stratified full-survey PSU resampling with OOF predictions fixed;')
print('does NOT propagate uncertainty from model refitting, hyperparameter selection,')
print('nor original survey finite population corrections. Model comparisons are exploratory.')
print('STEP 1.42 VERIFICATION SUCCESSFUL.', flush=True)
