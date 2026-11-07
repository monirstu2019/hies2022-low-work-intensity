# %% STEP 1.43 RECOVERY: SHAP checkpoints -> tables and publication figures
# Run in a FRESH Spyder kernel using a subprocess (see accompanying one-cell launcher).
# NO HIES data loading. NO model training. NO Boruta. NO SHAP recalculation.
# Each figure is rendered in a separate short-lived process to reduce peak memory.

import argparse
import csv
import gc
import os
import subprocess
import sys
from pathlib import Path

FINGERPRINT_DEFAULT = '7244d907321a2e433a44'
STEP143_VERSION = 'HIES2022_RF_OuterHoldout_SHAP_v1'
MODELS = [
    'Logistic Regression', 'Support Vector Machine', 'Random Forest',
    'Extra Trees', 'Gradient Boosting', 'AdaBoost',
]
FIGURES = [
    'shap_importance', 'shap_distribution', 'roc', 'pr', 'calibration'
]


def paths(out, fp):
    root = Path(out)
    return {
        'shap': root / f'Step1_43_RF_OuterHoldout_SHAP_{fp}',
        'outer': root / f'Step1_41_Nested_Outer_Predictions_{fp}',
        'eval': root / f'Step1_42_Calibration_Uncertainty_{fp}',
        'fp': fp,
    }


def require(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f'Missing required saved file: {path}')
    return path


def atomic_csv(df, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    df.to_csv(tmp, index=False, float_format='%.17g')
    os.replace(tmp, path)


def aggregate(out, fp):
    import numpy as np
    import pandas as pd

    d = paths(out, fp)
    shap_dir, outer_dir = d['shap'], d['outer']
    oof_path = require(outer_dir / 'Step1_41_All_Six_Model_OOF_Predictions.csv')
    all_preds = pd.read_csv(oof_path)
    rf_preds = all_preds.loc[all_preds['Model'] == 'Random Forest'].copy()
    assert len(rf_preds) == 2110 and rf_preds['original_row_index'].is_unique
    assert rf_preds['Outer_Fold'].nunique() == 5
    assert (rf_preds['Fingerprint'].astype(str) == fp).all()
    assert rf_preds['Actual'].sum() == 553

    frame_parts, selection_parts = [], []
    parent_names = None
    max_error = 0.0
    for fold in range(1, 6):
        checkpoint = require(shap_dir / 'outer_fold_checkpoints' / f'Outer{fold}_RF_GroupSHAP.npz')
        with np.load(checkpoint, allow_pickle=False) as c:
            assert c['version'].item() == STEP143_VERSION
            assert c['fingerprint'].item() == fp
            indices = c['row_indices'].copy()
            names = list(c['parent_names'].tolist())
            v = c['group_shap'].copy()
            base = c['baseline'].copy()
            probs = c['probabilities'].copy()
        if parent_names is None:
            parent_names = names
        assert names == parent_names and len(names) == 35
        ref = (rf_preds.loc[rf_preds['Outer_Fold'] == fold]
               .set_index('original_row_index').loc[indices].reset_index())
        target = ref['Predicted_Probability'].to_numpy(dtype=float)
        assert np.allclose(probs, target, atol=1e-8, rtol=1e-8)
        assert v.shape == (len(indices), 35)
        assert base.shape == probs.shape == (len(indices),)
        assert np.isfinite(v).all() and np.isfinite(base).all()
        error = float(np.max(np.abs(base + v.sum(axis=1) - target)))
        assert error < 1e-5
        max_error = max(error, max_error)
        frame = pd.DataFrame({
            'original_row_index': indices,
            'Outer_Fold': fold,
            'Actual': ref['Actual'].to_numpy(dtype=int),
            'population_weight': ref['population_weight'].to_numpy(dtype=float),
            'stratum16': ref['stratum16'].to_numpy(),
            'psu': ref['psu'].to_numpy(),
            'Predicted_Probability': target,
            'SHAP_Baseline_Probability': base,
        })
        for j, parent in enumerate(names):
            frame['SHAP__' + parent] = v[:, j]
        frame_parts.append(frame)
        training_cache = require(outer_dir / 'outer_train_fit_cache' /
                                 f'Outer{fold}_Preprocessing_Boruta.npz')
        with np.load(training_cache, allow_pickle=False) as cc:
            selected = cc['selected_groups'].tolist()
        assert set(selected).issubset(names)
        for parent in selected:
            selection_parts.append({'Outer_Fold': fold, 'Original_Predictor': parent})
        print(f'  Fold {fold}: {len(indices)} saved SHAP cases; additivity error {error:.3g}', flush=True)
        del frame, ref, v, base, probs

    combined = pd.concat(frame_parts, ignore_index=True)
    assert len(combined) == 2110 and combined['original_row_index'].is_unique
    assert combined['Outer_Fold'].nunique() == 5
    assert set(combined['original_row_index']) == set(rf_preds['original_row_index'])
    assert len(combined.groupby(['stratum16','psu'])['Outer_Fold'].nunique().loc[lambda s: s > 1]) == 0
    weight = combined['population_weight'].to_numpy(dtype=float)
    assert np.isfinite(weight).all() and (weight > 0).all()
    assert np.isclose(np.average(combined['Actual'].to_numpy(), weights=weight), 0.237812, atol=1e-5)
    full_array = combined[['SHAP__' + x for x in parent_names]].to_numpy(dtype=float)
    assert np.allclose(full_array.sum(axis=1) + combined['SHAP_Baseline_Probability'].to_numpy(),
                       combined['Predicted_Probability'].to_numpy(), atol=1e-5)
    selection = pd.DataFrame(selection_parts)
    counts = selection['Original_Predictor'].value_counts().to_dict()
    ranking = pd.DataFrame([{
        'Original_Predictor': name,
        'Selected_Outer_Folds': int(counts.get(name, 0)),
        'Weighted_Mean_Abs_SHAP_Probability': float(np.average(np.abs(full_array[:, j]), weights=weight)),
        'Weighted_Mean_Signed_SHAP_Probability': float(np.average(full_array[:, j], weights=weight)),
        'Observations_With_Nonzero_SHAP': int(np.sum(np.abs(full_array[:, j]) > 1e-12)),
    } for j, name in enumerate(parent_names)]).sort_values(
        ['Weighted_Mean_Abs_SHAP_Probability', 'Original_Predictor'],
        ascending=[False, True]
    ).reset_index(drop=True)
    assert len(ranking) == 35 and len(selection) >= 5
    shap_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(combined, shap_dir / 'Step1_43_RF_HeldOut_GroupSHAP_2110.csv')
    atomic_csv(ranking, shap_dir / 'Step1_43_RF_Weighted_GroupSHAP_Importance.csv')
    atomic_csv(selection, shap_dir / 'Step1_43_RF_Outer_Fold_Feature_Selections.csv')
    print('\nWeighted global feature importance (top 12):', flush=True)
    print(ranking.head(12).round(6).to_string(index=False), flush=True)
    print(f'\nRecovery data audit PASS: 2110 unique held-out cases; 35 groups; max additivity error {max_error:.3g}', flush=True)


def save_variant(fig, ax, folder, base, title, version):
    import matplotlib.pyplot as plt
    folder.mkdir(parents=True, exist_ok=True)
    if version == 'WithTitle':
        ax.set_title(title, fontsize=12)
    else:
        ax.set_title('')
    fig.tight_layout()
    # Generate one file at a time; every rendering process exits promptly.
    for ext, dpi, kw in (
        ('png', 300, {}),
        ('tiff', 600, {'pil_kwargs': {'compression': 'raw'}}),
        ('pdf', 150, {}),
    ):
        final = folder / f'{base}_{version}.{ext}'
        temp = folder / f'{base}_{version}.partial.{ext}'
        fig.savefig(str(temp), dpi=dpi, bbox_inches='tight',
                    facecolor='white', **kw)
        if temp.stat().st_size < 500:
            raise RuntimeError(f'Incomplete figure output: {temp}')
        os.replace(temp, final)
        print(f'    Saved {final.name} ({final.stat().st_size / 1024 / 1024:.2f} MB)', flush=True)
        gc.collect()
    plt.close(fig)
    gc.collect()


def render(out, fp, name, version):
    import matplotlib
    matplotlib.use('Agg', force=True)  # NO Spyder interactive backend, no notebook figure retention
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_curve, precision_recall_curve

    plt.rcParams['figure.max_open_warning'] = 5
    plt.rcParams['savefig.transparent'] = False
    d = paths(out, fp)
    shp, ev, od = d['shap'], d['eval'], d['outer']

    if name in ('shap_importance', 'shap_distribution'):
        rankings = pd.read_csv(require(shp / 'Step1_43_RF_Weighted_GroupSHAP_Importance.csv'))
        if name == 'shap_importance':
            top = rankings.head(12).iloc[::-1]
            fig, ax = plt.subplots(figsize=(9, 5.6))
            ax.barh(top['Original_Predictor'], top['Weighted_Mean_Abs_SHAP_Probability'])
            ax.set_xlabel('Survey-weighted mean absolute grouped SHAP (probability)')
            base = 'Step1_43_RF_GroupSHAP_Importance'
            title = 'Random Forest: held-out grouped SHAP importance'
        else:
            oof = pd.read_csv(require(shp / 'Step1_43_RF_HeldOut_GroupSHAP_2110.csv'))
            plot_names = rankings.head(10)['Original_Predictor'].tolist()
            rng = np.random.default_rng(20261030)
            fig, ax = plt.subplots(figsize=(9, 6))
            for j, label in enumerate(plot_names[::-1]):
                values = oof['SHAP__' + label].to_numpy(dtype=float)
                jitter = rng.normal(0, 0.08, len(values))
                ax.scatter(values, j + jitter, s=5, alpha=0.14, linewidths=0)
            ax.axvline(0, linestyle='--', color='gray', linewidth=1)
            ax.set_yticks(np.arange(len(plot_names)))
            ax.set_yticklabels(plot_names[::-1])
            ax.set_xlabel('SHAP contribution to predicted probability of low work intensity')
            base = 'Step1_43_RF_GroupSHAP_Distribution'
            title = 'Held-out RF SHAP distribution (unweighted observation dots)'
        folder = shp
    else:
        oof = pd.read_csv(require(od / 'Step1_41_All_Six_Model_OOF_Predictions.csv'))
        point = pd.read_csv(require(ev / 'Step1_42_Point_Metrics_Calibration.csv'))
        assert len(oof) == 12660
        assert set(oof['Model']) == set(MODELS)
        ref = oof.loc[oof['Model'] == 'Random Forest'].sort_values('original_row_index')
        ids = ref['original_row_index'].to_numpy()
        y = ref['Actual'].to_numpy(dtype=int)
        w = ref['population_weight'].to_numpy(dtype=float)
        assert len(ref) == 2110 and np.all(w > 0)
        model_preds = {}
        for model in MODELS:
            sub = oof.loc[oof['Model'] == model].sort_values('original_row_index')
            assert np.array_equal(sub['original_row_index'].to_numpy(), ids)
            assert np.array_equal(sub['Actual'].to_numpy(dtype=int), y)
            assert np.allclose(sub['population_weight'].to_numpy(), w)
            model_preds[model] = sub['Predicted_Probability'].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(8, 6))
        if name == 'roc':
            for model in MODELS:
                fpr, tpr, _ = roc_curve(y, model_preds[model], sample_weight=w)
                auc = float(point.loc[point['Model'] == model, 'Weighted_ROC_AUC'].iloc[0])
                ax.plot(fpr, tpr, label=f'{model} (AUC={auc:.3f})')
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
            ax.set_xlabel('False-positive rate (1 - specificity)')
            ax.set_ylabel('True-positive rate (sensitivity)')
            ax.legend(loc='lower right', fontsize=8)
            base, title = 'Step1_42_ROC_Curves', 'Survey-weighted outer-of-fold ROC curves'
        elif name == 'pr':
            for model in MODELS:
                prec, rec, _ = precision_recall_curve(y, model_preds[model], sample_weight=w)
                ap = float(point.loc[point['Model'] == model, 'Weighted_AP'].iloc[0])
                ax.plot(rec, prec, label=f'{model} (AP={ap:.3f})')
            baseline = float(np.average(y, weights=w))
            ax.axhline(baseline, linestyle='--', color='gray', linewidth=1,
                       label=f'Weighted prevalence ({baseline:.3f})')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.legend(loc='best', fontsize=8)
            base, title = 'Step1_42_PR_Curves', 'Survey-weighted outer-of-fold precision-recall curves'
        elif name == 'calibration':
            bins = pd.read_csv(require(ev / 'Step1_42_Weighted_Calibration_Bins.csv'))
            for model in MODELS:
                sub = bins.loc[bins['Model'] == model].sort_values('Bin')
                assert len(sub) > 0
                ax.plot(sub['Mean_Predicted'], sub['Observed_Frequency'],
                        marker='o', markersize=4, label=model)
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1,
                    label='Perfect calibration')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel('Mean predicted risk (survey weighted)')
            ax.set_ylabel('Observed low-work-intensity frequency (survey weighted)')
            ax.legend(loc='best', fontsize=8)
            base, title = 'Step1_42_Calibration_Curves', 'Outer-of-fold calibration by predicted-risk quantiles'
        else:
            raise ValueError(f'Invalid figure: {name}')
        folder = ev
    save_variant(fig, ax, folder, base, title, version)


def driver(out, fp):
    d = paths(out, fp)
    py = sys.executable
    src = Path(__file__).resolve()
    basecmd = [py, str(src), '--output-dir', str(out), '--fingerprint', fp]
    print('========== STEP 1.43 LOW-MEMORY RECOVERY ==========', flush=True)
    print('Existing SHAP checkpoints will be reused (NO model refitting).', flush=True)
    print('Figure renders are isolated from Spyder and from each other.', flush=True)

    job = subprocess.run(basecmd + ['--mode', 'aggregate'], check=False)
    if job.returncode:
        raise RuntimeError(f'SHAP aggregation failed (exit {job.returncode}). Saved SHAP checkpoints remain intact.')

    produced = []
    bases = {
        'shap_importance': ('shap', 'Step1_43_RF_GroupSHAP_Importance'),
        'shap_distribution': ('shap', 'Step1_43_RF_GroupSHAP_Distribution'),
        'roc': ('eval', 'Step1_42_ROC_Curves'),
        'pr': ('eval', 'Step1_42_PR_Curves'),
        'calibration': ('eval', 'Step1_42_Calibration_Curves'),
    }
    for fig in FIGURES:
        for version in ('WithTitle', 'NoTitle'):
            folder_key, name = bases[fig]
            expected = [d[folder_key] / f'{name}_{version}.{ext}' for ext in ('png', 'tiff', 'pdf')]
            if all(p.is_file() and p.stat().st_size > 500 for p in expected):
                print(f'  Existing exports verified: {fig} / {version} (skip).', flush=True)
            else:
                print(f'  Exporting: {fig} / {version}', flush=True)
                job = subprocess.run(basecmd + [
                    '--mode', 'render', '--figure', fig, '--version', version
                ], check=False)
                if job.returncode:
                    raise RuntimeError(
                        f'Figure export crashed: {fig}/{version} (exit {job.returncode}). '
                        'Spyder kernel is protected. Other saved files remain intact.'
                    )
            for path in expected:
                require(path)
                if path.stat().st_size <= 500:
                    raise RuntimeError(f'Corrupted/empty figure: {path}')
                produced.append((fig, version, path.suffix.lower().lstrip('.'), str(path), path.stat().st_size))

    assert len(produced) == 30
    audit = d['shap'] / 'Step1_43_Figure_Export_Audit.csv'
    with open(audit, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Figure', 'Version', 'Format', 'File', 'Size_Bytes'])
        writer.writerows(produced)
    for s in ('Step1_43_RF_HeldOut_GroupSHAP_2110.csv',
              'Step1_43_RF_Weighted_GroupSHAP_Importance.csv',
              'Step1_43_RF_Outer_Fold_Feature_Selections.csv'):
        require(d['shap'] / s)
    print('\n========== STEP 1.43 RECOVERY FINAL VERIFICATION ==========', flush=True)
    print('Saved SHAP outer folds reused: 5/5', flush=True)
    print('Held-out RF SHAP cases: 2110', flush=True)
    print('Original predictor groups: 35', flush=True)
    print('Five figures x two title versions x three formats: 30', flush=True)
    print('TIFF 600 DPI; PNG 300 DPI; PDF vector', flush=True)
    print('SHAP tables:', d['shap'], flush=True)
    print('ROC/PR/calibration figures:', d['eval'], flush=True)
    print('STEP 1.43 RECOVERY VERIFICATION SUCCESSFUL.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--fingerprint', default=FINGERPRINT_DEFAULT)
    parser.add_argument('--mode', default='driver', choices=['driver', 'aggregate', 'render'])
    parser.add_argument('--figure', choices=FIGURES)
    parser.add_argument('--version', choices=['WithTitle', 'NoTitle'])
    args = parser.parse_args()
    if args.mode == 'driver':
        driver(args.output_dir, args.fingerprint)
    elif args.mode == 'aggregate':
        aggregate(args.output_dir, args.fingerprint)
    else:
        if args.figure is None or args.version is None:
            parser.error('--figure and --version required for render')
        render(args.output_dir, args.fingerprint, args.figure, args.version)


if __name__ == '__main__':
    main()
