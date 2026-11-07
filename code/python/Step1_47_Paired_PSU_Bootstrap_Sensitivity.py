# %% STEP 1.47: PAIRED SURVEY PSU BOOTSTRAP FOR FULLY NESTED RF SENSITIVITY
# Standalone: only saved Step 1.46 cross-fitted predictions and full survey PSU frame.
# No feature selection, tuning, fitting or SHAP recalculation.
# Approximate design-based intervals conditional on fixed OOF predictions.

from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

VERSION = 'HIES22_Step147_PairedPSU_Bootstrap_v1'
EXPECTED = ('Full_35', 'Remove_Core_3', 'Remove_All_Employment_6')
METRICS = ('Weighted_AP', 'Weighted_ROC_AUC', 'Weighted_Brier')
SEED = 20261030


def atomic_csv(df, path):
    temporary = path.with_suffix(path.suffix + '.temporary')
    df.to_csv(temporary, index=False, float_format='%.17g')
    os.replace(temporary, path)


def score(y, p, w):
    keep = w > 0
    yy = y[keep]
    pp = p[keep]
    ww = w[keep]
    if (np.unique(yy).size != 2 or not np.isfinite(ww).all()
            or ww.sum() <= 0):
        raise RuntimeError('Bootstrap sample has invalid outcome/weight support.')
    return {
        'Weighted_AP': float(average_precision_score(yy, pp, sample_weight=ww)),
        'Weighted_ROC_AUC': float(roc_auc_score(yy, pp, sample_weight=ww)),
        'Weighted_Brier': float(np.average((yy - pp)**2, weights=ww)),
    }


def interval(values):
    return [float(v) for v in np.quantile(np.asarray(values, dtype=float), [.025, .975])]


def write_figure(plot_dir, panel_table, prevalence, show_title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    import tifffile

    title_version = 'WithTitle' if show_title else 'NoTitle'
    stem = 'Figure_RF_FullyNested_Employment_Sensitivity_' + title_version
    png_path = plot_dir / (stem + '.png')
    pdf_path = plot_dir / (stem + '.pdf')
    tiff_path = plot_dir / (stem + '.tiff')
    tmp_png = plot_dir / (stem + '.TEMP_600dpi.png')
    tmp_tiff = plot_dir / (stem + '.partial.tiff')

    colors = ['#287A61', '#6A57A7', '#DA8B29']  # style based on previous manuscript
    names = ['All 35 predictors', 'Without 3 core employment predictors',
             'Without all 6 employment predictors']
    plot_order = list(EXPECTED)
    yloc = [2, 1, 0]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8), sharey=True)
    pretty = [('Weighted_AP', 'Average precision (AP)', 'Higher is better'),
              ('Weighted_ROC_AUC', 'ROC-AUC', 'Higher is better'),
              ('Weighted_Brier', 'Brier score', 'Lower is better')]
    for ax, (metric, xlabel, direction) in zip(axes, pretty):
        row = panel_table.loc[panel_table['Metric'] == metric].set_index('Scenario')
        value = np.array([row.loc[n, 'Estimate'] for n in plot_order])
        lower = np.array([row.loc[n, 'Lower_95_CI'] for n in plot_order])
        upper = np.array([row.loc[n, 'Upper_95_CI'] for n in plot_order])
        assert np.all(lower <= value) and np.all(value <= upper)
        for j, (v, l, u) in enumerate(zip(value, lower, upper)):
            ax.errorbar(v, yloc[j], xerr=np.array([[v-l], [u-v]]),
                        fmt='o', markersize=8, color=colors[j],
                        ecolor=colors[j], elinewidth=2, capsize=4, capthick=1.5)
            ax.annotate(f'{v:.3f}', xy=(v, yloc[j]), xytext=(0, 12),
                        textcoords='offset points', ha='center', fontsize=9,
                        color=colors[j], weight='semibold')
        if metric == 'Weighted_AP':
            ax.axvline(prevalence, ls='--', lw=1.2, color='0.45', alpha=.8)
        ax.set_xlabel(xlabel + '\n(' + direction + ')', fontsize=10)
        ax.set_yticks(yloc)
        ax.grid(axis='x', color='#D9DFE6', linestyle=':', linewidth=0.9)
        ax.set_axisbelow(True)
        ax.tick_params(axis='x', labelsize=9)
        vals = np.concatenate([value, lower, upper])
        span = float(vals.max()-vals.min())
        margin = max(.014, span*.20)
        ax.set_xlim(max(0, float(vals.min()-margin)),
                    min(1, float(vals.max()+margin)))
        ax.set_ylim(-.65, 2.7)
    axes[0].set_yticklabels(names, fontsize=9.4)
    axes[0].tick_params(axis='y', length=0)
    for ax in axes[1:]:
        ax.tick_params(axis='y', length=0)
    if show_title:
        fig.suptitle('Sensitivity of Random Forest performance to employment predictors',
                     fontsize=13, fontweight='semibold', y=.985)
    fig.subplots_adjust(left=.255, right=.985, top=.82 if show_title else .90,
                        bottom=.20, wspace=.24)

    try:
        fig.savefig(png_path, dpi=300, facecolor='white')
        fig.savefig(pdf_path, facecolor='white')
        # A fresh 600-dpi raster -> lossless TIFF; no Pillow TIFF writer.
        fig.savefig(tmp_png, dpi=600, facecolor='white')
    finally:
        plt.close(fig)
        gc.collect()

    try:
        with Image.open(tmp_png) as img:
            array = np.array(img.convert('RGB'))
        tifffile.imwrite(str(tmp_tiff), array, photometric='rgb',
                         compression='zlib', compressionargs={'level': 6},
                         resolution=(600, 600), resolutionunit='INCH', metadata=None)
        del array
        gc.collect()
        with tifffile.TiffFile(str(tmp_tiff)) as tf:
            page = tf.pages[0]
            actual = page.tags['XResolution'].value
            dpi = actual[0] / actual[1]
            assert abs(dpi-600) < .01
            assert int(page.compression) == 8
            assert int(page.tags['ResolutionUnit'].value) == 2
        os.replace(tmp_tiff, tiff_path)
    finally:
        tmp_png.unlink(missing_ok=True)
        # if writer failed, keep partial TIFF for diagnosis; do not touch other outputs.

    assert all(p.exists() and p.stat().st_size > 1024
               for p in (png_path, pdf_path, tiff_path))
    print('  Saved:', png_path.name, tiff_path.name, pdf_path.name, flush=True)


def main():
    parser = argparse.ArgumentParser()
    project_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--root", default=os.environ.get("HIES_OUTPUT_DIR", str(project_root / "outputs")))
    parser.add_argument("--stata", default=os.environ.get("HIES_MERGED_DATA", str(project_root / "restricted_data" / "final_hies_women_lfp_earnings_ml_ready.dta")))
    parser.add_argument('--fingerprint', default='7244d907321a2e433a44')
    parser.add_argument('--run', default='Run_68763c127e9a15d939a1')
    parser.add_argument('--reps', type=int, default=1000)
    args = parser.parse_args()
    assert args.reps >= 20

    root = Path(args.root)
    run_dir = (root / f'Step1_46_FullyNested_RF_Sensitivity_{args.fingerprint}'
               / args.run)
    pred_file = run_dir / 'Step1_46_All_Scenario_OOF_Predictions.csv'
    summary_file = run_dir / 'Step1_46_Nested_Sensitivity_Summary.csv'
    stata_path = Path(args.stata)
    assert pred_file.is_file(), pred_file
    assert summary_file.is_file(), summary_file
    assert stata_path.is_file(), stata_path
    out_dir = run_dir / 'Step1_47_Paired_PSU_Bootstrap'
    out_dir.mkdir(exist_ok=True, parents=True)
    checkpoint = out_dir / 'Step1_47_Paired_Bootstrap_Checkpoint.csv'
    source_hash = hashlib.sha256(pred_file.read_bytes()).hexdigest()[:20]

    print('========== STEP 1.47 PAIRED PSU-BOOTSTRAP ==========', flush=True)
    print('Run:', run_dir, flush=True)
    print('Prediction-file fingerprint:', source_hash, flush=True)
    print('Survey PSU resampling; no new model fitting.', flush=True)
    oof = pd.read_csv(pred_file)
    stored = pd.read_csv(summary_file).set_index('Scenario')
    assert len(oof) == 2110*len(EXPECTED)
    assert set(oof['Scenario']) == set(EXPECTED)
    assert set(stored.index) == set(EXPECTED)
    assert not oof.duplicated(['Scenario', 'original_row_index']).any()
    assert oof.groupby('Scenario').size().eq(2110).all()
    assert oof.groupby('Scenario')['Outer_Fold'].nunique().eq(5).all()
    assert oof['Run_Hash'].nunique() == 1
    assert oof['Run_Hash'].iloc[0] == args.run.removeprefix('Run_')

    ref = oof[oof['Scenario'] == 'Full_35'].sort_values('original_row_index').reset_index(drop=True)
    ids = ref['original_row_index'].to_numpy(dtype=np.int64)
    y = ref['Actual'].to_numpy(dtype=int)
    w = ref['population_weight'].to_numpy(dtype=float)
    assert set(np.unique(y)) == {0, 1} and y.sum() == 553
    assert np.isfinite(w).all() and (w > 0).all()
    prevalence = float(np.average(y, weights=w))
    assert abs(prevalence-0.237812) < .00005
    preds = {}
    for sc in EXPECTED:
        df = oof[oof['Scenario'] == sc].sort_values('original_row_index').reset_index(drop=True)
        assert np.array_equal(df['original_row_index'].to_numpy(), ids)
        assert np.array_equal(df['Actual'].to_numpy(dtype=int), y)
        assert np.allclose(df['population_weight'].to_numpy(dtype=float), w, rtol=1e-12)
        assert np.array_equal(df['stratum16'].to_numpy(), ref['stratum16'].to_numpy())
        assert np.array_equal(df['psu'].to_numpy(), ref['psu'].to_numpy())
        assert np.array_equal(df['Outer_Fold'].to_numpy(), ref['Outer_Fold'].to_numpy())
        p = df['Predicted_Probability'].to_numpy(dtype=float)
        assert np.isfinite(p).all() and ((0<=p)&(p<=1)).all()
        preds[sc] = p

    # Exactly the same full-survey PSU-frame procedure used in Step 1.42.
    full_psus = (pd.read_stata(stata_path, columns=['stratum16', 'psu'],
                              convert_categoricals=False)
                 .drop_duplicates().reset_index(drop=True))
    assert len(full_psus) == 720 and full_psus['stratum16'].nunique() == 16
    survey_index = pd.MultiIndex.from_frame(full_psus)
    analytic_index = pd.MultiIndex.from_frame(ref[['stratum16', 'psu']])
    analytic_group_id = survey_index.get_indexer(analytic_index)
    assert (analytic_group_id >= 0).all()
    assert len(np.unique(analytic_group_id)) == 589
    strata = [np.asarray(pos, dtype=int)
              for pos in full_psus.groupby('stratum16', sort=True).indices.values()]
    assert len(strata) == 16 and all(len(pos) >= 2 for pos in strata)
    assert sum(map(len, strata)) == 720

    point = {sc:score(y, preds[sc], w) for sc in EXPECTED}
    for sc in EXPECTED:
        for metric in METRICS:
            assert np.isclose(point[sc][metric], float(stored.loc[sc,metric]), atol=1e-9), (sc,metric)
    print('Full survey PSUs: 720 | Domain PSUs: 589 | Strata: 16', flush=True)
    print('Matched predictions: 2110 per scenario; weighted prevalence:',
          round(prevalence, 6), flush=True)

    # Paired bootstrap replicates; the SAME PSU draw is applied to all 3 scenarios.
    if checkpoint.is_file():
        boot = pd.read_csv(checkpoint)
        assert (boot['Version'] == VERSION).all()
        assert (boot['Prediction_Hash'] == source_hash).all()
        assert (boot['Seed'] == SEED).all()
        assert boot['Replicate'].is_unique
        assert boot['Replicate'].between(1, args.reps).all()
        print('Previously saved replicates:', len(boot), flush=True)
    else:
        boot = pd.DataFrame()
    completed = set() if boot.empty else set(boot['Replicate'].astype(int))
    batch = []
    for rep in range(1, args.reps+1):
        if rep in completed:
            continue
        rng = np.random.default_rng(SEED+rep)
        multiplicities = np.zeros(len(full_psus), dtype=np.int32)
        for indices in strata:
            n = len(indices)
            multiplicities[indices] = rng.multinomial(n, np.full(n, 1/n))
        replicate_weights = w * multiplicities[analytic_group_id]
        assert replicate_weights[y == 0].sum() > 0 and replicate_weights[y == 1].sum() > 0
        row = {'Version': VERSION, 'Prediction_Hash': source_hash,
               'Seed':SEED, 'Replicate':rep}
        for sc in EXPECTED:
            for metric, value in score(y,preds[sc],replicate_weights).items():
                row[sc+'__'+metric] = value
        batch.append(row)
        if rep%50 == 0 or rep == args.reps:
            addition = pd.DataFrame(batch)
            boot = addition if boot.empty else pd.concat([boot, addition], ignore_index=True)
            boot = boot.sort_values('Replicate').reset_index(drop=True)
            assert boot['Replicate'].is_unique
            atomic_csv(boot, checkpoint)
            batch=[]
            print(f'  PSU bootstrap completed and saved: {rep}/{args.reps}', flush=True)
    assert len(boot) == args.reps and boot['Replicate'].nunique() == args.reps

    point_rows, comparison_rows = [], []
    for sc in EXPECTED:
        for metric in METRICS:
            est=point[sc][metric]
            lower,upper=interval(boot[sc+'__'+metric])
            point_rows.append({'Scenario':sc, 'Metric':metric,
                               'Estimate':est, 'Lower_95_CI':lower,
                               'Upper_95_CI':upper, 'Bootstrap_Replicates':args.reps})
    for sc in EXPECTED[1:]:
        for metric in METRICS:
            # For AP/AUC, full minus reduced; for Brier, reduced minus full.
            if metric == 'Weighted_Brier':
                direction='Reduced minus full (increase = worse)'
                est=point[sc][metric]-point['Full_35'][metric]
                vals=boot[sc+'__'+metric]-boot['Full_35__'+metric]
            else:
                direction='Full minus reduced (decrease = worse)'
                est=point['Full_35'][metric]-point[sc][metric]
                vals=boot['Full_35__'+metric]-boot[sc+'__'+metric]
            lower,upper=interval(vals)
            comparison_rows.append({'Contrast':'Full_35 vs '+sc,
                                    'Metric':metric, 'Direction':direction,
                                    'Performance_Loss':est,
                                    'Lower_95_CI':lower,'Upper_95_CI':upper,
                                    'CI_Excludes_Zero': bool(lower > 0 or upper < 0),
                                    'Bootstrap_Replicates':args.reps})
    point_table = pd.DataFrame(point_rows)
    comp_table = pd.DataFrame(comparison_rows)
    atomic_csv(point_table, out_dir/'Step1_47_Scenario_Metrics_95CI.csv')
    atomic_csv(comp_table, out_dir/'Step1_47_Paired_Performance_Loss_95CI.csv')

    print('\n========== STEP 1.47: SCENARIO METRICS (95% CIs) ==========', flush=True)
    print(point_table.round(5).to_string(index=False),flush=True)
    print('\n========== STEP 1.47: PAIRED PERFORMANCE LOSS ==========', flush=True)
    print(comp_table.round(5).to_string(index=False),flush=True)

    fig_dir=out_dir/'Figures_Journal_Style'
    fig_dir.mkdir(exist_ok=True)
    for title in (True, False):
        write_figure(fig_dir,point_table,prevalence,show_title=title)

    annotation=(
        'Figure. Fully nested sensitivity of Random Forest performance to exclusion of '
        'employment-structure predictors among working women in HIES 2022. '
        'Points indicate survey-weighted performance on five-fold outer out-of-fold predictions; '
        'horizontal error bars show approximate 95% percentile intervals from '
        f'{args.reps:,} paired bootstrap replicates resampling the full 720-PSU survey frame '
        'within 16 strata. The precision-recall baseline equals the survey-weighted '
        f'outcome prevalence ({prevalence:.2%}). All pre-processing, PCA, Boruta feature selection, '
        'and random-forest hyperparameter tuning were repeated inside each training fold. '
        'The same held-out women and PSU partitions were used in all three scenarios. '
        'Confidence intervals condition on the fitted cross-validated predictions and '
        'exclude refitting/tuning uncertainty and finite-population corrections. '
        'Paired differences are exploratory and not adjusted for multiple comparisons. '
        'Lower Brier score indicates better probability accuracy. These associations are not causal.\n'
        'Labels: Full model, 35 candidate predictors; core-3 exclusion removes agricultural '
        'and non-agricultural employment status plus organization/workplace type; all-6 '
        'exclusion additionally removes sector, industry and occupation group.\n'
    )
    (out_dir/'Step1_47_Figure_Caption_and_Notes.txt').write_text(annotation,encoding='utf-8')
    pngs=list(fig_dir.glob('*.png')); tiffs=list(fig_dir.glob('*.tiff')); pdfs=list(fig_dir.glob('*.pdf'))
    assert len(pngs)==len(tiffs)==len(pdfs)==2
    assert all(p.stat().st_size > 1024 for p in pngs+tiffs+pdfs)
    print('\n========== STEP 1.47 FINAL VERIFICATION ==========', flush=True)
    print('Scenarios: 3; 2110 held-out predictions each',flush=True)
    print('Matched prediction rows: 6330; PSU frame: 720 (16 strata)',flush=True)
    print(f'Paired PSU bootstrap replicates: {args.reps}',flush=True)
    print('Paired contrasts: 6 (2 exclusions x 3 metrics)',flush=True)
    print('Output figures: 2 title variants x 3 formats = 6',flush=True)
    print('TIFF: 600 DPI, lossless compression; PNG: 300 DPI; PDF: vector',flush=True)
    print('Model refitting: NO; fixed OOF predictions used',flush=True)
    print('Output folder:',out_dir,flush=True)
    print('STEP 1.47 VERIFICATION SUCCESSFUL.',flush=True)


if __name__ == '__main__':
    main()
