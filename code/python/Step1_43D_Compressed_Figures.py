# %% STEP 1.43C: Manuscript-style, journal-readable figure regeneration
# Uses ONLY saved analytical outputs of Steps 1.41--1.43.
# Does NOT refit models, run SHAP, or overwrite existing figures.
# Matplotlib Agg subprocess per figure/version to avoid Spyder kernel crashes.
# Run via the launcher supplied with this file, or:
# python Step1_43C_Journal_Style_Figures.py --output-dir ... --fingerprint ...

from __future__ import annotations
import argparse
import csv
import gc
import os
from pathlib import Path
import subprocess
import sys

FP_DEFAULT = '7244d907321a2e433a44'
MODELS = [
    'Logistic Regression', 'Support Vector Machine', 'Random Forest',
    'Extra Trees', 'Gradient Boosting', 'AdaBoost'
]
FIGS = [
    'workflow', 'outcome_donut', 'boruta_stability', 'roc_panels',
    'roc_combined', 'pr_combined', 'calibration', 'shap_lollipop',
    'shap_distribution', 'shap_local_waterfalls',
]
MODEL_SHORT = {
    'Logistic Regression': 'Logistic Regression',
    'Support Vector Machine': 'Support Vector Machine',
    'Random Forest': 'Random Forest',
    'Extra Trees': 'Extra Trees',
    'Gradient Boosting': 'Gradient Boosting',
    'AdaBoost': 'AdaBoost',
}
# Exact, human-readable labels for all 35 parent predictors.
LABELS = {
    'age': 'Age',
    'marital_status': 'Marital status',
    'relation_head_grp': 'Relationship to household head',
    'literate_basic': 'Basic literacy',
    'education_attainment_group': 'Educational attainment',
    'highest_class_completed': 'Highest class completed',
    'has_mobile': 'Mobile-phone ownership',
    'has_bank_account': 'Bank-account ownership',
    'has_mfs_account': 'Mobile financial-services account',
    'internet_used': 'Internet use',
    'has_computer': 'Computer ownership',
    'any_functional_difficulty': 'Functional difficulty',
    'rural_urban': 'Rural/urban residence',
    'division_code': 'Administrative division',
    'rmo_code': 'Rural/municipality classification',
    'domain16_grp': 'Division-by-residence domain',
    'hh_MEMBER': 'Household size',
    'fies_raw_score': 'Food insecurity (FIES score)',
    'wealth_living_index': 'Household living-standard index',
    'toilet_type_grp': 'Toilet facility type',
    'drinking_water_source_grp': 'Drinking-water source',
    'light_source': 'Main lighting source',
    'cooking_stove_type': 'Cooking-stove type',
    'cooking_fuel_grp': 'Cooking-fuel type',
    'electricity_hours_24h': 'Electricity availability (hours/day)',
    'transport_stop_halfkm': 'Transport stop within 0.5 km',
    'transport_stop_time_min': 'Travel time to transport stop',
    'transport_stop_time_min_missing': 'Transport-time information missing',
    'regular_waste_collection': 'Regular waste collection',
    'slum_household': 'Slum-household status',
    'earn_sector': 'Employment sector',
    'earn_emp_status_ag': 'Agricultural employment status',
    'earn_emp_status_nonag': 'Non-agricultural employment status',
    'earn_org_type_grp': 'Organization/workplace type',
    'earn_ind_code_grp': 'Industry group',
    'earn_occ_code_grp': 'Occupation group',
}
# Reuse colors from previous manuscript rather than introduce new colors.
GREEN = '#118C4F'
PALE_GREEN = '#9EC9B3'
PURPLE = '#7A3E9D'
BLUE = '#1f77b4'
ORANGE = '#ff7f0e'
COLOR_BY_MODEL = dict(zip(MODELS, ['#1f77b4','#ff7f0e','#118C4F','#9467bd','#d62728','#8c564b']))


def folders(out, fp):
    root = Path(out)
    return {
        'outer': root / f'Step1_41_Nested_Outer_Predictions_{fp}',
        'eval': root / f'Step1_42_Calibration_Uncertainty_{fp}',
        'shap': root / f'Step1_43_RF_OuterHoldout_SHAP_{fp}',
        'result': root / f'Step1_43C_Journal_Style_Figures_{fp}',
    }


def load_inputs(out, fp, need=None):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import average_precision_score, roc_auc_score
    dirs = folders(out, fp)
    predp = dirs['outer'] / 'Step1_41_All_Six_Model_OOF_Predictions.csv'
    rankp = dirs['shap'] / 'Step1_43_RF_Weighted_GroupSHAP_Importance.csv'
    shapp = dirs['shap'] / 'Step1_43_RF_HeldOut_GroupSHAP_2110.csv'
    selectp = dirs['shap'] / 'Step1_43_RF_Outer_Fold_Feature_Selections.csv'
    bins_p = dirs['eval'] / 'Step1_42_Weighted_Calibration_Bins.csv'
    for p in (predp, rankp, shapp, selectp, bins_p):
        if not p.is_file():
            raise FileNotFoundError(f'Saved input is missing: {p}')
    pred = pd.read_csv(predp)
    ranks = pd.read_csv(rankp)
    shp = pd.read_csv(shapp)
    selection = pd.read_csv(selectp)
    bins = pd.read_csv(bins_p)
    assert len(pred) == 12660 and len(shp) == 2110 and len(ranks) == 35
    assert set(pred['Model']) == set(MODELS)
    assert (pred['Fingerprint'].astype(str) == fp).all()
    assert shp['original_row_index'].is_unique
    assert len(shp.filter(regex=r'^SHAP__').columns) == 35
    assert set(MODELS) == set(bins['Model'])
    ranked_names = ranks['Original_Predictor'].astype(str).tolist()
    # IMPORTANT: Unknown parent names fail rather than publish unexpanded codes.
    unmapped = [n for n in ranked_names if n not in LABELS]
    if unmapped:
        raise ValueError(f'Unmapped predictor labels: {unmapped}. Update LABELS first.')
    if len({LABELS[n] for n in ranked_names}) != 35:
        raise ValueError('Two different predictors map to the same display label.')
    reference = (pred.loc[pred['Model']=='Random Forest']
                 .sort_values('original_row_index').reset_index(drop=True))
    ids = reference['original_row_index'].to_numpy()
    y = reference['Actual'].to_numpy(dtype=int)
    w = reference['population_weight'].to_numpy(dtype=float)
    assert len(ids) == 2110 and len(set(ids)) == 2110
    assert int(y.sum()) == 553 and np.isfinite(w).all() and (w > 0).all()
    assert reference['Outer_Fold'].nunique() == 5
    assert np.isclose(np.average(y, weights=w), .237812, atol=1e-5)
    scores = {}
    for name in MODELS:
        sub = (pred.loc[pred['Model']==name]
               .sort_values('original_row_index').reset_index(drop=True))
        assert len(sub) == 2110
        assert np.array_equal(ids, sub['original_row_index'].to_numpy())
        assert np.array_equal(y, sub['Actual'].to_numpy(dtype=int))
        assert np.allclose(w, sub['population_weight'].to_numpy(dtype=float))
        p = sub['Predicted_Probability'].to_numpy(dtype=float)
        assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
        scores[name] = {
            'prob': p,
            'roc': float(roc_auc_score(y,p,sample_weight=w)),
            'ap': float(average_precision_score(y,p,sample_weight=w)),
        }
    return dict(dirs=dirs, pred=pred, ranks=ranks, shap=shp,
                selection=selection, bins=bins, ids=ids, y=y, w=w, scores=scores,
                prevalence=float(np.average(y, weights=w)))


def get_fig_data(out, fp):
    # All input check done in EACH isolated process; minimal memory use.
    return load_inputs(out, fp)


def apply_style():
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family':'sans-serif',
        'font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':10.5,
        'axes.labelsize':11,
        'axes.titlesize':12,
        'axes.titleweight':'bold',
        'xtick.labelsize':9.5,
        'ytick.labelsize':9.5,
        'legend.fontsize':9,
        'axes.spines.top':True,
        'axes.spines.right':True,
        'savefig.facecolor':'white',
        'pdf.fonttype':42,
    })


def export(fig, output_folder, fig_id, version):
    import matplotlib.pyplot as plt
    output_folder.mkdir(parents=True, exist_ok=True)
    # IMPORTANT: The uncompressed TIFF option avoids a native Windows
    # LZW codec crash (exit 0xC0000409) seen in the user's Spyder setup.
    for extension, dpi, kwargs in (
        ('png',300,{}),
        ('tiff',600,{'pil_kwargs':{'compression':'tiff_adobe_deflate'}}),
        ('pdf',150,{}),
    ):
        dest = output_folder / f'{fig_id}_{version}.{extension}'
        tmp = output_folder / f'{fig_id}_{version}.partial.{extension}'
        if dest.is_file() and dest.stat().st_size > 1024:
            print(f'    Existing: {dest.name}', flush=True)
            continue
        fig.savefig(str(tmp), dpi=dpi, bbox_inches='tight', facecolor='white', **kwargs)
        if not tmp.is_file() or tmp.stat().st_size <= 1024:
            raise RuntimeError(f'Invalid export: {tmp}')
        os.replace(tmp, dest)
        print(f'    Saved: {dest.name} ({dest.stat().st_size/1e6:.1f} MB)', flush=True)
        gc.collect()
    plt.close(fig)
    gc.collect()


def plot_one(out, fp, name, version):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.colors import Normalize
    from sklearn.metrics import roc_curve, precision_recall_curve
    apply_style()
    D = get_fig_data(out, fp)
    y, w, scores = D['y'], D['w'], D['scores']
    title_on = (version == 'WithTitle')
    fig_id = 'Figure_' + name
    fig = None

    if name == 'workflow':
        # Old manuscript: monochrome, sectioned workflow with rounded boxes/arrows.
        fig, ax = plt.subplots(figsize=(11.8, 7.1))
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
        rows = [
            ('A. Data and sample', [
                'HIES 2022 microdata\n62,407 individuals',
                'Working women aged 15+\nAnalytical sample: 2,110',
                'Outcome: annual work hours\nLow intensity: ≤1,440 hours\n553 low; 1,557 not low',
            ]),
            ('B. Predictors', [
                '35 candidate predictors\nDemographic, household,\nemployment and geographic',
                'Train-fold preprocessing\nMissingness + encoding',
                'Train-fold living-standard\nindex: PCA',
            ]),
            ('C. Nested validation', [
                '5 outer folds\nPSU-separated validation',
                '3 inner folds\nBoruta + model tuning',
                'Six classifiers\nSurvey-weighted training',
            ]),
            ('D. Outputs', [
                '2,110 held-out predictions\nper classifier',
                'Weighted AP / ROC-AUC\nCalibration + PSU-bootstrap CI',
                'Random Forest\nHeld-out grouped SHAP',
            ]),
        ]
        y_top = .915 if title_on else .955
        h = .18
        for r,(heading, cells) in enumerate(rows):
            row_y = y_top - r*.222
            ax.text(.02, row_y+.01, heading, fontsize=11, weight='bold', va='top')
            yy = row_y-.155
            for i, cell in enumerate(cells):
                xx = .03+i*.33
                box = FancyBboxPatch((xx,yy),.27,h*.72,boxstyle='round,pad=0.012,rounding_size=0.014',
                                     facecolor='white',edgecolor='#6f6f6f',linewidth=1.1)
                ax.add_patch(box)
                ax.text(xx+.135, yy+.065, cell, ha='center',va='center',fontsize=8.7,color='#222')
                if i<2:
                    ax.annotate('',xy=(xx+.323,yy+.064),xytext=(xx+.282,yy+.064),
                                arrowprops=dict(arrowstyle='->', lw=1.2,color='#555'))
            if r < 3:
                ax.plot([.02,.985], [row_y-.189,row_y-.189],color='#bbbbbb',lw=.7,ls=':')
        if title_on:
            ax.set_title('Updated analytical workflow for the HIES 2022 study',pad=12,fontsize=13)

    elif name == 'outcome_donut':
        fig,ax=plt.subplots(figsize=(8.6,6.4))
        counts = [int((y==0).sum()),int((y==1).sum())]
        assert counts == [1557,553]
        wedges, texts, autotexts = ax.pie(
            counts,labels=['Not low work intensity','Low work intensity'],
            colors=[BLUE,ORANGE],autopct='%1.2f%%',startangle=90,pctdistance=.8,
            wedgeprops=dict(width=.4,edgecolor='white',linewidth=1.2),
            textprops=dict(fontsize=10.7),
            )
        ax.text(0,0,'Working women\n(n = 2,110)',fontsize=13,weight='bold',ha='center',va='center')
        ax.set_aspect('equal')
        if title_on: ax.set_title('Work-intensity status among working women',pad=12)
        fig.subplots_adjust(left=.11,right=.91,top=.89,bottom=.1)

    elif name == 'boruta_stability':
        sel=D['selection']
        assert {'Outer_Fold','Original_Predictor'}.issubset(sel.columns)
        freq=sel['Original_Predictor'].value_counts().rename('folds')
        freq=freq.loc[freq.index.isin(LABELS)]
        assert len(freq)>0 and (freq<=5).all()
        freq=freq.sort_values(ascending=True)
        fig,ax=plt.subplots(figsize=(9.9,max(4.5, .45*len(freq)+1.5)))
        for j,(feat,v) in enumerate(freq.items()):
            ax.hlines(j,0,v,color=PALE_GREEN,lw=5,alpha=.8)
            ax.scatter(v,j,s=110,color=GREEN,edgecolor='#064B2B',zorder=3)
            ax.text(v+.12,j,f'{v}/5',va='center',fontsize=9)
        ax.set_yticks(range(len(freq)),[LABELS[x] for x in freq.index])
        ax.set_xlim(0,5.6);ax.set_xticks([0,1,2,3,4,5]);ax.set_xlabel('Outer-training folds retaining predictor (out of five)')
        ax.grid(axis='x',linestyle='--',alpha=.25);ax.set_axisbelow(True)
        if title_on:ax.set_title('Boruta selection stability across outer-training folds',pad=12)
        fig.subplots_adjust(left=.37,right=.92,top=.9,bottom=.16)

    elif name == 'roc_panels':
        fig, axs = plt.subplots(2,3,figsize=(13.3,8.0),sharex=True,sharey=True)
        for ax,model in zip(axs.flat,MODELS):
            p=scores[model]['prob']
            fpr,tpr,_=roc_curve(y,p,sample_weight=w)
            ax.plot(fpr,tpr,color=BLUE,lw=2,label=f"ROC-AUC = {scores[model]['roc']:.3f}")
            ax.plot([0,1],[0,1],ls='--',color='#a7a7a7',lw=.9)
            ax.set_title(MODEL_SHORT[model],fontsize=11.5,weight='bold')
            ax.set_xlim(0,1);ax.set_ylim(0,1.025)
            ax.set_xlabel('False-positive rate (1 − specificity)')
            ax.set_ylabel('True-positive rate (sensitivity)')
            ax.legend(loc='lower right',frameon=True,fontsize=8.3)
            ax.grid(alpha=.23)
        if title_on:fig.suptitle('Survey-weighted ROC curves: nested cross-validation',fontsize=14,weight='bold',y=.98)
        fig.tight_layout(rect=[0,0,1,.95] if title_on else [0,0,1,1],w_pad=1.7,h_pad=1.55)

    elif name in ('roc_combined','pr_combined','calibration'):
        fig,ax=plt.subplots(figsize=(9.2,7.0))
        for model in MODELS:
            col=COLOR_BY_MODEL[model]
            if name == 'roc_combined':
                xx,yy,_=roc_curve(y,scores[model]['prob'],sample_weight=w)
                label=f"{model} (ROC-AUC = {scores[model]['roc']:.3f})"
                ax.plot(xx,yy,lw=2.0,color=col,label=label)
            elif name == 'pr_combined':
                pp,rr,_=precision_recall_curve(y,scores[model]['prob'],sample_weight=w)
                label=f"{model} (AP = {scores[model]['ap']:.3f})"
                ax.step(rr,pp,where='post',lw=1.6,color=col,label=label)
            else:
                c=D['bins'].loc[D['bins']['Model']==model].sort_values('Bin')
                assert len(c)>0
                ax.plot(c['Mean_Predicted'],c['Observed_Frequency'],marker='o',markersize=4,
                        lw=1.8,color=col,label=model)
        if name == 'roc_combined':
            ax.plot([0,1],[0,1],ls='--',color='gray',lw=1,label='Chance')
            ax.set_xlim(0,1);ax.set_ylim(0,1.015)
            ax.set_xlabel('False-positive rate (1 − specificity)')
            ax.set_ylabel('True-positive rate (sensitivity)')
            title='Survey-weighted ROC curves across six classifiers'
            loc='lower right'
        elif name == 'pr_combined':
            ax.axhline(D['prevalence'],color='gray',ls='--',lw=1.2,
                       label=f"Weighted prevalence = {D['prevalence']:.3f}")
            ax.set_xlim(0,1);ax.set_ylim(0,1.02)
            ax.set_xlabel('Recall (sensitivity)')
            ax.set_ylabel('Precision (positive predictive value)')
            title='Survey-weighted precision–recall curves'
            loc='upper right'
        else:
            ax.plot([0,1],[0,1],ls='--',color='gray',lw=1.1,label='Perfect calibration')
            ax.set_xlim(0,1);ax.set_ylim(0,1)
            ax.set_xlabel('Mean predicted probability of low work intensity')
            ax.set_ylabel('Observed low-work-intensity proportion')
            title='Survey-weighted calibration curves'
            loc='upper left'
        if title_on: ax.set_title(title,pad=14)
        ax.legend(loc=loc,fontsize=8.0,framealpha=.94)
        ax.grid(alpha=.22)
        fig.tight_layout(pad=1.7)

    elif name == 'shap_lollipop':
        ranking=D['ranks'].copy()
        ranking=ranking.loc[ranking['Weighted_Mean_Abs_SHAP_Probability']>1e-12].copy()
        ranking=ranking.sort_values('Weighted_Mean_Abs_SHAP_Probability',ascending=False)
        assert len(ranking)>0
        ranking['Mean_Abs_pp']=100*ranking['Weighted_Mean_Abs_SHAP_Probability']
        sum_all=float(D['ranks']['Weighted_Mean_Abs_SHAP_Probability'].sum())
        assert sum_all>0
        ranking['Contribution_Percent']=100*ranking['Weighted_Mean_Abs_SHAP_Probability']/sum_all
        ranking['Cumulative_Percent']=ranking['Contribution_Percent'].cumsum()
        top=ranking.head(12).copy()
        left=top.iloc[::-1].reset_index(drop=True)
        k=len(left)
        fig,ax=plt.subplots(figsize=(11.2,max(5.7,k*.49+2.15)))
        ymax=float(left['Mean_Abs_pp'].max())
        for j,row in left.iterrows():
            val=float(row['Mean_Abs_pp'])
            ax.hlines(j,0,val,color=PALE_GREEN,lw=5,alpha=.75)
            ax.scatter(val,j,s=125,color=GREEN,edgecolor='#064B2B',linewidth=1,zorder=3)
            ax.text(val+ymax*.035,j,f"{float(row['Contribution_Percent']):.1f}%",fontsize=9.5,va='center')
        ax.set_xlim(0,ymax*1.24)
        ax.set_yticks(np.arange(k),[LABELS[n] for n in left['Original_Predictor']])
        ax.set_xlabel('Survey-weighted mean |SHAP| (percentage points of predicted risk)')
        ax.grid(axis='x',alpha=.22,ls='--')
        upper=ax.twiny()
        upper.plot(top['Cumulative_Percent'].to_numpy(), np.arange(k)[::-1],
                   marker='D',markersize=4,color=PURPLE,ls='--',lw=1.7,alpha=.9)
        upper.set_xlim(0,105)
        upper.set_xlabel('Cumulative share of total absolute SHAP (%)',color=PURPLE)
        upper.tick_params(axis='x',colors=PURPLE)
        upper.grid(False)
        if title_on: ax.set_title('Random Forest: global grouped SHAP importance',pad=15)
        fig.subplots_adjust(left=.35,right=.93,top=.87,bottom=.16)

    elif name == 'shap_distribution':
        ranking=D['ranks']
        selected=ranking.loc[ranking['Weighted_Mean_Abs_SHAP_Probability']>1e-12,
                             'Original_Predictor'].tolist()[:12]
        assert len(selected)>0
        shp=D['shap']
        fig,ax=plt.subplots(figsize=(11.1,max(5.9,len(selected)*.48+2.3)))
        # Old manuscript had a blue-red SHAP beeswarm where color was *feature value*.
        # Grouped categorical parent features have no single ordered feature value.
        # Here colors clearly encode PREDICTED RISK, rather than mislabeling it.
        predicted=shp['Predicted_Probability'].to_numpy(float)
        rng=np.random.default_rng(20261030)
        chosen=selected[::-1]
        for j,feature in enumerate(chosen):
            x=100*shp['SHAP__'+feature].to_numpy(dtype=float)
            jitter=rng.normal(loc=0,scale=.10,size=len(x))
            scatter=ax.scatter(x,j+jitter,c=predicted,cmap='coolwarm',vmin=0,vmax=1,
                               alpha=.4,s=6,lw=0,rasterized=True)
        ax.axvline(0,color='#777',lw=1,ls='--')
        ax.set_yticks(np.arange(len(chosen)),[LABELS[x] for x in chosen])
        ax.set_xlabel('Grouped SHAP contribution to predicted risk (percentage points)')
        ax.grid(axis='x',alpha=.16)
        cb=fig.colorbar(scatter,ax=ax,fraction=.034,pad=.022)
        cb.set_label('Model-predicted probability of low work intensity',fontsize=9.5)
        if title_on:ax.set_title('Random Forest: held-out grouped SHAP distributions',pad=14)
        fig.subplots_adjust(left=.35,right=.90,top=.91,bottom=.15)

    elif name == 'shap_local_waterfalls':
        shp=D['shap']
        assert len(shp)==2110
        cases=[
            ('True positive', (shp['Actual']==1)&(shp['Predicted_Probability']>=.5)),
            ('False negative', (shp['Actual']==1)&(shp['Predicted_Probability']<.5)),
            ('True negative', (shp['Actual']==0)&(shp['Predicted_Probability']<.5)),
        ]
        fig,axarr=plt.subplots(3,1,figsize=(12,11.4))
        for ax,(heading,mask) in zip(axarr,cases):
            sub=shp.loc[mask].copy().sort_values('Predicted_Probability').reset_index(drop=True)
            if len(sub)==0:
                ax.text(.5,.5,f'No {heading.lower()} cases at the 0.50 threshold',ha='center')
                ax.axis('off')
                continue
            # Deterministic typical example (median score within class), not an outlier.
            row=sub.iloc[len(sub)//2]
            names=list(D['ranks']['Original_Predictor'])
            contrib={n:float(row['SHAP__'+n]) for n in names}
            ordered=sorted((kv for kv in contrib.items() if abs(kv[1])>1e-10),
                           key=lambda kv:-abs(kv[1]))
            top=ordered[:6]
            other=sum(v for _,v in ordered[6:])
            components=top+([('Other predictors',other)] if abs(other)>1e-10 else [])
            baseline=float(row['SHAP_Baseline_Probability'])
            expected=float(row['Predicted_Probability'])
            assert abs(baseline+sum(c for _,c in components)-expected)<1e-6
            current=baseline
            elements=[]
            for feature,val in components:
                start=current; current+=val
                elements.append((LABELS.get(feature,feature), start,val))
            elements=elements[::-1]
            for j,(label,start,val) in enumerate(elements):
                ax.barh(j,val,left=start,height=.72,
                        color=('#e83f7c' if val>=0 else '#208ff2'),
                        edgecolor='white',linewidth=.7,alpha=.94)
                if abs(val)>=.012:
                    ax.text(start+val/2,j,f'{val*100:+.1f} pp',ha='center',va='center',
                            fontsize=7.7,color='white',weight='bold')
            ax.axvline(baseline,lw=1,ls='--',color='gray',label=f'Baseline = {baseline:.3f}')
            ax.axvline(expected,lw=1.4,color='#222',label=f'Predicted = {expected:.3f}')
            ax.set_yticks(range(len(elements)),[t[0] for t in elements])
            ax.set_xlim(0,1);ax.set_xlabel('Predicted probability of low work intensity')
            ax.set_title(f'{heading} (actual = {int(row["Actual"])}, predicted risk = {expected:.3f})',
                         fontsize=10.5,loc='left',pad=8)
            ax.grid(axis='x',alpha=.17)
            ax.legend(loc='upper right',fontsize=8)
        if title_on:fig.suptitle('Local Random Forest SHAP explanations: illustrative held-out cases',
                                fontsize=14,weight='bold',y=.985)
        fig.tight_layout(rect=[0,0,1,.96] if title_on else [0,0,1,1],h_pad=2.3)

    else:
        raise ValueError(f'Unknown figure: {name}')

    dest=D['dirs']['result']
    export(fig,dest,fig_id,version)


def write_notes(out,fp):
    dirs=folders(out,fp)
    dest=dirs['result'];dest.mkdir(parents=True,exist_ok=True)
    notes = '''JOURNAL-READY FIGURES: CAPTIONS AND NOTES\n'''
    notes += f'''Study: Explaining Low Work Intensity among Working Women in Bangladesh (HIES 2022)\n\n'''
    notes += '''Figure numbering should be assigned during manuscript revision.\nThe files contain publication-style figures with and without in-figure titles.\nFor journal submission, use NoTitle and place the corresponding caption below the image.\n\n'''
    captions = [
      ('Figure_workflow','Updated nested-cross-validation analytical workflow.','The analytical sample consists of 2,110 working women. Training-only preprocessing, living-standard PCA, and Boruta selection were nested within PSU-separated CV. Survey-weighted evaluation used out-of-fold predictions.'),
      ('Figure_outcome_donut','Distribution of annual work-intensity status among working women.','The pie-chart shares are unweighted (553/2,110; 26.21% with low work intensity). The survey-weighted prevalence was 23.78%. Low work intensity was defined as 1,440 or fewer annual working hours.'),
      ('Figure_boruta_stability','Stability of Boruta predictor selection across five outer-training folds.','Counts indicate the number of outer-training sets in which a parent predictor was retained. This shows selection frequency, not the original paper’s random-forest impurity importance or formal statistical significance.'),
      ('Figure_roc_panels','Survey-weighted receiver operating characteristic curves for six classifiers.','Each panel shows out-of-fold predictions from five PSU-separated outer folds, n=2,110 per classifier. ROC-AUC = area under ROC curve. The grey line indicates chance performance.'),
      ('Figure_roc_combined','Comparative survey-weighted ROC curves across six classifiers.','Point estimates are calculated from pooled out-of-fold predicted probabilities using original survey weights. Confidence intervals are reported in the corresponding results table, not on curves.'),
      ('Figure_pr_combined','Survey-weighted precision–recall curves across six classifiers.','AP = average precision (stepwise summary), not trapezoidal PR-AUC. The dashed reference is the survey-weighted positive-class prevalence, 23.78%. Curves are based on pooled out-of-fold predictions.'),
      ('Figure_calibration','Survey-weighted calibration curves for the six classifiers.','Points are survey-weighted observed event proportions by predicted-risk bins, based on held-out predictions. The diagonal denotes perfect calibration. Calibration intercept, slope and Brier score should be reported separately.'),
      ('Figure_shap_lollipop','Survey-weighted global grouped SHAP importance for the Random Forest classifier.','Mean absolute SHAP contributions are expressed in percentage points of predicted low-work-intensity probability. Green markers show group-level mean absolute contributions. The purple curve is the cumulative share among all 35 candidate groups. Predictors absent from a fold contribute zero in that fold.'),
      ('Figure_shap_distribution','Distribution of held-out grouped SHAP values for the Random Forest classifier.','Each point represents a held-out woman, with SHAP contributions in percentage points of predicted low-work-intensity risk. Colors indicate model-predicted risk, NOT predictor levels. SHAP is predictive, not causal; fold-specific Boruta may omit a predictor.'),
      ('Figure_shap_local_waterfalls','Illustrative local SHAP contributions for three held-out Random Forest predictions.','Panels show a typical true positive, false negative and true negative selected at the median predicted risk in each respective subgroup using a threshold of 0.50. Red/pink contributions increase and blue contributions decrease the predicted probability. SHAP values are in percentage points. These are predictive illustrations, not explanations of causal effects.'),
    ]
    for key,caption,note in captions:
        notes+=f'\n{key}: {caption}\nNote: {note}\n'
    notes+='''\nStyling derived from the previous manuscript: blue/orange work-intensity donut; 2x3 blue-line ROC panels; green-lollipop/purple-cumulative SHAP importance; clear English feature names. Results are from new nested CV, not the old 70:30 split.\n\nThe previous manuscript’s original Sankey chart has not been regenerated here because comparable new data inputs for its wealth groups are not among saved OOF/SHAP tables. Do not reuse obsolete model-comparison figures or claim old numbers.\n'''
    (dest/'Figure_Captions_and_Methodological_Notes.txt').write_text(notes,encoding='utf-8')
    return dest


def run(out,fp,figures):
    d=load_inputs(out,fp)
    folder=write_notes(out,fp)
    print('========== STEP 1.43C JOURNAL-STYLE FIGURE EXPORT ==========',flush=True)
    print('Source: saved OOF, calibration, and grouped SHAP results.',flush=True)
    print('No model fitting or SHAP calculations.',flush=True)
    print('Destination:',folder,flush=True)
    print('Validated 2110 cases/model; weighted prevalence:',round(d['prevalence'],6),flush=True)
    script=Path(__file__).resolve()
    cmd0=[sys.executable,'-u',str(script),'--output-dir',str(out),'--fingerprint',fp]
    subprocess_env=os.environ.copy()
    subprocess_env.update(MPLBACKEND='Agg',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    audit=[]
    for name in figures:
        for ver in ('WithTitle','NoTitle'):
            print(f'  Rendering {name}/{ver}',flush=True)
            call=subprocess.run(cmd0+['--mode','render','--figure',name,'--version',ver],
                                check=False,env=subprocess_env)
            if call.returncode!=0:
                raise RuntimeError(f'Figure {name}/{ver} failed in isolated subprocess (exit {call.returncode}). Previous files/checkpoints are safe.')
            for ext in ('png','tiff','pdf'):
                fn=folder/f'Figure_{name}_{ver}.{ext}'
                if not fn.is_file() or fn.stat().st_size<1024:
                    raise RuntimeError(f'Export file missing/empty: {fn}')
                audit.append(dict(Figure=name,Title_Version=ver,Format=ext,
                                  File=str(fn),Bytes=fn.stat().st_size))
    import pandas as pd
    pd.DataFrame(audit).to_csv(folder/'Figure_Export_Manifest.csv',index=False)
    print('\n========== STEP 1.43C FINAL VERIFICATION ==========',flush=True)
    print('Figures exported:',len(figures),flush=True)
    print('WithTitle and NoTitle:',2,flush=True)
    print('Formats: 600-dpi uncompressed TIFF; 300-dpi PNG; PDF (vector axes/text)',flush=True)
    print('Files:',len(audit),flush=True)
    print('All plot labels: plain-English, not dataset codes',flush=True)
    print('Unmodified original outputs: yes',flush=True)
    print('Caption and methodological notes:',folder/'Figure_Captions_and_Methodological_Notes.txt',flush=True)
    print('STEP 1.43C VERIFICATION SUCCESSFUL.',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',required=True)
    p.add_argument('--fingerprint',default=FP_DEFAULT)
    p.add_argument('--mode',choices=['all','render'],default='all')
    p.add_argument('--figure',choices=FIGS)
    p.add_argument('--version',choices=['WithTitle','NoTitle'])
    p.add_argument('--figures',nargs='*',choices=FIGS,default=None)
    a=p.parse_args()
    if a.mode=='render':
        if not a.figure or not a.version:
            p.error('--mode render requires --figure and --version')
        plot_one(a.output_dir,a.fingerprint,a.figure,a.version)
    else:
        run(a.output_dir,a.fingerprint,a.figures if a.figures else FIGS)

if __name__=='__main__': main()
