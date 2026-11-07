# %% STEP 1.48 — READ-ONLY JOURNAL EVIDENCE AUDIT (NO MODEL REFITTING)
from pathlib import Path
import os
import json
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
root = Path(os.environ.get("HIES_OUTPUT_DIR", str(project_root / "outputs")))
fp = '7244d907321a2e433a44'
run = 'Run_68763c127e9a15d939a1'
eval_dir = root / f'Step1_42_Calibration_Uncertainty_{fp}'
shap_dir = root / f'Step1_43_RF_OuterHoldout_SHAP_{fp}'
figure_dir = root / f'Step1_43C_Journal_Style_Figures_{fp}'
sens_dir = root / f'Step1_46_FullyNested_RF_Sensitivity_{fp}' / run / 'Step1_47_Paired_PSU_Bootstrap'
sens_fig_dir = sens_dir / 'Figures_Journal_Style'
pack_dir = root / f'Step1_48_Journal_Results_Pack_{fp}'
pack_dir.mkdir(parents=True, exist_ok=True)

sources = {
    'Model metrics': eval_dir / 'Step1_42_Main_Performance_95CI.csv',
    'SHAP ranking': shap_dir / 'Step1_43_RF_Weighted_GroupSHAP_Importance.csv',
    'Sensitivity metrics': sens_dir / 'Step1_47_Scenario_Metrics_95CI.csv',
    'Paired contrasts': sens_dir / 'Step1_47_Paired_Performance_Loss_95CI.csv',
}
for key, p in sources.items():
    assert p.is_file(), f'Missing {key}: {p}'

main = pd.read_csv(sources['Model metrics'])
shap = pd.read_csv(sources['SHAP ranking'])
sens = pd.read_csv(sources['Sensitivity metrics'])
contrasts = pd.read_csv(sources['Paired contrasts'])

models = {
    'Logistic Regression', 'Support Vector Machine', 'Random Forest',
    'Extra Trees', 'Gradient Boosting', 'AdaBoost'
}
assert len(main) == 6 and set(main['Model']) == models
assert len(shap) == 35 and shap['Original_Predictor'].is_unique
assert len(sens) == 9 and set(sens['Scenario']) == {
    'Full_35', 'Remove_Core_3', 'Remove_All_Employment_6'
}
assert len(contrasts) == 6 and contrasts['Bootstrap_Replicates'].eq(1000).all()
assert sens['Bootstrap_Replicates'].eq(1000).all()

for metric in ('Weighted_AP', 'Weighted_ROC_AUC', 'Weighted_Brier'):
    assert metric in main.columns
    for suffix in ('_CI_Lower', '_CI_Upper'):
        assert metric + suffix in main.columns
    a = main[metric + '_CI_Lower']
    b = main[metric]
    c = main[metric + '_CI_Upper']
    assert ((a <= b) & (b <= c)).all(), f'Invalid CI: {metric}'

rf = main.set_index('Model').loc['Random Forest']
assert abs(float(rf['Weighted_AP']) - 0.49894) < 0.00002
assert abs(float(rf['Weighted_ROC_AUC']) - 0.77145) < 0.00002
assert abs(float(rf['Weighted_Brier']) - 0.14907) < 0.00002

# Generate publication-ready table of the six original classifiers.
main_cols = ['Model', 'Weighted_AP', 'Weighted_AP_CI_Lower', 'Weighted_AP_CI_Upper',
             'Weighted_ROC_AUC', 'Weighted_ROC_AUC_CI_Lower', 'Weighted_ROC_AUC_CI_Upper',
             'Weighted_Brier', 'Weighted_Brier_CI_Lower', 'Weighted_Brier_CI_Upper',
             'Calibration_Intercept', 'Calibration_Slope', 'Calibration_In_Large']
assert set(main_cols).issubset(main.columns)
model_table = main[main_cols].sort_values('Weighted_AP', ascending=False)
model_table.to_csv(pack_dir / 'Table_ML_Performance_Calibration.csv', index=False)

# Sensitivity table plus the paired loss estimates.
sens_wide = sens.pivot(index='Scenario', columns='Metric',
                       values=['Estimate', 'Lower_95_CI', 'Upper_95_CI'])
sens_wide.columns = ['_'.join(x) for x in sens_wide.columns]
sens_wide = sens_wide.reset_index()
sens_wide.to_csv(pack_dir / 'Table_FullyNested_Sensitivity_CI.csv', index=False)
contrasts.to_csv(pack_dir / 'Table_Paired_Sensitivity_Loss_CI.csv', index=False)

# Human-readable labels for SHAP predictors appearing in manuscript tables.
label_map = {
    'earn_emp_status_nonag': 'Non-agricultural employment status',
    'earn_org_type_grp': 'Organization/workplace type',
    'division_code': 'Administrative division',
    'age': 'Age',
    'wealth_living_index': 'Household living-standard index',
    'earn_emp_status_ag': 'Agricultural employment status',
    'fies_raw_score': 'Food insecurity (FIES score)'
}
shap_export = shap.copy()
shap_export.insert(1, 'Readable_Label',
                   shap_export['Original_Predictor'].map(label_map).fillna(
                       shap_export['Original_Predictor']))
shap_export.to_csv(pack_dir / 'Table_RF_SHAP_Importance_All35.csv', index=False)

# Verify existence and nonempty size of journal-style figures.
manifest = []
for group, folder, expected_per_format in [
    ('Main paper figures', figure_dir, 20),
    ('Sensitivity figure', sens_fig_dir, 2),
]:
    for extension in ('png', 'tiff', 'pdf'):
        fs = sorted(folder.glob('*.' + extension))
        assert len(fs) == expected_per_format, (group, extension, len(fs), str(folder))
        assert all(f.stat().st_size > 1024 for f in fs)
        for f in fs:
            manifest.append({'Group': group, 'Format': extension.upper(),
                             'Filename': f.name, 'Full_Path': str(f),
                             'Size_MB': round(f.stat().st_size / 1e6, 3)})
pd.DataFrame(manifest).to_csv(pack_dir / 'Figure_Inventory_66_Files.csv', index=False)
assert len(manifest) == 66

notes = (
    'MANUSCRIPT REVISION — MANDATORY CONSISTENCY CHECKS\n\n'
    'Primary models: six original classifiers, five outer PSU-separated CV folds, '
    'three inner PSU-separated folds; train-only preprocessing, PCA, Boruta and tuning.\n'
    'Primary evaluation uses original survey weights and pooled out-of-fold predictions.\n'
    'Use Average Precision (AP), NOT trapezoidal PR-AUC, for the reported AP values.\n'
    'Random Forest is the best observed primary model on AP, ROC-AUC, and Brier; '
    'its superiority is not established against all other models on every metric.\n'
    'The sensitivity analysis has a newly rebuilt 35-feature baseline; compare only '
    'against that new baseline, not against the primary RF fit.\n'
    'Sensitivity pairs use the same PSU bootstrap draws; uncertainty is approximate '
    'and conditional on fixed OOF predictions, without model-refitting uncertainty.\n'
    'AP and Brier paired intervals exclude zero for both exclusions; ROC-AUC intervals include zero.\n'
    'SHAP is based on held-out Random Forest predictions and grouped feature attributions, '
    'not causal effects. Employment structural missingness requires cautious discussion.\n'
    'Replace obsolete original-manuscript claims: 70:30 random split, global Boruta 12 features, '
    'Gradient Boosting as best with ROC-AUC 0.803, and old SHAP rankings.\n'
    'Figure naming and numbering must be reconciled with final manuscript captions; '
    'NoTitle TIFF 600 DPI is suitable for journal figure submission.\n'
)
(pack_dir / 'Mandatory_Manuscript_Revision_Notes.txt').write_text(notes, encoding='utf-8')
print('========== STEP 1.48 FINAL EVIDENCE AUDIT ==========')
print('Primary models:', len(main))
print('SHAP parent predictors:', len(shap))
print('Sensitivity scenario metric rows:', len(sens))
print('Paired sensitivity comparisons:', len(contrasts))
print('Figure files verified:', len(manifest), '(60 main + 6 sensitivity)')
print('\nPRIMARY MODEL TABLE (TOP 6):')
print(model_table[['Model', 'Weighted_AP', 'Weighted_ROC_AUC', 'Weighted_Brier']]
      .round(5).to_string(index=False))
print('\nPAIRED SENSITIVITY LOSSES:')
print(contrasts[['Contrast','Metric','Performance_Loss','Lower_95_CI','Upper_95_CI']]
      .round(5).to_string(index=False))
print('\nSaved journal evidence pack:', pack_dir)
print('STEP 1.48 VERIFICATION SUCCESSFUL.')
