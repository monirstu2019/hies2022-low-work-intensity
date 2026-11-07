# %% STEP 1.46: FULLY NESTED RANDOM FOREST EMPLOYMENT-STRUCTURE SENSITIVITY
# Standalone / safe in a fresh Spyder kernel. Reads only original Stata data,
# saved Step 1.40 INNER split indices and Step 1.41 OUTER OOF membership.
# IMPORTANT: Reconstructs a specified common preprocessing pipeline for EACH
# scenario. It is a NEW matched sensitivity experiment; historical Step 1.41
# RF metrics are contextual only and are NOT its within-experiment comparator.
# No outer-validation scores are used to tune, impute, fit PCA, select features.
# Save checkpoints after every inner-fold selection, and each outer prediction.

from pathlib import Path
import json
import os
import hashlib
import gc
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from boruta import BorutaPy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("HIES_OUTPUT_DIR", str(PROJECT_ROOT / "outputs")))
FP = '7244d907321a2e433a44'
STATA = Path(os.environ.get("HIES_MERGED_DATA", str(PROJECT_ROOT / "restricted_data" / "final_hies_women_lfp_earnings_ml_ready.dta")))
FOLDER_140 = ROOT / f'Step1_40_Remaining_Tuning_{FP}'
FOLDER_141 = ROOT / f'Step1_41_Nested_Outer_Predictions_{FP}'
OUTDIR = ROOT / f'Step1_46_FullyNested_RF_Sensitivity_{FP}'
OUTDIR.mkdir(exist_ok=True, parents=True)
VERSION = 'HIES22_Sensitivity_Nested_RF_v1_PCA_Boruta_20261030'
SEED = 20261030
RF_TREES = 400
BORUTA_TREES = 160
BORUTA_ITERATIONS = 40
CATEGORIES_MIN_FREQUENCY = 15

# Model covariates are the same 35 conceptual groups as the primary analysis.
CATEGORICAL = [
    'marital_status', 'relation_head_grp', 'literate_basic',
    'education_attainment_group', 'has_mobile', 'has_bank_account',
    'has_mfs_account', 'internet_used', 'has_computer',
    'any_functional_difficulty', 'rural_urban', 'division_code',
    'rmo_code', 'domain16_grp', 'toilet_type_grp',
    'drinking_water_source_grp', 'light_source', 'cooking_stove_type',
    'cooking_fuel_grp', 'transport_stop_halfkm',
    'transport_stop_time_min_missing', 'regular_waste_collection',
    'slum_household', 'earn_sector', 'earn_emp_status_ag',
    'earn_emp_status_nonag', 'earn_org_type_grp',
    'earn_ind_code_grp', 'earn_occ_code_grp'
]
NUMERIC = [
    'age', 'hh_MEMBER', 'fies_raw_score',
    'electricity_hours_24h', 'transport_stop_time_min'
]
WEALTH = [
    'hh_rooms_pca', 'covered_area', 'house_value',
    'land_property_value', 'other_asset_value',
    'sec9a1_food_value', 'sec9b2_value_total',
    'sec9c_total_value', 'sec9d1_value_total',
    'sec9d2_value_total', 'durable_owned_count',
    'durable_total_number', 'durable_total_value',
    'sec9_total_observed_value'
]
ALL_GROUPS = CATEGORICAL + NUMERIC + ['wealth_living_index']
assert len(CATEGORICAL) == 29 and len(NUMERIC) == 5
assert len(WEALTH) == 14 and len(ALL_GROUPS) == 35

# Remove only the three main structural/skip-pattern predictors, and then
# all six employment-structure predictor groups (pre-specified sensitivity).
CORE_3 = {'earn_emp_status_nonag', 'earn_emp_status_ag', 'earn_org_type_grp'}
EMPLOYMENT_6 = CORE_3 | {
    'earn_sector', 'earn_ind_code_grp', 'earn_occ_code_grp'
}
SCENARIOS = [
    ('Full_35', set()),
    ('Remove_Core_3', CORE_3),
    ('Remove_All_Employment_6', EMPLOYMENT_6),
]
GRID = list(ParameterGrid({
    'max_depth': [None, 8, 16],
    'min_samples_leaf': [2, 8],
    'max_features': ['sqrt', 0.7],
}))
assert len(GRID) == 12


def atomic_csv(df, destination):
    tmp = destination.with_suffix(destination.suffix + '.tmp')
    df.to_csv(tmp, index=False, float_format='%.17g')
    os.replace(tmp, destination)


def atomic_npz(destination, **arrays):
    tmp = destination.with_suffix(destination.suffix + '.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, destination)


# Use 1%/99% clipping, nonnegative log1p, median imputation, standardization,
# PCA=1. Every statistic learned ONLY from the current training partition.
class WealthPCATransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    @staticmethod
    def to_numeric(x):
        a = np.asarray(x, dtype=float).copy()
        a[~np.isfinite(a)] = np.nan
        a[a < 0] = np.nan
        return a

    def fit(self, X, y=None):
        raw = self.to_numeric(X)
        self.lower_ = np.zeros(raw.shape[1], dtype=float)
        self.upper_ = np.zeros(raw.shape[1], dtype=float)
        for c in range(raw.shape[1]):
            valid = raw[:, c][np.isfinite(raw[:, c])]
            if len(valid):
                self.lower_[c] = np.percentile(valid, 1)
                self.upper_[c] = np.percentile(valid, 99)
            else:
                self.lower_[c] = 0
                self.upper_[c] = 0
        clipped = np.minimum(np.maximum(raw, self.lower_), self.upper_)
        logged = np.log1p(clipped)
        self.imputer_ = SimpleImputer(strategy='median', keep_empty_features=True)
        mat = self.imputer_.fit_transform(logged)
        self.scaler_ = StandardScaler().fit(mat)
        self.pca_ = PCA(n_components=1, random_state=SEED).fit(
            self.scaler_.transform(mat)
        )
        pc = self.pca_.transform(self.scaler_.transform(mat)).reshape(-1)
        reference = np.log1p(np.maximum(
            np.nan_to_num(raw[:, WEALTH.index('house_value')], nan=0), 0
        ))
        if np.std(pc) > 0 and np.std(reference) > 0:
            corr = np.corrcoef(pc, reference)[0, 1]
            self.sign_ = -1.0 if np.isfinite(corr) and corr < 0 else 1.0
        else:
            self.sign_ = 1.0
        return self

    def transform(self, X):
        raw = self.to_numeric(X)
        logged = np.log1p(np.minimum(np.maximum(
            raw, self.lower_), self.upper_
        ))
        mat = self.imputer_.transform(logged)
        values = self.pca_.transform(self.scaler_.transform(mat))
        return self.sign_ * values

    def get_feature_names_out(self, input_features=None):
        return np.array(['wealth_living_index'], dtype=object)


class BorutaWeightedForest(RandomForestClassifier):
    # BorutaPy calls estimator.fit(X, y) without sample_weight; this adapter
    # ensures original training-only survey weights enter each Boruta RF fit.
    def fit(self, X, y, sample_weight=None):
        w = getattr(self, '_survey_weights', None)
        return super().fit(X, y, sample_weight=(w if sample_weight is None else sample_weight))


def new_preprocessor(allowed_groups):
    cats = [g for g in CATEGORICAL if g in allowed_groups]
    nums = [g for g in NUMERIC if g in allowed_groups]
    return ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(
                handle_unknown='infrequent_if_exist',
                min_frequency=CATEGORIES_MIN_FREQUENCY,
                sparse_output=False
            ), cats),
            ('num', Pipeline([
                ('impute', SimpleImputer(
                    strategy='median', keep_empty_features=True
                )),
                ('scale', StandardScaler())
            ]), nums),
            ('pca', WealthPCATransformer(), WEALTH)
        ], remainder='drop', verbose_feature_names_out=True,
        sparse_threshold=0
    )


def feature_parent(feature_name, active_groups):
    tail = str(feature_name).split('__', 1)[-1]
    matches = [g for g in active_groups if tail == g or tail.startswith(g + '_')]
    if not matches:
        raise ValueError(f'Cannot map {feature_name!r} to parent in {active_groups}')
    return max(matches, key=len)


def selection_fit(x_train, y_train, weights, x_val, excluded):
    allowed = [g for g in ALL_GROUPS if g not in excluded]
    prep = new_preprocessor(set(allowed))
    Xtr_all = np.asarray(prep.fit_transform(x_train), dtype=np.float32)
    Xval_all = np.asarray(prep.transform(x_val), dtype=np.float32)
    names = list(prep.get_feature_names_out())
    assert Xtr_all.shape == (len(y_train), len(names))
    assert Xval_all.shape == (len(x_val), len(names))
    assert np.isfinite(Xtr_all).all() and np.isfinite(Xval_all).all()
    parents = [feature_parent(n, allowed) for n in names]
    assert set(parents) == set(allowed)

    estimator = BorutaWeightedForest(
        n_estimators=BORUTA_TREES,
        max_depth=5, min_samples_leaf=3,
        random_state=SEED, n_jobs=-1
    )
    estimator._survey_weights = weights / weights.mean()
    selector = BorutaPy(
        estimator,
        n_estimators=BORUTA_TREES,
        max_iter=BORUTA_ITERATIONS,
        perc=100, alpha=0.05,
        random_state=SEED, verbose=0
    )
    selector.fit(Xtr_all, y_train.astype(int))
    confirmed_groups = set(np.asarray(parents)[selector.support_].tolist())
    fallback = False
    if not confirmed_groups:
        # Pre-defined zero-confirmation fallback avoids invalid zero-column RF.
        # Choose up to three parent groups with smallest minimum dummy-level rank.
        ranks = {g: min(int(selector.ranking_[i]) for i, p in enumerate(parents)
                        if p == g) for g in allowed}
        selected_groups = sorted(ranks, key=lambda g: (ranks[g], allowed.index(g)))[:3]
        fallback = True
    else:
        selected_groups = [g for g in allowed if g in confirmed_groups]
    retain = [i for i, parent in enumerate(parents) if parent in selected_groups]
    assert len(retain) > 0
    return (Xtr_all[:, retain], Xval_all[:, retain],
            selected_groups, len(retain), len(names), fallback)


# Reconstruct the 48 raw model columns from the original 324-variable Stata.
# Both the new full and reduced models use IDENTICAL preparation and folds.
def reconstruct_source(raw):
    d = raw.loc[raw['sample_earnings_women_15plus'] == 1].copy()
    assert len(d) == 2110
    aliases = {
        'relation_head_grp': 'relation_head',
        'domain16_grp': 'domain16',
        'toilet_type_grp': 'toilet_type',
        'drinking_water_source_grp': 'drinking_water_source',
        'cooking_fuel_grp': 'cooking_fuel',
        'earn_org_type_grp': 'earn_org_type',
        'earn_ind_code_grp': 'earn_ind_code',
        'earn_occ_code_grp': 'earn_occ_code',
    }
    for new, old in aliases.items():
        d[new] = d[old]
    d['hh_rooms_pca'] = d['hh_rooms']
    d['transport_stop_time_min_missing'] = d['transport_stop_time_min'].isna().astype(int)
    completed = pd.to_numeric(d['highest_class_completed'], errors='coerce')
    educ = np.where(completed == 99, 'Other specified',
                    np.where(completed.isna(),
                             'No reported completed class',
                             completed.fillna(0).astype(int).astype(str)))
    d['education_attainment_group'] = educ
    # Do not let codes act as continuous predictors for categorical variables.
    for c in CATEGORICAL:
        v = d[c]
        if c == 'education_attainment_group':
            d[c] = v.astype(str)
        else:
            d[c] = v.map(
                lambda q: '-99' if pd.isna(q) else (
                    str(int(q)) if isinstance(q, (float, int, np.integer,
                                                  np.floating)) and float(q).is_integer()
                    else str(q)
                )
            ).astype(str)
    for c in NUMERIC + WEALTH:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    return d


def test_group_separation(tr, va):
    assert len(set(tr['original_row_index']) & set(va['original_row_index'])) == 0
    trpsu = set(zip(tr['stratum16'], tr['psu']))
    vpsu = set(zip(va['stratum16'], va['psu']))
    assert not (trpsu & vpsu)


def model_metrics(df):
    y = df['Actual'].to_numpy(dtype=int)
    p = df['Predicted_Probability'].to_numpy(dtype=float)
    w = df['population_weight'].to_numpy(dtype=float)
    return {
        'N': len(df),
        'Weighted_AP': float(average_precision_score(y, p, sample_weight=w)),
        'Weighted_ROC_AUC': float(roc_auc_score(y, p, sample_weight=w)),
        'Weighted_Brier': float(np.average((p-y)**2, weights=w)),
        'Mean_Predicted_Risk': float(np.average(p, weights=w))
    }


def main():
    assert STATA.is_file()
    reference = FOLDER_141 / 'Step1_41_All_Six_Model_OOF_Predictions.csv'
    assert reference.is_file()
    ref = pd.read_csv(reference)
    ref = ref.loc[ref['Model'] == 'Random Forest'].copy()
    assert len(ref) == 2110 and ref['original_row_index'].is_unique
    assert ref['Outer_Fold'].nunique() == 5
    assert (ref['Fingerprint'] == FP).all()
    ref = ref.sort_values('original_row_index').reset_index(drop=True)
    raw = pd.read_stata(STATA, convert_categoricals=False)
    data = reconstruct_source(raw)
    # Strict alignment is anchored on Stata original row number, not arbitrary
    # row order within the analytic subset.
    data['original_row_index'] = data.index
    data = data.set_index('original_row_index', verify_integrity=True)
    assert set(data.index) == set(ref['original_row_index'])
    data = data.loc[ref['original_row_index'].to_numpy()]
    assert np.array_equal(
        (data['earn_annual_hours'].to_numpy(dtype=float) <= 1440).astype(int),
        ref['Actual'].to_numpy(dtype=int)
    )
    assert np.allclose(data['population_weight'], ref['population_weight'],
                       rtol=1e-5, atol=1e-5)
    assert np.array_equal(data['stratum16'].to_numpy(), ref['stratum16'].to_numpy())
    assert np.array_equal(data['psu'].to_numpy(), ref['psu'].to_numpy())
    rows_by_id = ref.set_index('original_row_index', verify_integrity=True)
    features = data[CATEGORICAL + NUMERIC + WEALTH]
    assert features.shape == (2110, 48)
    assert (ref['population_weight'] > 0).all()

    # Verify the original exact 15 inner fold boundaries from the Step 1.35
    # and 1.40 caches. We read ONLY indices, never Boruta-selected columns.
    folds = {}
    for outer in range(1, 6):
        oof_outer = rows_by_id.index[rows_by_id['Outer_Fold'] == outer].to_numpy()
        expected_train = set(rows_by_id.index) - set(oof_outer)
        for inner in range(1, 4):
            if outer == 1:
                cache = ROOT / 'Step1_35_Inner_Fold_Cache' / f'Outer1_Inner{inner}_Boruta_Matrices.npz'
            else:
                cache = FOLDER_140 / 'inner_fold_matrices' / f'Outer{outer}_Inner{inner}_Boruta.npz'
            assert cache.is_file(), cache
            with np.load(cache, allow_pickle=False) as z:
                itrain, ival = z['train_idx'].copy(), z['val_idx'].copy()
            assert set(itrain) | set(ival) == expected_train
            assert not set(itrain) & set(ival)
            test_group_separation(
                rows_by_id.loc[itrain].reset_index(),
                rows_by_id.loc[ival].reset_index()
            )
            folds[(outer, inner)] = (itrain, ival)

    # Stable, data-sensitive fingerprint for resume-safety; never share
    # checkpoints with a different dataset, preprocessing or parameter grid.
    h = hashlib.sha256()
    h.update(VERSION.encode())
    h.update(FP.encode())
    h.update(json.dumps(GRID, sort_keys=True).encode())
    h.update(pd.util.hash_pandas_object(features, index=True).values.tobytes())
    h.update(pd.util.hash_pandas_object(ref[['Outer_Fold','Actual','population_weight']],
                                        index=False).values.tobytes())
    run_hash = h.hexdigest()[:20]
    work = OUTDIR / f'Run_{run_hash}'
    work.mkdir(parents=True, exist_ok=True)
    print('========== STEP 1.46 FULLY NESTED RF SENSITIVITY ==========', flush=True)
    print('Three matched sensitivity scenarios, 5 PSU-separated outer folds.', flush=True)
    print('12 RF candidates x 3 inner folds per outer and scenario.', flush=True)
    print('All preprocessing, PCA, Boruta, tuning refitted on training only.', flush=True)
    print('Run hash:', run_hash, flush=True)
    print('Output:', work, flush=True)

    prediction_parts = []
    for scenario, excluded in SCENARIOS:
        groups_expected = 35-len(excluded)
        print('\nSCENARIO:', scenario, '| candidate groups:', groups_expected, flush=True)
        sc_dir = work / scenario
        sc_dir.mkdir(parents=True, exist_ok=True)
        for outer in range(1, 6):
            fold_dir = sc_dir / f'Outer{outer}'
            fold_dir.mkdir(exist_ok=True)
            outer_pred_file = fold_dir / 'Heldout_OOF_Predictions.csv'
            validx = rows_by_id.index[rows_by_id['Outer_Fold'] == outer].to_numpy()
            tridx = rows_by_id.index[rows_by_id['Outer_Fold'] != outer].to_numpy()
            test_group_separation(rows_by_id.loc[tridx].reset_index(),
                                  rows_by_id.loc[validx].reset_index())

            if outer_pred_file.is_file():
                saved = pd.read_csv(outer_pred_file)
                assert (saved['Run_Hash'] == run_hash).all()
                assert (saved['Scenario'] == scenario).all()
                assert (saved['Outer_Fold'] == outer).all()
                assert set(saved['original_row_index']) == set(validx)
                assert len(saved) == len(validx)
                assert saved['Predicted_Probability'].between(0, 1).all()
                prediction_parts.append(saved)
                print(f'  Outer {outer}: completed checkpoint reused.', flush=True)
                continue

            fold_records = []
            for inner in range(1, 4):
                inner_file = fold_dir / f'Inner{inner}_Boruta_Matrices.npz'
                itrain, ival = folds[(outer, inner)]
                if inner_file.exists():
                    with np.load(inner_file, allow_pickle=False) as z:
                        assert z['Run_Hash'].item() == run_hash
                        assert np.array_equal(z['Train_Index'], itrain)
                        assert np.array_equal(z['Val_Index'], ival)
                        Xtr = z['Xtrain'].copy()
                        Xva = z['Xval'].copy()
                        selected = list(z['Selected_Groups'].tolist())
                        fallback = bool(z['Fallback'].item())
                    print(f'    Inner {inner}: cached new selection, groups={selected}', flush=True)
                else:
                    trr = rows_by_id.loc[itrain]
                    Xtr, Xva, selected, ncol, nall, fallback = selection_fit(
                        features.loc[itrain], trr['Actual'].to_numpy(dtype=int),
                        trr['population_weight'].to_numpy(dtype=float),
                        features.loc[ival], excluded
                    )
                    atomic_npz(inner_file,
                               Run_Hash=np.array(run_hash),
                               Train_Index=itrain, Val_Index=ival,
                               Xtrain=Xtr, Xval=Xva,
                               Selected_Groups=np.asarray(selected,dtype=str),
                               Fallback=np.array(int(fallback)))
                    print(f'    Inner {inner}: new Boruta, groups={selected}, '
                          f'encoded={ncol}, fallback={fallback}', flush=True)
                assert len(selected) > 0 and not (set(selected) & excluded)
                trr = rows_by_id.loc[itrain]
                var = rows_by_id.loc[ival]
                ytr = trr['Actual'].to_numpy(dtype=int)
                wtr = trr['population_weight'].to_numpy(dtype=float)
                yva = var['Actual'].to_numpy(dtype=int)
                wva = var['population_weight'].to_numpy(dtype=float)
                assert np.isfinite(Xtr).all() and np.isfinite(Xva).all()
                for c_id, params in enumerate(GRID, 1):
                    m = RandomForestClassifier(n_estimators=RF_TREES,
                                               random_state=SEED, n_jobs=-1,
                                               **params)
                    m.fit(Xtr, ytr, sample_weight=wtr/wtr.mean())
                    p = m.predict_proba(Xva)[:, 1]
                    fold_records.append({
                        'Outer_Fold': outer, 'Inner_Fold': inner,
                        'Candidate_ID': c_id,
                        'Parameters_JSON': json.dumps(params, sort_keys=True),
                        'Weighted_AP': float(average_precision_score(
                            yva, p, sample_weight=wva)),
                        'Weighted_ROC_AUC': float(roc_auc_score(
                            yva, p, sample_weight=wva)),
                        'Selected_Groups': '; '.join(selected),
                        'Fallback_Used': int(fallback)
                    })
                del Xtr, Xva
                gc.collect()
            scores = pd.DataFrame(fold_records)
            assert len(scores) == 36
            ranking = scores.groupby(['Candidate_ID','Parameters_JSON'],
                                     as_index=False).agg(
                Inner_Folds=('Inner_Fold', 'nunique'),
                Mean_Weighted_AP=('Weighted_AP', 'mean'),
                Mean_Weighted_ROC_AUC=('Weighted_ROC_AUC', 'mean'))
            assert len(ranking) == 12 and (ranking['Inner_Folds'] == 3).all()
            ranking = ranking.sort_values(
                ['Mean_Weighted_AP','Mean_Weighted_ROC_AUC','Candidate_ID'],
                ascending=[False,False,True]).reset_index(drop=True)
            winning = json.loads(ranking.iloc[0]['Parameters_JSON'])
            atomic_csv(scores, fold_dir / 'Inner_CV_36_Fits.csv')
            atomic_csv(ranking, fold_dir / 'Inner_CV_Ranking.csv')

            outer_matrix = fold_dir / 'Outer_Train_Boruta_Matrices.npz'
            if outer_matrix.exists():
                with np.load(outer_matrix, allow_pickle=False) as z:
                    assert z['Run_Hash'].item() == run_hash
                    assert np.array_equal(z['Train_Index'], tridx)
                    assert np.array_equal(z['Val_Index'], validx)
                    Ztr = z['Xtrain'].copy()
                    Zva = z['Xval'].copy()
                    selected = list(z['Selected_Groups'].tolist())
                    fallback = bool(z['Fallback'].item())
            else:
                trr = rows_by_id.loc[tridx]
                Ztr, Zva, selected, ncol, nall, fallback = selection_fit(
                    features.loc[tridx], trr['Actual'].to_numpy(dtype=int),
                    trr['population_weight'].to_numpy(dtype=float),
                    features.loc[validx], excluded
                )
                atomic_npz(outer_matrix,
                           Run_Hash=np.array(run_hash),
                           Train_Index=tridx, Val_Index=validx,
                           Xtrain=Ztr, Xval=Zva,
                           Selected_Groups=np.asarray(selected, dtype=str),
                           Fallback=np.array(int(fallback)))
            trr = rows_by_id.loc[tridx]
            vv = rows_by_id.loc[validx]
            final = RandomForestClassifier(n_estimators=RF_TREES,
                                           random_state=SEED,n_jobs=-1,
                                           **winning)
            final.fit(Ztr, trr['Actual'].to_numpy(dtype=int),
                      sample_weight=trr['population_weight'].to_numpy(dtype=float)
                      /trr['population_weight'].mean())
            p = final.predict_proba(Zva)[:, 1]
            assert np.isfinite(p).all() and ((0<=p)&(p<=1)).all()
            out = pd.DataFrame({
                'Run_Hash': run_hash,
                'Version': VERSION,
                'Scenario': scenario,
                'Outer_Fold': outer,
                'original_row_index': validx,
                'Actual': vv['Actual'].to_numpy(dtype=int),
                'population_weight': vv['population_weight'].to_numpy(dtype=float),
                'stratum16': vv['stratum16'].to_numpy(),
                'psu': vv['psu'].to_numpy(),
                'Predicted_Probability': p,
                'Best_Params_JSON': json.dumps(winning, sort_keys=True),
                'Selected_Original_Groups': '; '.join(selected),
                'Retained_Encoded_Columns': Ztr.shape[1],
                'Zero_Confirmed_Fallback': int(fallback)
            })
            atomic_csv(out, outer_pred_file)
            prediction_parts.append(out)
            print(f'  Outer {outer}: {len(out)} held-out predictions saved; '
                  f'groups={selected}; best={winning}', flush=True)
            del Ztr, Zva
            gc.collect()

    all_oof = pd.concat(prediction_parts, ignore_index=True)
    assert len(all_oof) == 3 * 2110
    assert not all_oof.duplicated(['Scenario','original_row_index']).any()
    assert all_oof.groupby('Scenario')['Outer_Fold'].nunique().eq(5).all()
    assert all_oof.groupby('Scenario').size().eq(2110).all()
    summary = pd.DataFrame([
        {'Scenario': s, **model_metrics(all_oof.loc[all_oof['Scenario']==s])}
        for s, _ in SCENARIOS
    ])
    base = summary.loc[summary['Scenario']=='Full_35'].iloc[0]
    for metric in ('Weighted_AP', 'Weighted_ROC_AUC', 'Weighted_Brier'):
        summary[f'Difference_vs_New_Full_{metric}'] = summary[metric] - base[metric]
    atomic_csv(all_oof, work / 'Step1_46_All_Scenario_OOF_Predictions.csv')
    atomic_csv(summary, work / 'Step1_46_Nested_Sensitivity_Summary.csv')
    print('\n========== STEP 1.46 FULLY NESTED SENSITIVITY RESULTS ==========', flush=True)
    print(summary.round(5).to_string(index=False), flush=True)
    print('\nORIGINAL PRIMARY RF (CONTEXT ONLY): Weighted AP 0.49894; '
          'AUC 0.77145; Brier 0.14907.', flush=True)
    print('New full baseline is the fair within-experiment comparator.', flush=True)
    print('\n========== STEP 1.46 FINAL VERIFICATION ==========', flush=True)
    print('Three scenarios x 5 outer folds; 15 full outer Boruta fits.', flush=True)
    print('45 inner Boruta fits; 540 inner RF candidate fits.', flush=True)
    print('15 final outer Random Forest fits; 6330 cross-fitted predictions.', flush=True)
    print('PSU leakage: 0 (original 5 outer / 15 inner partitions).', flush=True)
    print('Tuning and feature selection used outer validation: 0.', flush=True)
    print('Outputs:', work, flush=True)
    print('STEP 1.46 VERIFICATION SUCCESSFUL.', flush=True)


if __name__ == '__main__':
    main()
