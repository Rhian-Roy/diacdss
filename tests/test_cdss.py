"""Functional tests for DiaCDSS (Task 10). Run:  pytest -v tests/"""
import os, sys, itertools
import numpy as np, pandas as pd, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cdss_core import CDSS, ExpertSystem, validate_input, engineer, pathway, GRAPH, BASE

@pytest.fixture(scope='session')
def cdss():
    return CDSS()

@pytest.fixture(scope='session')
def data():
    return pd.read_csv('diabetes_features_final.csv')

def patient(**kw):
    p = dict(Pregnancies=2, Glucose=110, BloodPressure=70, SkinThickness=25, Insulin=100, BMI=26, DiabetesPedigreeFunction=0.3, Age=30)
    p.update(kw); return p

# ---- Task 1 / Task 4: input validation ----
def test_rejects_impossible_values(cdss):
    _, issues = validate_input(patient(Glucose=900), cdss.medians, cdss.trained)
    assert any(lvl == 'error' and 'Glucose' in m for lvl, m in issues)

def test_imputes_zero_as_missing(cdss):
    clean, issues = validate_input(patient(Insulin=0), cdss.medians, cdss.trained)
    assert clean['Insulin'] == cdss.medians['Insulin'] and any('imputed' in m for _, m in issues)

def test_warns_outside_training_range(cdss):
    _, issues = validate_input(patient(Age=85), cdss.medians, cdss.trained)
    assert any(lvl == 'warning' and 'Age' in m for lvl, m in issues)

# ---- Task 5: feature engineering ----
def test_feature_engineering_reproduces_task5(data):
    eng = engineer(data[BASE])
    for col in ['Insulin_BMI_Ratio', 'Glucose_Insulin_Ratio', 'High_Risk_Flag']:
        assert np.allclose(eng[col].values, data[col].values)

# ---- Task 2: rule engine reproduces the Task 2 validation cases ----
@pytest.mark.parametrize('facts,diag,rec', [
    (dict(Age=55, BMI=36.0, FastingGlucose=140, HbA1c=7.2, FamilyHistory='Yes', PhysicalActivity='Low'), 'Diabetes Positive', 'Consult Endocrinologist & Lifestyle Intervention'),
    (dict(Age=30, BMI=22.5, FastingGlucose=90, HbA1c=5.2, FamilyHistory='No', PhysicalActivity='High'), 'Normal', 'Routine Annual Checkup'),
    (dict(Age=48, BMI=28.0, FastingGlucose=110, HbA1c=6.0, FamilyHistory='Yes', PhysicalActivity='Low'), 'Pre-diabetes', 'Lifestyle Modification & Close Monitoring'),
])
def test_task2_cases(facts, diag, rec):
    out, _ = ExpertSystem().run({**facts, 'Glucose': 120, 'DL_Risk': None, 'ML_Risk': None, 'Symptoms': None})
    assert out['Diagnosis'] == diag and out['Recommendation'] == rec

def test_ogtt_rule_overrides_models(cdss):
    r = cdss.assess(patient(Glucose=230), explain=False)
    assert r['decision'] == 'HIGH' and 'R9' in [t[0] for t in r['trace']]

# ---- Task 3: A* finds the optimal (minimum-cost) evidence path ----
@pytest.mark.parametrize('ai,goal', [('High', 'S10'), ('Low', 'S8'), ('Moderate', 'S9')])
def test_astar_is_optimal(ai, goal):
    facts = {'AI_Risk': ai, 'Symptoms': None}
    res = pathway(facts, goal)
    def paths(node, acc):  # brute force: every path S0 -> goal
        if node == goal: yield acc; return
        for n in GRAPH[node]: yield from paths(n, acc + [n])
    all_costs = [pathway(facts, goal)['cost']]  # A* result
    # recompute every path's cost with the same evidence-based costs
    risky = ai in ('High', 'Moderate')
    c = {('S0','S1'):1, ('S0','S2'):3, ('S1','S3'):3 if risky else 1, ('S1','S4'):1 if risky else 3,
         ('S2','S5'):3, ('S2','S6'):3}
    cost = lambda p: sum(c.get((a, b), 1) for a, b in zip(p, p[1:]))
    best = min(cost(p) for p in paths('S0', ['S0']))
    assert res['cost'] == best and res['path'][-1] == goal

# ---- Tasks 6-7: model outputs ----
def test_model_outputs_are_probabilities_and_deterministic(cdss, data):
    sub = data[BASE].iloc[:20]
    a, b = cdss.assess_batch(sub), cdss.assess_batch(sub)
    assert a['p_dnn'].between(0, 1).all() and a['p_xgb'].between(0, 1).all()
    assert np.allclose(a['p_dnn'].astype(float), b['p_dnn'].astype(float))

def test_batch_matches_single(cdss, data):
    sub = data[BASE].iloc[[5, 50, 185]]
    batch = cdss.assess_batch(sub)
    for i, (_, row) in enumerate(sub.iterrows()):
        assert cdss.assess(row.to_dict(), explain=False)['decision'] == batch['decision'].iloc[i]

# ---- Task 8 + Task 9: explanation, latency, report ----
def test_end_to_end_with_explanation(cdss, data):
    r = cdss.assess(data.loc[185, BASE].to_dict(), explain=True)
    assert r['status'] == 'ok' and len(r['drivers']) == 5 and r['pathway']['path'][-1] in ('S8', 'S9', 'S10')
    assert r['latency_s'] < 5

def test_invalid_input_is_not_assessed(cdss):
    assert cdss.assess(patient(Age=15), explain=False)['status'] == 'invalid'

def test_clinical_report_pdf(cdss, data):
    r = cdss.assess(data.loc[185, BASE].to_dict(), explain=True)
    pdf = cdss.report_pdf(r, patient_id='TEST')
    assert pdf[:4] == b'%PDF' and len(pdf) > 2000

def test_same_input_same_output(cdss, data):
    a = cdss.assess(data.loc[185, BASE].to_dict(), explain=True)
    b = cdss.assess(data.loc[185, BASE].to_dict(), explain=True)
    assert a['mc_sd'] == b['mc_sd'] and [d['shap'] for d in a['drivers']] == [d['shap'] for d in b['drivers']]