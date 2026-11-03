"""
DiaCDSS core engine - Task 9 (CELBC710, Group 1)
Integrates every earlier task into one decision pipeline:
  Task 1  knowledge representation  -> attribute constraints used for input validation
  Task 4  data preparation          -> zero-as-missing handling, training-median imputation
  Task 5  feature engineering       -> engineer()
  Task 2  rule-based reasoning      -> ExpertSystem (forward chaining, original rules R1-R8 + integration rules R9-R16)
  Task 6  machine learning          -> XGBoost probability
  Task 7  deep learning             -> DNN probability (+ MC-Dropout uncertainty)
  Task 8  explainable AI            -> SHAP drivers, trust score, counterfactual
  Task 3  state-space search        -> A* diagnostic pathway (S0 ... S8/S9/S10)
"""
import os, json, time, heapq, warnings
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import joblib

BASE = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# ---------------- Task 1: attribute constraints (unit, hard limits, zero means missing) ----------------
ATTRIBUTES = {
    'Pregnancies':              dict(unit='count',             hard=(0, 20),     zero_missing=False),
    'Glucose':                  dict(unit='mg/dL, 2-h OGTT',   hard=(30, 600),   zero_missing=True),
    'BloodPressure':            dict(unit='mm Hg, diastolic',  hard=(30, 150),   zero_missing=True),
    'SkinThickness':            dict(unit='mm, triceps',       hard=(3, 100),    zero_missing=True),
    'Insulin':                  dict(unit='uU/mL, 2-h',        hard=(5, 1000),   zero_missing=True),
    'BMI':                      dict(unit='kg/m2',             hard=(12, 80),    zero_missing=True),
    'DiabetesPedigreeFunction': dict(unit='score',             hard=(0.05, 3.0), zero_missing=False),
    'Age':                      dict(unit='years',             hard=(18, 110),   zero_missing=False),
}
OPTIONAL = {'HbA1c': (3.0, 20.0), 'FastingGlucose': (40, 600)}


# ---------------- Task 5: feature engineering (identical to the Task 5 notebook) ----------------
def engineer(base: pd.DataFrame) -> pd.DataFrame:
    d = base.copy()
    d['Insulin_BMI_Ratio'] = d['Insulin'] / d['BMI']
    d['Glucose_Insulin_Ratio'] = d['Glucose'] / (d['Insulin'] + 1)
    d['High_Risk_Flag'] = (((d['Age'] > 40) & (d['BMI'] >= 30) & (d['Glucose'] >= 126)) |
                           (d['DiabetesPedigreeFunction'] > 0.5)).astype(int)
    return d


# ---------------- Task 1 + Task 4: input validation ----------------
def validate_input(raw: dict, medians: dict, trained: dict):
    """Returns (clean values, issues). Issue levels: error (cannot assess), warning, info."""
    clean, issues = {}, []
    for f in BASE:
        v = raw.get(f, None)
        try:
            v = None if v is None or (isinstance(v, str) and v.strip() == '') else float(v)
        except (TypeError, ValueError):
            issues.append(('error', f'{f}: "{raw.get(f)}" is not a number')); continue
        if v is not None and np.isnan(v):
            v = None
        if v is None or (v == 0 and ATTRIBUTES[f]['zero_missing']):
            clean[f] = float(medians[f])
            issues.append(('warning', f'{f} missing - imputed with training median {medians[f]:g}'))
            continue
        lo, hi = ATTRIBUTES[f]['hard']
        if not lo <= v <= hi:
            issues.append(('error', f'{f} = {v:g} is outside the physiological limits {lo}-{hi} ({ATTRIBUTES[f]["unit"]})'))
            continue
        tlo, thi = trained[f]
        if not tlo <= v <= thi:
            issues.append(('warning', f'{f} = {v:g} is outside the training range {tlo:g}-{thi:g}; prediction less reliable'))
        clean[f] = v
    for f, (lo, hi) in OPTIONAL.items():
        v = raw.get(f)
        if v in (None, '', 0, 0.0):
            clean[f] = None
        else:
            v = float(v)
            if not lo <= v <= hi:
                issues.append(('error', f'{f} = {v:g} is outside {lo}-{hi}'))
            clean[f] = v
    fh = raw.get('FamilyHistory')
    if fh in ('Yes', 'No'):
        clean['FamilyHistory'] = fh
    else:
        clean['FamilyHistory'] = 'Yes' if clean.get('DiabetesPedigreeFunction', 0) > 0.5 else 'No'
        issues.append(('info', f"Family history not given - derived from pedigree score ({clean['FamilyHistory']})"))
    clean['PhysicalActivity'] = raw.get('PhysicalActivity') if raw.get('PhysicalActivity') in ('Low', 'Moderate', 'High') else None
    clean['Symptoms'] = raw.get('Symptoms') if raw.get('Symptoms') in ('Yes', 'No') else None
    return clean, issues


# ---------------- Task 2: forward-chaining expert system ----------------
DIAG_RANK = {None: 0, 'Normal': 1, 'Impaired glucose tolerance': 2, 'Pre-diabetes': 2,
             'Diabetes range (OGTT >= 200)': 3, 'Diabetes Positive': 3}
RISK_RANK = {None: 0, 'Low': 1, 'Moderate': 2, 'High': 3}

def _between(x, lo, hi):
    return x is not None and lo <= x <= hi

RULES = [  # (id, description, condition, conclusions)
    ('R1', 'IF FPG >= 126 AND HbA1c >= 6.5 THEN Diagnosis = Diabetes Positive',
     lambda f: (f['FastingGlucose'] or 0) >= 126 and (f['HbA1c'] or 0) >= 6.5, {'Diagnosis': 'Diabetes Positive'}),
    ('R2', 'IF FPG 100-125 AND HbA1c 5.7-6.4 THEN Diagnosis = Pre-diabetes',
     lambda f: _between(f['FastingGlucose'], 100, 125) and _between(f['HbA1c'], 5.7, 6.4), {'Diagnosis': 'Pre-diabetes'}),
    ('R3', 'IF FPG < 100 AND HbA1c < 5.7 THEN Diagnosis = Normal',
     lambda f: f['FastingGlucose'] is not None and f['FastingGlucose'] < 100 and f['HbA1c'] is not None and f['HbA1c'] < 5.7,
     {'Diagnosis': 'Normal'}),
    ('R4', 'IF BMI >= 30 AND Family_History = Yes THEN Risk_Level = High',
     lambda f: f['BMI'] >= 30 and f['FamilyHistory'] == 'Yes', {'Risk_Level': 'High'}),
    ('R5', 'IF Age > 45 AND Physical_Activity = Low THEN Risk_Level = Moderate',
     lambda f: f['Age'] > 45 and f['PhysicalActivity'] == 'Low', {'Risk_Level': 'Moderate'}),
    ('R9', 'IF 2-h OGTT >= 200 THEN Diagnosis = Diabetes range (OGTT >= 200)',
     lambda f: f['Glucose'] >= 200, {'Diagnosis': 'Diabetes range (OGTT >= 200)'}),
    ('R10', 'IF 140 <= 2-h OGTT < 200 THEN Diagnosis = Impaired glucose tolerance',
     lambda f: 140 <= f['Glucose'] < 200, {'Diagnosis': 'Impaired glucose tolerance'}),
    ('R11', 'IF DNN = High AND XGBoost = High THEN AI_Risk = High',
     lambda f: f['DL_Risk'] == 'High' and f['ML_Risk'] == 'High', {'AI_Risk': 'High'}),
    ('R12', 'IF exactly one model = High THEN AI_Risk = Moderate AND Flag = models disagree',
     lambda f: (f['DL_Risk'] == 'High') != (f['ML_Risk'] == 'High'),
     {'AI_Risk': 'Moderate', 'Flag': 'Models disagree - clinician review'}),
    ('R13', 'IF DNN = Low AND XGBoost = Low THEN AI_Risk = Low',
     lambda f: f['DL_Risk'] == 'Low' and f['ML_Risk'] == 'Low', {'AI_Risk': 'Low'}),
    ('R14', 'IF Trust = Low THEN Flag = low-trust prediction',
     lambda f: f.get('Trust') == 'Low', {'Flag': 'Low-trust prediction - clinician review'}),
    ('R15', 'IF AI_Risk in (High, Moderate) AND HbA1c unknown THEN Next_Test = HbA1c',
     lambda f: f.get('AI_Risk') in ('High', 'Moderate') and f['HbA1c'] is None, {'Next_Test': 'HbA1c or fasting plasma glucose (confirmatory)'}),
    ('R16', 'IF input warnings exist THEN Flag = interpret with caution',
     lambda f: f.get('Input_Warnings', 0) > 0, {'Flag': 'Input imputed or outside training range - interpret with caution'}),
    ('R6', 'IF Diagnosis = Diabetes OR Risk_Level = High OR AI_Risk = High THEN Recommendation = Consult Endocrinologist',
     lambda f: DIAG_RANK.get(f.get('Diagnosis')) == 3 or f.get('Risk_Level') == 'High' or f.get('AI_Risk') == 'High',
     {'Recommendation': 'Consult Endocrinologist & Lifestyle Intervention'}),
    ('R7', 'IF Diagnosis = Pre-diabetes/IGT OR AI_Risk = Moderate THEN Recommendation = Lifestyle Modification',
     lambda f: DIAG_RANK.get(f.get('Diagnosis')) == 2 or f.get('AI_Risk') == 'Moderate',
     {'Recommendation': 'Lifestyle Modification & Close Monitoring'}),
    ('R8', 'IF Diagnosis = Normal (or AI_Risk = Low with no abnormal finding) THEN Recommendation = Routine Annual Checkup',
     lambda f: (f.get('Diagnosis') == 'Normal' or (f.get('Diagnosis') is None and f.get('AI_Risk') == 'Low'))
               and f.get('AI_Risk') in ('Low', None) and f.get('Risk_Level') != 'High',
     {'Recommendation': 'Routine Annual Checkup'}),
]
REC_RANK = {None: 0, 'Routine Annual Checkup': 1, 'Lifestyle Modification & Close Monitoring': 2,
            'Consult Endocrinologist & Lifestyle Intervention': 3}


class ExpertSystem:
    """Data-driven forward chaining with conflict resolution (the more severe conclusion wins)."""
    def run(self, facts: dict):
        facts = dict(facts); facts.setdefault('Flags', [])
        fired, trace = set(), []
        changed = True
        while changed:
            changed = False
            for rid, text, cond, concl in RULES:
                if rid in fired:
                    continue
                try:
                    ok = bool(cond(facts))
                except Exception:
                    ok = False
                if not ok:
                    continue
                fired.add(rid); changed = True
                for k, v in concl.items():
                    if k == 'Flag':
                        facts['Flags'].append(v); trace.append((rid, text, f'Flag added: {v}')); continue
                    rank = {'Diagnosis': DIAG_RANK, 'Risk_Level': RISK_RANK, 'AI_Risk': RISK_RANK, 'Recommendation': REC_RANK}.get(k)
                    old = facts.get(k)
                    if rank and old is not None and rank.get(v, 0) <= rank.get(old, 0):
                        trace.append((rid, text, f'{k} = {v} (kept {old}: higher priority)'))
                    else:
                        facts[k] = v; trace.append((rid, text, f'{k} = {v}'))
        return facts, trace


# ---------------- Task 3: A* diagnostic pathway ----------------
GRAPH = {'S0': ['S1', 'S2'], 'S1': ['S3', 'S4'], 'S2': ['S5', 'S6'], 'S3': ['S7'], 'S4': ['S7'], 'S5': ['S7'],
         'S6': ['S7'], 'S7': ['S8', 'S9', 'S10'], 'S8': [], 'S9': [], 'S10': []}
STATE_DESC = {'S0': 'Undiagnosed Patient', 'S1': 'Evaluate Risk Factors', 'S2': 'Evaluate Symptoms', 'S3': 'Low Risk',
              'S4': 'High Risk', 'S5': 'Symptoms Absent', 'S6': 'Symptoms Present', 'S7': 'Preliminary Assessment',
              'S8': 'Healthy / Low risk', 'S9': 'Prediabetes / At risk', 'S10': 'Diabetes / High risk'}
STEPS_TO_GOAL = {'S0': 4, 'S1': 3, 'S2': 3, 'S3': 2, 'S4': 2, 'S5': 2, 'S6': 2, 'S7': 1}

def pathway(facts: dict, goal: str):
    """A* over the Task 3 state space. Edge costs follow the patient's evidence:
    a transition that agrees with the evidence costs 1, one that contradicts or lacks evidence costs 3.
    Heuristic = steps remaining to the goal (Task 3 heuristic); admissible because every edge costs >= 1."""
    risky = facts.get('AI_Risk') in ('High', 'Moderate') or facts.get('Risk_Level') == 'High'
    sym = facts.get('Symptoms')
    cost = {('S0', 'S1'): 1, ('S0', 'S2'): 1 if sym in ('Yes', 'No') else 3,
            ('S1', 'S3'): 3 if risky else 1, ('S1', 'S4'): 1 if risky else 3,
            ('S2', 'S5'): 1 if sym == 'No' else 3, ('S2', 'S6'): 1 if sym == 'Yes' else 3}
    for s in ('S3', 'S4', 'S5', 'S6'):
        cost[(s, 'S7')] = 1
    for g in ('S8', 'S9', 'S10'):
        cost[('S7', g)] = 1
    h = lambda s: 0 if s == goal else STEPS_TO_GOAL.get(s, float('inf'))
    pq, best, expanded = [(h('S0'), 0, 'S0', ['S0'])], {'S0': 0}, 0
    while pq:
        f_, g_, s, path = heapq.heappop(pq)
        if g_ > best.get(s, float('inf')):
            continue
        expanded += 1
        if s == goal:
            return {'path': path, 'cost': g_, 'expanded': expanded,
                    'text': ' -> '.join(f'{p} ({STATE_DESC[p]})' for p in path)}
        for n in GRAPH[s]:
            ng = g_ + cost[(s, n)]
            if ng < best.get(n, float('inf')):
                best[n] = ng; heapq.heappush(pq, (ng + h(n), ng, n, path + [n]))
    return {'path': None, 'cost': None, 'expanded': expanded, 'text': 'No path'}


# ---------------- Robust artifact loading (library versions can differ between machines) ----------------
def load_dnn(keras, art, params):
    for fn in (f'{art}/DNN.keras', f'{art}/DNN.h5'):
        try:
            return keras.models.load_model(fn, compile=False)
        except Exception:
            pass
    # last resort: rebuild the Task 7 architecture and load only the weights
    units, dropout = params.get('units', [64, 32, 16]), params.get('dropout', 0.3)
    inp = keras.Input(shape=(11,), name='clinical_features'); x = inp
    for i, u in enumerate(units, 1):
        x = keras.layers.Dense(u, activation='relu', name=f'dense_{i}')(x)
        x = keras.layers.BatchNormalization(name=f'bn_{i}')(x)
        x = keras.layers.Dropout(dropout, name=f'dropout_{i}')(x)
    out = keras.layers.Dense(1, activation='sigmoid', name='diabetes_probability')(x)
    model = keras.Model(inp, out, name='DNN'); model.load_weights(f'{art}/DNN.h5')
    return model


class NumpyDNN:
    """The trained Task 7 DNN (Dense 64-32-16 + BatchNorm + Dropout 0.3, sigmoid) evaluated in NumPy.
    Same weights as artifacts/DNN.keras (exported to DNN_weights.npz), same outputs to 1e-6, but no
    TensorFlow at run time, so the demo cannot crash on TensorFlow/OpenMP problems. Set CDSS_USE_TF=1
    to use Keras instead."""
    def __init__(self, path, rate=0.3, eps=1e-3):
        w = np.load(path)
        self.layers = []
        for i in (1, 2, 3):
            scale = w[f'bn_{i}__gamma'] / np.sqrt(w[f'bn_{i}__var'] + eps)
            self.layers.append((w[f'dense_{i}__kernel'], w[f'dense_{i}__bias'], scale,
                                w[f'bn_{i}__beta'] - w[f'bn_{i}__mean'] * scale))
        self.out_w, self.out_b = w['diabetes_probability__kernel'], w['diabetes_probability__bias']
        self.rate = rate

    def __call__(self, X, rng=None):
        """rng=None: normal prediction. rng=np.random.Generator: MC-Dropout pass (dropout kept on)."""
        h = np.asarray(X, dtype='float32')
        for k, b, scale, shift in self.layers:
            h = np.maximum(h @ k + b, 0) * scale + shift          # Dense+ReLU, then BatchNorm (inference)
            if rng is not None:                                     # inverted dropout, as in Keras
                h = h * (rng.random(h.shape) >= self.rate) / (1 - self.rate)
        z = (h @ self.out_w + self.out_b).ravel()
        return 1 / (1 + np.exp(-z))


# ---------------- Integrated system ----------------
class CDSS:
    def __init__(self, artifacts=None, data_file=None, seed=42):
        art = artifacts or os.environ.get('CDSS_ARTIFACTS', 'artifacts')
        data_file = data_file or os.environ.get('CDSS_DATA', 'diabetes_features_final.csv')
        cfg = json.load(open(f'{art}/task7_config.json'))
        self.features, self.task6_order = cfg['features'], cfg['task6_order']
        self.thr = {'DNN': cfg['thresholds']['DNN'], 'XGBoost': cfg['thresholds']['XGBoost']}
        self.keras = None
        if os.environ.get('CDSS_USE_TF') == '1' or not os.path.exists(f'{art}/DNN_weights.npz'):
            from tensorflow import keras            # optional: the original Keras model
            self.keras = keras
            self.dnn = load_dnn(keras, art, cfg['best_params'].get('DNN', {}))
        else:
            self.dnn = NumpyDNN(f'{art}/DNN_weights.npz')
        splits = json.load(open(f'{art}/splits.json'))
        df = pd.read_csv(data_file)
        tr = df.loc[splits['train']]
        try:
            self.scaler = joblib.load(f'{art}/scaler.pkl')
        except Exception:   # refit on the same training rows -> identical scaler
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler().fit(tr[self.features].astype('float32'))
        try:
            self.xgb = joblib.load(f'{art}/ml_baselines.pkl')['XGBoost']
        except Exception:   # refit with the Task 6 tuned hyperparameters on the same rows
            from xgboost import XGBClassifier
            trval = df.loc[splits['train'] + splits['val']]
            self.xgb = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.01, eval_metric='logloss',
                                     random_state=42).fit(trval[self.task6_order], trval['Outcome'])
        self.medians = tr[BASE].median().to_dict()
        self.trained = {f: (float(tr[f].min()), float(tr[f].max())) for f in BASE}
        self.Xtr = self.scaler.transform(tr[self.features].astype('float32')).astype('float32')
        self.ytr = tr['Outcome'].values
        self.sd = tr[['Glucose', 'BMI', 'BloodPressure']].std().to_dict()
        # Task 8 trust score: density-filtered training points per class
        from sklearn.neighbors import NearestNeighbors
        self._nn = {}
        for c in (0, 1):
            Xc = self.Xtr[self.ytr == c]
            r = NearestNeighbors(n_neighbors=10).fit(Xc).kneighbors(Xc)[0][:, -1]
            self._nn[c] = NearestNeighbors(n_neighbors=1).fit(Xc[r <= np.quantile(r, 0.9)])
        va = df.loc[splits['val']]
        Xva = self.scaler.transform(va[self.features].astype('float32')).astype('float32')
        ts_val = self._trust(Xva, (self._dnn_prob(Xva) >= self.thr['DNN']).astype(int))
        self.trust_cut = np.quantile(ts_val, [1 / 3, 2 / 3])
        self._bg = None
        self.rules = ExpertSystem()
        self.seed = seed

    # ----- Task 8 MC-Dropout: a copy of the DNN with dropout kept on (BatchNorm stays in inference mode) -----
    def _mc_model(self):
        keras = self.keras
        class MCDropout(keras.layers.Dropout):
            def call(self, inputs, training=None):
                return super().call(inputs, training=True)
        swap = lambda l: MCDropout(rate=l.rate, name=l.name) if isinstance(l, keras.layers.Dropout) \
            else l.__class__.from_config(l.get_config())
        mc = keras.models.clone_model(self.dnn, clone_function=swap)
        mc.set_weights(self.dnn.get_weights())
        return mc

    # ----- models -----
    def _dnn_prob(self, Xs):
        if self.keras is None:
            return self.dnn(Xs)
        return self.dnn(np.asarray(Xs, dtype='float32'), training=False).numpy().ravel()

    def _mc_passes(self, Xs, n=50):
        """Task 8 MC-Dropout: n stochastic passes with dropout on; fixed seed -> same patient, same answer."""
        if self.keras is None:
            rng = np.random.default_rng(self.seed)
            return np.stack([self.dnn(Xs, rng=rng) for _ in range(n)])
        self.keras.utils.set_random_seed(self.seed)
        mc_model = self._mc_model()
        return np.stack([mc_model(Xs, training=False).numpy().ravel() for _ in range(n)])

    def _trust(self, Xs, yhat):
        d = np.column_stack([self._nn[c].kneighbors(Xs)[0][:, 0] for c in (0, 1)])
        r = np.arange(len(Xs))
        return d[r, 1 - yhat] / (d[r, yhat] + 1e-12)

    def _features(self, clean_rows):
        base = pd.DataFrame(clean_rows)[BASE].astype(float)
        feats = engineer(base)
        Xs = self.scaler.transform(feats[self.features].astype('float32')).astype('float32')
        return feats, Xs

    def predict(self, feats, Xs):
        p_dnn = self._dnn_prob(Xs)
        p_xgb = self.xgb.predict_proba(feats[self.task6_order])[:, 1]
        yhat = (p_dnn >= self.thr['DNN']).astype(int)
        ts = self._trust(Xs, yhat)
        level = np.where(ts < self.trust_cut[0], 'Low', np.where(ts < self.trust_cut[1], 'Medium', 'High'))
        return p_dnn, p_xgb, ts, level

    @staticmethod
    def decision_of(facts):
        if DIAG_RANK.get(facts.get('Diagnosis')) == 3 or facts.get('AI_Risk') == 'High':
            return 'HIGH'
        if facts.get('AI_Risk') == 'Moderate' or DIAG_RANK.get(facts.get('Diagnosis')) == 2 or facts.get('Risk_Level') == 'High':
            return 'MODERATE'
        return 'LOW'

    def _reason(self, clean, issues, p_dnn, p_xgb, trust_level):
        facts = {**clean, 'DL_Risk': 'High' if p_dnn >= self.thr['DNN'] else 'Low',
                 'ML_Risk': 'High' if p_xgb >= self.thr['XGBoost'] else 'Low', 'Trust': trust_level,
                 'Input_Warnings': sum(1 for lvl, _ in issues if lvl == 'warning')}
        facts, trace = self.rules.run(facts)
        decision = self.decision_of(facts)
        goal = {'LOW': 'S8', 'MODERATE': 'S9', 'HIGH': 'S10'}[decision]
        return facts, trace, decision, pathway(facts, goal)

    # ----- Task 8 explanations -----
    def explain(self, Xs_row, raw_row, top=5):
        import shap
        if self._bg is None:
            self._bg = shap.kmeans(self.Xtr, 25)
        ex = shap.KernelExplainer(lambda X: self._dnn_prob(X), self._bg)
        sv = np.asarray(ex.shap_values(Xs_row, nsamples=300, silent=True))[0]
        order = np.argsort(-np.abs(sv))[:top]
        return [{'feature': self.features[j], 'value': float(raw_row[self.features[j]]), 'shap': float(sv[j])} for j in order]

    def counterfactual(self, clean):
        row = pd.DataFrame([{f: clean[f] for f in BASE}])
        grid = [(g, b, bp) for g in range(0, 85, 5) for b in np.arange(0, 12.5, 0.5) for bp in range(0, 25, 5)]
        cand = pd.concat([row] * len(grid), ignore_index=True)
        G = np.array(grid)
        cand['Glucose'] -= G[:, 0]; cand['BMI'] -= G[:, 1]; cand['BloodPressure'] -= G[:, 2]
        ok = (cand['Glucose'] >= 80) & (cand['BMI'] >= 18.5) & (cand['BloodPressure'] >= 60)
        feats = engineer(cand)
        risk = self._dnn_prob(self.scaler.transform(feats[self.features].astype('float32')).astype('float32'))
        costv = G[:, 0] / self.sd['Glucose'] + G[:, 1] / self.sd['BMI'] + G[:, 2] / self.sd['BloodPressure']
        feas = np.where(ok.values & (risk < self.thr['DNN']))[0]
        if len(feas) == 0:
            return None
        j = feas[np.argmin(costv[feas])]
        return {'glucose_change': -int(G[j, 0]), 'bmi_change': -float(G[j, 1]), 'bp_change': -int(G[j, 2]), 'new_risk': float(risk[j])}

    # ----- public API -----
    def assess(self, raw: dict, explain=True):
        t0 = time.perf_counter()
        clean, issues = validate_input(raw, self.medians, self.trained)
        if any(lvl == 'error' for lvl, _ in issues):
            return {'status': 'invalid', 'issues': issues, 'input': raw}
        np.random.seed(self.seed)                      # same patient -> same SHAP values
        feats, Xs = self._features([clean])
        p_dnn, p_xgb, ts, level = self.predict(feats, Xs)
        mc = self._mc_passes(Xs)
        facts, trace, decision, path = self._reason(clean, issues, float(p_dnn[0]), float(p_xgb[0]), str(level[0]))
        out = {'status': 'ok', 'input': raw, 'clean': clean, 'issues': issues, 'features': feats.iloc[0].to_dict(),
               'p_dnn': float(p_dnn[0]), 'p_xgb': float(p_xgb[0]), 'thresholds': self.thr,
               'mc_mean': float(mc.mean()), 'mc_sd': float(mc.std()), 'trust_score': float(ts[0]), 'trust': str(level[0]),
               'facts': facts, 'trace': trace, 'decision': decision, 'pathway': path}
        if explain:
            out['drivers'] = self.explain(Xs, feats.iloc[0])
            out['counterfactual'] = self.counterfactual(clean) if p_dnn[0] >= self.thr['DNN'] else None
        out['latency_s'] = time.perf_counter() - t0
        return out

    def assess_batch(self, df: pd.DataFrame):
        """Vectorised model calls, then per-patient reasoning (used for evaluation and batch mode)."""
        rows, valid, issues_all = [], [], []
        for _, r in df.iterrows():
            clean, issues = validate_input(r.to_dict(), self.medians, self.trained)
            issues_all.append(issues); valid.append(not any(l == 'error' for l, _ in issues)); rows.append(clean)
        idx = [i for i, v in enumerate(valid) if v]
        out = pd.DataFrame(index=df.index, columns=['valid', 'p_dnn', 'p_xgb', 'trust', 'AI_Risk', 'Diagnosis', 'Risk_Level',
                                                    'decision', 'recommendation', 'rules_fired', 'pathway', 'flags'])
        out['valid'] = valid
        if idx:
            feats, Xs = self._features([rows[i] for i in idx])
            p_dnn, p_xgb, ts, level = self.predict(feats, Xs)
            for k, i in enumerate(idx):
                facts, trace, decision, path = self._reason(rows[i], issues_all[i], float(p_dnn[k]), float(p_xgb[k]), str(level[k]))
                out.iloc[i, 1:] = [p_dnn[k], p_xgb[k], level[k], facts.get('AI_Risk'), facts.get('Diagnosis'), facts.get('Risk_Level'),
                                   decision, facts.get('Recommendation'), ' '.join(t[0] for t in trace),
                                   ' > '.join(path['path']), '; '.join(facts.get('Flags', []))]
        return out

    # ----- clinical report generator -----
    def report_pdf(self, res, path=None, patient_id='Patient'):
        from io import BytesIO
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        buf = BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=40, rightMargin=40, topMargin=36, bottomMargin=36,
                                title='DiaCDSS clinical report')
        from reportlab.lib.styles import ParagraphStyle
        ss = getSampleStyleSheet(); body = ss['BodyText']; body.fontSize = 9.5; body.leading = 12
        cell = ParagraphStyle('cell', parent=body, fontSize=8.5, leading=10.5)
        def tbl(rows, widths, head=True):
            t = Table(rows, colWidths=widths)
            st = [('GRID', (0, 0), (-1, -1), 0.4, colors.grey), ('FONTSIZE', (0, 0), (-1, -1), 8.5), ('VALIGN', (0, 0), (-1, -1), 'TOP')]
            if head:
                st += [('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#DCE6F1')), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')]
            t.setStyle(TableStyle(st)); return t
        colour = {'HIGH': '#C0392B', 'MODERATE': '#D68910', 'LOW': '#1E8449'}[res['decision']]
        el = [Paragraph('DiaCDSS - Clinical Decision Support Report', ss['Title']),
              Paragraph(f'Patient: <b>{patient_id}</b> &nbsp;&nbsp; Generated: {time.strftime("%d %b %Y %H:%M")}', body), Spacer(1, 6),
              Paragraph(f'<font color="{colour}" size="13"><b>Decision: {res["decision"]} RISK</b></font> &nbsp; '
                        f'Recommendation: <b>{res["facts"].get("Recommendation", "-")}</b>', body), Spacer(1, 8),
              Paragraph('<b>1. Patient measurements</b>', body)]
        rows = [['Measurement', 'Value', 'Unit']] + [[f, f'{res["clean"][f]:g}', ATTRIBUTES[f]['unit']] for f in BASE]
        for k in ('HbA1c', 'FastingGlucose', 'FamilyHistory', 'PhysicalActivity', 'Symptoms'):
            rows.append([k, str(res['clean'].get(k) if res['clean'].get(k) is not None else 'not given'), ''])
        el += [tbl(rows, [170, 110, 150]), Spacer(1, 8), Paragraph('<b>2. AI predictions</b>', body),
               tbl([['Model', 'Risk', 'Threshold', 'Vote'],
                    ['DNN (Task 7)', f'{res["p_dnn"]:.1%}', f'{self.thr["DNN"]:.2f}', res['facts']['DL_Risk']],
                    ['XGBoost (Task 6)', f'{res["p_xgb"]:.1%}', f'{self.thr["XGBoost"]:.2f}', res['facts']['ML_Risk']],
                    ['MC-Dropout uncertainty', f'{res["mc_mean"]:.1%} +/- {2 * res["mc_sd"]:.1%}', '', ''],
                    ['Trust score (Task 8)', f'{res["trust_score"]:.2f}', '', res['trust']]], [170, 110, 80, 80]),
               Spacer(1, 8), Paragraph('<b>3. Expert-system inference trace (forward chaining)</b>', body),
               tbl([['Rule', 'Conclusion']] + [[r, Paragraph(c, cell)] for r, _, c in res['trace']], [50, 440]), Spacer(1, 8),
               Paragraph('<b>4. Diagnostic pathway (A* search)</b>', body), Paragraph(res['pathway']['text'], body), Spacer(1, 8)]
        if res.get('drivers'):
            el += [Paragraph('<b>5. Main drivers of the AI risk (SHAP)</b>', body),
                   tbl([['Feature', 'Value', 'Effect on risk']] +
                       [[d['feature'], f'{d["value"]:.3g}', f'{"raises" if d["shap"] > 0 else "lowers"} by {abs(d["shap"]):.2f}'] for d in res['drivers']],
                       [170, 110, 170]), Spacer(1, 6)]
        if res.get('counterfactual'):
            c = res['counterfactual']
            el.append(Paragraph(f'What-if (model explanation, not a treatment effect): risk falls below threshold if glucose were '
                                f'{-c["glucose_change"]} mg/dL lower, BMI {-c["bmi_change"]:.1f} lower and diastolic BP {-c["bp_change"]} mm Hg lower.', body))
        flags = res['facts'].get('Flags', []) + [m for l, m in res['issues'] if l == 'warning']
        if flags:
            el += [Spacer(1, 6), Paragraph('<b>Flags and data warnings:</b> ' + '; '.join(dict.fromkeys(flags)), body)]
        el += [Spacer(1, 10), Paragraph('<i>Decision support only - the final diagnosis rests with the clinician. Models were developed on the '
                                        'Pima Indians Diabetes dataset (adult women of Pima heritage) and need external validation.</i>', body)]
        doc.build(el)
        data = buf.getvalue()
        if path:
            open(path, 'wb').write(data)
        return data
