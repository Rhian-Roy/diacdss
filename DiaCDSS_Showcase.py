"""
DiaCDSS — complete project in ONE app (CELBC710 Computational Intelligence Lab, Comp B Batch 4, Group 1)

    streamlit run DiaCDSS_Showcase.py

Runs fully offline. Page 1 = the project in one screen, page 2 = one patient through all ten tasks,
page 3 = the evidence (figures, tables, live mini-runs) for any single task.
"""
import os, sys, subprocess, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
from cdss_core import CDSS, ATTRIBUTES, BASE
import project_tasks as pt

st.set_page_config(page_title="DiaCDSS · CELBC710 Group 1", page_icon="🩺", layout="wide")

@st.cache_resource(show_spinner="Loading the trained models (DNN, XGBoost), rules and explainers…")
def load_cdss():
    return CDSS()

def show_image(path, caption=None):
    if os.path.exists(path):
        st.image(path, caption=caption)
    else:
        st.info(f"Figure not found: {path}")

def show_csv(path, **kw):
    if os.path.exists(path):
        st.dataframe(pd.read_csv(path, **kw), hide_index=True)
    else:
        st.info(f"Report not found: {path}")

PAGES = ["1 · Project overview", "2 · Live demo: one patient, all 10 tasks", "3 · Evidence for each task"]
st.sidebar.title("🩺 DiaCDSS")
st.sidebar.caption("Intelligent Clinical Decision Support System for Early Diabetes Diagnosis & Risk Prediction")
page = st.sidebar.radio("Show", PAGES, key="page")
st.sidebar.markdown("---")
st.sidebar.caption("CELBC710 Computational Intelligence Laboratory · Comp B Batch 4 · Group 1 · runs offline")

# =====================================================================================================
if page == PAGES[0]:
    st.title("DiaCDSS: Intelligent CDSS for Early Diabetes")
    st.markdown("**Computational Intelligence-Based Intelligent Clinical Decision Support System for Early Diabetes "
                "Diagnosis and Risk Prediction using Explainable AI and Causal Inference**")
    st.write("Type 2 diabetes is often detected late. DiaCDSS takes a patient's routine measurements and returns a risk "
             "decision (LOW / MODERATE / HIGH), a recommendation, and an explanation a clinician can check. It combines "
             "**AI reasoning** (knowledge model, IF-THEN rules, A* search), **machine learning** (XGBoost), "
             "**deep learning** (DNN) and **explainable AI** (SHAP, trust score, counterfactuals) in one pipeline.")
    KPIS = [("Patients (Pima dataset)", "768", "500 healthy · 268 diabetic"),
            ("Best ML test AUC", "0.819", "XGBoost · Task 6"),
            ("DNN 5-fold CV AUC", "0.842", "Task 7"),
            ("Screening sensitivity", "0.907", "49 of 54 diabetics caught"),
            ("Automated tests", "17 / 17", "+ 10/10 clinical cases"),
            ("Age fairness gap", "0.28", "DNN alone: 0.39")]
    for col, (label, value, note) in zip(st.columns(6), KPIS):
        col.metric(label, value)
        col.caption(note)
    if True:
        st.subheader("How the 10 tasks form one system")
        st.dataframe(pd.DataFrame([
            ("1", "Knowledge representation", "Ontology + constraints → validates every input", "11 entities, 6/6 checks pass"),
            ("2", "Rule-based expert system", "Forward-chaining IF-THEN rules make the decision", "8 rules → 16 in the CDSS"),
            ("3", "State-space search", "A* finds the diagnostic pathway S0 → S8/S9/S10", "BFS 11, DFS 7, A* 9 nodes"),
            ("4", "Data preparation & EDA", "Zeros → missing → median imputation", "374 insulin values imputed"),
            ("5", "Feature engineering", "Ratios + risk flag, causal DAG, 4 selection methods", "Glucose, BMI, Age top"),
            ("6", "Machine learning", "XGBoost vote (threshold 0.30)", "DT recall 0.741 · XGB AUC 0.819"),
            ("7", "Deep learning", "DNN vote (threshold 0.42) + MC-Dropout", "Sensitivity 0.796"),
            ("8", "Explainable AI", "SHAP drivers, trust score, what-if", "Glucose > BMI in every method"),
            ("9", "Integration & deployment", "One pipeline + this app + PDF report", "0.64 s per patient"),
            ("10", "Validation & responsible AI", "Tests, vignettes, robustness, fairness", "17/17 · 10/10 · gap 0.28"),
        ], columns=["Task", "Experiment", "Role in DiaCDSS", "Key result"]), hide_index=True)
    st.subheader("Architecture")
    _, mid, _ = st.columns([1, 3, 1])
    with mid:
        show_image("task9_outputs/task9_architecture.png")
    st.warning("Decision support only, not a diagnosis. Trained on 768 Pima women; needs external validation before clinical use.")
    st.info("Next: open **2 · Live demo** to run one patient through all ten tasks.")

# =====================================================================================================
elif page == PAGES[1]:
    cdss = load_cdss()
    st.title("One patient through the complete system")
    PRESETS = {
        "Patient 185: diabetic, high risk": dict(Pregnancies=7, Glucose=194, BloodPressure=68, SkinThickness=28, Insulin=125, BMI=35.9, DiabetesPedigreeFunction=0.745, Age=41),
        "Patient 700: borderline, models disagree": dict(Pregnancies=2, Glucose=122, BloodPressure=76, SkinThickness=27, Insulin=200, BMI=35.9, DiabetesPedigreeFunction=0.483, Age=26),
        "Patient 32: healthy, lowest risk": dict(Pregnancies=3, Glucose=88, BloodPressure=58, SkinThickness=11, Insulin=54, BMI=24.8, DiabetesPedigreeFunction=0.267, Age=22),
        "Patient 6: diabetic with normal glucose (missed case)": dict(Pregnancies=3, Glucose=78, BloodPressure=50, SkinThickness=32, Insulin=88, BMI=31.0, DiabetesPedigreeFunction=0.248, Age=26),
        "Invalid input (glucose 900, age 15)": dict(Pregnancies=2, Glucose=900, BloodPressure=70, SkinThickness=20, Insulin=80, BMI=25.0, DiabetesPedigreeFunction=0.3, Age=15),
    }
    preset = st.selectbox("Choose a patient (you can edit any value)", list(PRESETS), key="preset")
    vals = PRESETS[preset]
    cols = st.columns(8)
    raw = {}
    for c, f in zip(cols, BASE):
        raw[f] = c.number_input(f"{f}", value=float(vals[f]), key=f"{preset}|{f}",
                                step=0.001 if f == "DiabetesPedigreeFunction" else (0.1 if f == "BMI" else 1.0),
                                format="%.3f" if f == "DiabetesPedigreeFunction" else ("%.1f" if f == "BMI" else "%.0f"),
                                help=ATTRIBUTES[f]["unit"] + (" · 0 = not measured" if ATTRIBUTES[f]["zero_missing"] else ""))
    with st.expander("Optional clinical information (used by rules R1–R5)"):
        o = st.columns(5)
        hb = o[0].number_input("HbA1c % (0 = not available)", 0.0, 20.0, 0.0, 0.1, key="hb")
        fpg = o[1].number_input("Fasting glucose (0 = n/a)", 0.0, 600.0, 0.0, 1.0, key="fpg")
        raw["HbA1c"] = hb or None
        raw["FastingGlucose"] = fpg or None
        raw["FamilyHistory"] = o[2].selectbox("Family history", ["Unknown", "Yes", "No"], key="fh")
        raw["PhysicalActivity"] = o[3].selectbox("Physical activity", ["Unknown", "Low", "Moderate", "High"], key="pa")
        raw["Symptoms"] = o[4].selectbox("Classic symptoms", ["Unknown", "Yes", "No"], key="sym")
    for k in ("FamilyHistory", "PhysicalActivity", "Symptoms"):
        if raw[k] == "Unknown":
            raw[k] = None

    if st.button("▶ Run the complete system", type="primary", key="run"):
        with st.spinner("Validation → features → XGBoost + DNN → rules → A* → SHAP…"):
            st.session_state["res"] = cdss.assess(raw, explain=True)
            st.session_state["res_for"] = preset
    res = st.session_state.get("res")
    if res is None:
        st.info("Click **Run the complete system**. Each step below is labelled with the task that performs it.")
        st.stop()
    if res["status"] == "invalid":
        with st.container(border=True):
            st.markdown("#### Step 1 · Task 1 (knowledge model constraints): input rejected")
            for lvl, msg in res["issues"]:
                (st.error if lvl == "error" else st.warning)(msg)
            st.write("Impossible values violate the constraints defined in the Task 1 knowledge model, so the system refuses to "
                     "predict instead of producing a confident wrong answer.")
        st.stop()

    d = res["decision"]
    {"HIGH": st.error, "MODERATE": st.warning, "LOW": st.success}[d](
        f"### Final decision: {d} RISK → {res['facts'].get('Recommendation', 'Clinical judgement')}")
    st.caption(f"Whole pipeline ran in {res['latency_s']:.2f} s. Patient: {st.session_state.get('res_for')}.")

    with st.container(border=True):
        st.markdown("#### Step 1 · Task 1: knowledge model → input validation")
        st.write("Every value is checked against the units and physiological limits defined in the Task 1 ontology "
                 "(zero means 'not measured' for glucose, BP, skin thickness, insulin and BMI).")
        if res["issues"]:
            for lvl, msg in res["issues"]:
                {"warning": st.warning, "info": st.info}.get(lvl, st.write)(msg)
        else:
            st.success("All 8 inputs are within physiological limits and the training range.")

    with st.container(border=True):
        st.markdown("#### Step 2 · Tasks 4 & 5: data preparation and feature engineering")
        fe = {k: round(float(v), 3) for k, v in res["features"].items()}
        st.dataframe(pd.DataFrame([fe]), hide_index=True)
        st.caption("Missing values use the training median (Task 4). Insulin_BMI_Ratio, Glucose_Insulin_Ratio and "
                   "High_Risk_Flag are the Task 5 engineered features; 11 features go to the models.")

    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Step 3 · Task 6: machine learning (XGBoost)")
            st.metric("XGBoost probability", f"{res['p_xgb']:.1%}")
            st.write(f"Threshold {res['thresholds']['XGBoost']:.2f} → vote **{res['facts']['ML_Risk']}**")
            st.caption("Best ROC-AUC of the five Task 6 models (0.819); threshold chosen by Youden's J on training data.")
        with c2:
            st.markdown("#### Step 4 · Task 7: deep learning (DNN)")
            st.metric("DNN probability", f"{res['p_dnn']:.1%}")
            st.write(f"Threshold {res['thresholds']['DNN']:.2f} → vote **{res['facts']['DL_Risk']}**")
            st.caption(f"Dense 64-32-16 network, best by 5-fold CV. MC-Dropout (50 passes): {res['mc_mean']:.0%} ± {2 * res['mc_sd']:.0%}.")

    with st.container(border=True):
        st.markdown("#### Step 5 · Task 2: expert system (forward chaining over patient facts + model votes)")
        st.dataframe(pd.DataFrame([{"Rule fired": r, "Rule": txt, "Conclusion": c} for r, txt, c in res["trace"]]), hide_index=True)
        st.caption("R1–R8 are the Task 2 rules; R9–R16 were added in Task 9 (2-h OGTT criteria, combining the two model votes, "
                   "safety flags). When rules conflict, the more severe conclusion wins.")

    with st.container(border=True):
        st.markdown("#### Step 6 · Task 3: A* diagnostic pathway")
        st.code(res["pathway"]["text"], language=None)
        st.write(f"Path cost **{res['pathway']['cost']}**, nodes expanded **{res['pathway']['expanded']}**. "
                 "Steps supported by the patient's evidence cost 1, unsupported steps cost 3; h = steps to goal (admissible).")

    with st.container(border=True):
        st.markdown("#### Step 7 · Task 8: explainable AI")
        a, b = st.columns([1.3, 1])
        with a:
            dr = pd.DataFrame(res["drivers"])
            fig, ax = plt.subplots(figsize=(6, 2.6))
            ax.barh([f"{r.feature} = {r.value:.3g}" for r in dr.itertuples()][::-1], dr["shap"][::-1],
                    color=["#C0392B" if v > 0 else "#2471A3" for v in dr["shap"][::-1]])
            ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("SHAP contribution to DNN risk (red raises, blue lowers)")
            st.pyplot(fig, clear_figure=True)
        with b:
            st.metric("Trust level", res["trust"])
            st.caption(f"Trust score {res['trust_score']:.2f} (distance to other class ÷ distance to predicted class)")
            cf = res.get("counterfactual")
            if cf:
                st.write(f"**What-if:** risk would drop below the threshold ({cf['new_risk']:.0%}) if glucose were "
                         f"{-cf['glucose_change']} mg/dL lower, BMI {-cf['bmi_change']:.1f} lower and BP {-cf['bp_change']} mm Hg lower.")
            else:
                st.write("**What-if:** not needed; the DNN risk is already below its threshold.")
            for flag in dict.fromkeys(res["facts"].get("Flags", [])):
                st.warning(flag)

    with st.container(border=True):
        st.markdown("#### Step 8 · Task 9: decision, recommendation and clinical report")
        st.write(f"Decision **{d}** · Recommendation **{res['facts'].get('Recommendation', '-')}**"
                 + (f" · Next test: **{res['facts']['Next_Test']}**" if res['facts'].get('Next_Test') else ""))
        st.download_button("⬇️ Download the PDF clinical report", data=cdss.report_pdf(res, patient_id=preset.split(":")[0]),
                           file_name="DiaCDSS_report.pdf", mime="application/pdf", key="pdf")

    with st.container(border=True):
        st.markdown("#### Step 9 · Task 10: validation")
        st.write("This same pipeline was tested with 17 automated tests, 10 clinical vignettes, robustness checks and a "
                 "fairness audit. Run the tests live:")
        if st.button("Run the 17 automated tests now", key="pytest"):
            with st.spinner("pytest running…"):
                out = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests", "-p", "no:cacheprovider"],
                                     capture_output=True, text=True, timeout=300, cwd=HERE)
            last = [l for l in (out.stdout + out.stderr).strip().splitlines() if l.strip()]
            summary = last[-1] if last else "no output"
            (st.success if out.returncode == 0 else st.error)(summary)

# =====================================================================================================
else:
    st.title("Evidence for each task")
    tabs = st.tabs([f"Task {i}" for i in range(1, 11)])
    with tabs[0]:
        st.subheader("Task 1 · Knowledge representation (ontology + knowledge graph)")
        st.write("11 entities (Patient, Medical Record, Symptom, Risk Factor, Laboratory Test, Lifestyle, Prediction Model, "
                 "Risk Assessment, Explainability Module, Causal Inference Module, Clinical Recommendation), attribute tables with "
                 "types, ranges and constraints, 11 relationships with cardinality, an is-a hierarchy and a frame view of each patient. "
                 "Validation: 6/6 checks passed; impossible inputs are rejected.")
        show_image("task1_knowledge_graph.png")
    with tabs[1]:
        st.subheader("Task 2 · Rule-based expert system (forward chaining)")
        a, b = st.columns([1, 1])
        with a:
            st.dataframe(pd.DataFrame([(r, t, f"{c[0]} = {c[1]}") for r, t, _, c in pt.TASK2_RULES],
                                      columns=["Rule", "IF", "THEN"]), hide_index=True)
            pid = st.selectbox("Run the rule engine on", list(pt.TASK2_PATIENTS), key="t2pid")
            facts, trace = pt.forward_chain(pt.TASK2_PATIENTS[pid])
            st.write(pt.TASK2_PATIENTS[pid])
            st.code("\n".join(trace), language=None)
            st.success(f"Diagnosis: {facts.get('Diagnosis')} · Risk: {facts['Risk_Level']} · {facts.get('Recommendation')}")
        with b:
            show_image("task2_rule_tree.png")
    with tabs[2]:
        st.subheader("Task 3 · State-space search (BFS, DFS, A*)")
        a, b = st.columns([1, 1])
        with a:
            goal = st.selectbox("Goal state", ["S10", "S9", "S8"], format_func=lambda s: f"{s} · {pt.STATES[s]}", key="t3goal")
            st.dataframe(pd.DataFrame(pt.compare_search("S0", goal)), hide_index=True)
            st.caption("All paths cost 4 because every step costs 1, so BFS is also optimal here; A* is optimal because "
                       "h = steps remaining never overestimates. Inside the CDSS (live demo, step 6) costs depend on the evidence.")
            show_image("task3_comparison.png")
        with b:
            show_image("task3_state_space.png")
    with tabs[3]:
        st.subheader("Task 4 · Data preparation & EDA")
        st.write("768 patients × 9 columns. Physically impossible zeros were treated as missing and replaced by the median: "
                 "Glucose 5 → 117, BloodPressure 35 → 72, SkinThickness 227 → 29, Insulin 374 → 125, BMI 11 → 32.3. "
                 "No duplicates. Class balance 65.1% / 34.9% (1.87 : 1).")
        a, b = st.columns(2)
        with a: show_image("missing_values_chart.png"); show_image("bivariate_bmi_glucose_vs_outcome.png")
        with b: show_image("correlation_heatmap.png"); show_image("target_distribution.png")
    with tabs[4]:
        st.subheader("Task 5 · Feature engineering, causal DAG & feature selection")
        st.write("Engineered BMI_Category, Age_Group, Glucose_Category, Insulin_BMI_Ratio, Glucose_Insulin_Ratio and High_Risk_Flag. "
                 "Four selection methods (correlation, mutual information, RFE, Random Forest importance) agree on "
                 "Glucose, BMI, Age and Pregnancies. 11 numerical features go to the models.")
        a, b = st.columns(2)
        with a: show_image("causal_dag.png")
        with b: show_image("rf_feature_importance.png"); show_image("mutual_information.png")
    with tabs[5]:
        st.subheader("Task 6 · Machine learning models")
        st.dataframe(pd.DataFrame([
            ("Decision Tree (chosen: best recall/F1)", 0.766, 0.741, 0.780, 0.690, 0.774),
            ("Random Forest", 0.721, 0.537, 0.820, 0.574, 0.816), ("SVM", 0.727, 0.519, 0.840, 0.571, 0.789),
            ("XGBoost", 0.727, 0.500, 0.850, 0.563, 0.819), ("Logistic Regression", 0.695, 0.444, 0.830, 0.505, 0.790)],
            columns=["Model", "Accuracy", "Recall", "Specificity", "F1", "ROC-AUC"]), hide_index=True)
        a, b = st.columns(2)
        with a: show_image("confusion_matrices.png"); show_image("roc_curves.png")
        with b: show_image("shap_summary.png")
    with tabs[6]:
        st.subheader("Task 7 · Deep learning (DNN, 1D-CNN, Attention-DNN)")
        st.write("Train 491 / validation 123 / test 154 (same test patients as Task 6). Dropout, L2, BatchNorm, early stopping, "
                 "balanced class weights (0.767 / 1.436). DNN best by 5-fold CV: AUC 0.842 ± 0.025; test sensitivity 0.796 at threshold 0.42.")
        show_csv("task7_outputs/evaluation_report.csv", usecols=["Model", "Family", "Threshold", "Accuracy", "Recall (Sens)", "Specificity", "F1-Score", "ROC-AUC", "AUC 95% CI"])
        a, b = st.columns(2)
        with a: show_image("task7_outputs/task7_architectures.png")
        with b: show_image("task7_outputs/task7_training_curves.png")
    with tabs[7]:
        st.subheader("Task 8 · Interpretability & explainable AI")
        st.write("Activations, CNN feature maps + Grad-CAM, attention weights, Integrated Gradients, SHAP, LIME, counterfactuals, "
                 "trust score, MC-Dropout, fairness by age. Glucose then BMI dominate every method; SHAP is more faithful than attention.")
        a, b = st.columns(2)
        with a: show_image("task8_outputs/task8_shap_beeswarm.png"); show_image("task8_outputs/task8_cnn_gradcam.png")
        with b: show_image("task8_outputs/task8_attention_weights.png"); show_image("task8_outputs/task8_fairness.png")
        show_csv("task8_outputs/faithfulness_report.csv")
    with tabs[8]:
        st.subheader("Task 9 · System integration & deployment")
        st.write("cdss_core.py integrates Tasks 1–8; app.py and this showcase are the Streamlit interfaces; a PDF report is "
                 "generated per patient. One patient with explanations: about 0.6 s. Run the live demo (page 2) to see it work.")
        a, b = st.columns(2)
        with a: show_image("task9_outputs/task9_architecture.png")
        with b: show_image("task9_outputs/task9_dashboard_P185.png")
    with tabs[9]:
        st.subheader("Task 10 · Validation, responsible AI & documentation")
        show_csv("task10_outputs/evaluation_report.csv")
        a, b = st.columns(2)
        with a:
            show_csv("task10_outputs/clinical_validation.csv")
            show_csv("task10_outputs/robustness.csv")
        with b:
            show_image("task10_outputs/task10_fairness.png")
            show_csv("task10_outputs/responsible_ai_checklist.csv")
        show_csv("task10_outputs/testing_report.csv")
