# DiaCDSS · live web demo

Intelligent Clinical Decision Support System for Early Diabetes Diagnosis and Risk Prediction
(CELBC710 Computational Intelligence Laboratory · Comp B Batch 4 · Group 1).

Streamlit app with three pages: project overview, a live demo that runs one patient through all ten
tasks (validation → features → XGBoost + DNN → IF-THEN rules → A* pathway → SHAP → decision and PDF
report → automated tests), and the evidence for each task.

- Main file: `DiaCDSS_Showcase.py`
- Python 3.12, packages in `requirements.txt` (no TensorFlow needed: the trained DNN runs in NumPy
  from `artifacts/DNN_weights.npz`, identical outputs to the Keras model)
- Data: Pima Indians Diabetes dataset (public)

Run locally: `pip install -r requirements.txt` then `streamlit run DiaCDSS_Showcase.py`.
