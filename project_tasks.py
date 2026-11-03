"""Small, importable versions of the Task 2 rule engine and Task 3 search (same logic as Task02/Task03 notebooks).
Used by the one-go showcase app (DiaCDSS_Showcase.py)."""
from collections import deque
import heapq, time

# ---------------- Task 2: original 8 IF-THEN rules + forward chaining ----------------
TASK2_RULES = [
    ("R1", "IF Glucose > 126 AND HbA1c ≥ 6.5", lambda f: f["Glucose"] > 126 and f["HbA1c"] >= 6.5, ("Diagnosis", "Diabetes Positive")),
    ("R2", "IF 100 ≤ Glucose ≤ 125 AND 5.7 ≤ HbA1c ≤ 6.4", lambda f: 100 <= f["Glucose"] <= 125 and 5.7 <= f["HbA1c"] <= 6.4, ("Diagnosis", "Pre-diabetes")),
    ("R3", "IF Glucose < 100 AND HbA1c < 5.7", lambda f: f["Glucose"] < 100 and f["HbA1c"] < 5.7, ("Diagnosis", "Normal")),
    ("R4", "IF BMI ≥ 30 AND Family_History = Yes", lambda f: f["BMI"] >= 30 and f["Family_History"] == "Yes", ("Risk_Level", "High")),
    ("R5", "IF Age > 45 AND Physical_Activity = Low", lambda f: f["Age"] > 45 and f["Physical_Activity"] == "Low", ("Risk_Level", "Moderate")),
    ("R6", "IF Diagnosis = Diabetes Positive OR Risk_Level = High", lambda f: f.get("Diagnosis") == "Diabetes Positive" or f.get("Risk_Level") == "High",
     ("Recommendation", "Consult Endocrinologist & Lifestyle Intervention")),
    ("R7", "IF Diagnosis = Pre-diabetes", lambda f: f.get("Diagnosis") == "Pre-diabetes", ("Recommendation", "Lifestyle Modification & Close Monitoring")),
    ("R8", "IF Diagnosis = Normal", lambda f: f.get("Diagnosis") == "Normal", ("Recommendation", "Routine Annual Checkup")),
]
SEVERITY = {"Risk_Level": ["Low", "Moderate", "High"],
            "Recommendation": ["Routine Annual Checkup", "Lifestyle Modification & Close Monitoring",
                               "Consult Endocrinologist & Lifestyle Intervention"]}
TASK2_PATIENTS = {
    "P001": {"Age": 55, "BMI": 36.0, "Glucose": 140, "HbA1c": 7.2, "Family_History": "Yes", "Physical_Activity": "Low"},
    "P002": {"Age": 30, "BMI": 22.5, "Glucose": 90, "HbA1c": 5.2, "Family_History": "No", "Physical_Activity": "High"},
    "P003": {"Age": 48, "BMI": 28.0, "Glucose": 110, "HbA1c": 6.0, "Family_History": "Yes", "Physical_Activity": "Low"},
    "P004": {"Age": 60, "BMI": 31.0, "Glucose": 190, "HbA1c": 8.5, "Family_History": "No", "Physical_Activity": "Low"},
}

def forward_chain(initial):
    facts, fired, trace, changed = dict(initial), set(), [], True
    while changed:
        changed = False
        for rid, text, cond, (attr, val) in TASK2_RULES:
            if rid in fired or not cond(facts):
                continue
            fired.add(rid)
            if attr not in facts:
                facts[attr] = val; changed = True; trace.append(f"{rid}: {attr} = {val}")
            else:
                order = SEVERITY.get(attr, [])
                if val in order and facts[attr] in order and order.index(val) > order.index(facts[attr]):
                    trace.append(f"{rid}: {attr} = {val} (overrides {facts[attr]})"); facts[attr] = val; changed = True
                else:
                    trace.append(f"{rid}: {attr} = {val} (overridden by {facts[attr]})")
    if "Risk_Level" not in facts:
        facts["Risk_Level"] = "Low"; trace.append("Default: Risk_Level = Low")
    return facts, trace

# ---------------- Task 3: state space + BFS / DFS / A* ----------------
GRAPH = {"S0": ["S1", "S2"], "S1": ["S3", "S4"], "S2": ["S5", "S6"], "S3": ["S7"], "S4": ["S7"], "S5": ["S7"],
         "S6": ["S7"], "S7": ["S8", "S9", "S10"], "S8": [], "S9": [], "S10": []}
STATES = {"S0": "Undiagnosed Patient", "S1": "Evaluate Risk Factors", "S2": "Evaluate Symptoms", "S3": "Low Risk",
          "S4": "High Risk", "S5": "Symptoms Absent", "S6": "Symptoms Present", "S7": "Preliminary Assessment",
          "S8": "Healthy", "S9": "Prediabetes", "S10": "Diabetes"}
H = {"S0": 4, "S1": 3, "S2": 3, "S3": 2, "S4": 2, "S5": 2, "S6": 2, "S7": 1, "S8": 1, "S9": 1, "S10": 0}
COST = {(s, n): 1 for s in GRAPH for n in GRAPH[s]}

def bfs(start, goal):
    q, seen, n = deque([(start, [start])]), set(), 0
    while q:
        s, p = q.popleft()
        if s in seen: continue
        seen.add(s); n += 1
        if s == goal: return p, n
        q.extend((x, p + [x]) for x in GRAPH[s] if x not in seen)
    return None, n

def dfs(start, goal):
    st, seen, n = [(start, [start])], set(), 0
    while st:
        s, p = st.pop()
        if s in seen: continue
        seen.add(s); n += 1
        if s == goal: return p, n
        st.extend((x, p + [x]) for x in reversed(GRAPH[s]) if x not in seen)
    return None, n

def a_star(start, goal, cost=COST):
    pq, best, n = [(H[start], 0, start, [start])], {start: 0}, 0
    while pq:
        f, g, s, p = heapq.heappop(pq)
        if g > best.get(s, float("inf")): continue
        n += 1
        if s == goal: return p, n
        for x in GRAPH[s]:
            g2 = g + cost[(s, x)]
            if g2 < best.get(x, float("inf")):
                best[x] = g2; heapq.heappush(pq, (g2 + H[x], g2, x, p + [x]))
    return None, n

def compare_search(start="S0", goal="S10"):
    rows = []
    for name, fn, optimal in [("BFS", bfs, "Yes (equal costs)"), ("DFS", dfs, "No guarantee"), ("A*", a_star, "Yes (admissible h)")]:
        t0 = time.perf_counter(); path, n = fn(start, goal); t = time.perf_counter() - t0
        rows.append({"Algorithm": name, "Path": " → ".join(path), "Path cost": len(path) - 1,
                     "Nodes explored": n, "Time (µs)": round(t * 1e6, 1), "Optimal?": optimal})
    return rows
