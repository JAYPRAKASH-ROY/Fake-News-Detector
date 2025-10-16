# train.py
import pandas as pd, numpy as np, json, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

DATA_DIR = Path(".")
true = pd.read_csv(DATA_DIR / "True.csv")
fake = pd.read_csv(DATA_DIR / "Fake.csv")

for df in (true, fake):
    if "title" not in df: df["title"] = ""
    if "text" not in df:  df["text"] = ""

true["label"] = 1   # TRUE
fake["label"] = 0   # FAKE

def fuse(df): 
    return (df["title"].astype(str).fillna("") + " " + df["text"].astype(str).fillna("")).str.strip()

true["content"] = fuse(true)
fake["content"] = fuse(fake)

df = (pd.concat([true[["content","label"]], fake[["content","label"]]], ignore_index=True)
        .dropna(subset=["content"])
        .drop_duplicates(subset=["content"])
        .reset_index(drop=True))

X_train, X_test, y_train, y_test = train_test_split(
    df["content"].values, df["label"].values, test_size=0.15, random_state=42, stratify=df["label"].values
)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=60000, ngram_range=(1,2), lowercase=True, stop_words="english")),
    ("clf", LogisticRegression(max_iter=300, solver="liblinear"))
])

pipe.fit(X_train, y_train)

# metrics
proba = pipe.predict_proba(X_test)[:,1]
pred  = (proba >= 0.5).astype(int)
acc   = accuracy_score(y_test, pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
try: roc_auc = roc_auc_score(y_test, proba)
except: roc_auc = float("nan")

metrics = dict(accuracy=float(acc), precision=float(prec), recall=float(rec), f1=float(f1),
               roc_auc=float(roc_auc), n_train=int(len(X_train)), n_test=int(len(X_test)))
Path("metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

# submission
pd.DataFrame({"id": np.arange(len(proba)), "label": np.where(pred==1,"TRUE","FAKE"), "proba_true": proba}) \
  .to_csv("submission.csv", index=False)

joblib.dump(pipe, "model.pkl")
print("Saved: model.pkl, submission.csv, metrics.json")
print("Metrics:", metrics)
