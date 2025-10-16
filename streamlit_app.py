# streamlit_app.py
import streamlit as st, joblib, numpy as np, pandas as pd, json

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

pipe = load_model()

def explain(text, top_k=10):
    vec = pipe.named_steps['tfidf']
    clf = pipe.named_steps['clf']
    X = vec.transform([text])
    coefs = clf.coef_[0]
    feat = vec.get_feature_names_out()
    # contribution = tfidf_value * coefficient
    contrib = X.toarray()[0] * coefs
    idx = np.argsort(contrib)
    neg = idx[:top_k]
    pos = idx[-top_k:][::-1]
    def rows(idxs):
        return [{"ngram": feat[i], "contribution": float(contrib[i])}
                for i in idxs if X.toarray()[0][i] != 0]
    return rows(pos), rows(neg)

st.title("📰 Fake News Detector")
st.write("Paste a headline or article; the model predicts whether it looks **FAKE** or **TRUE**.")

tab1, tab2 = st.tabs(["Single text", "Batch (CSV)"])

with tab1:
    text = st.text_area("Headline or Article", height=220, placeholder="Paste news text here...")
    threshold = st.slider("Decision threshold (prob TRUE ≥)", 0.0, 1.0, 0.5, 0.01)
    if st.button("Analyze", use_container_width=True):
        if not text.strip():
            st.warning("Please paste some text.")
        else:
            p_true = float(pipe.predict_proba([text])[0][1])
            label = "TRUE ✅" if p_true >= threshold else "FAKE ❌"
            st.metric("Prediction", label, delta=f"Confidence (prob TRUE): {p_true:.4f}")
            pos, neg = explain(text, top_k=10)
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Top supporting n-grams (→ TRUE)")
                st.dataframe(pd.DataFrame(pos))
            with c2:
                st.subheader("Top opposing n-grams (→ FAKE)")
                st.dataframe(pd.DataFrame(neg))

with tab2:
    st.write("Upload a CSV with a **text** or **content** column; download predictions.")
    file = st.file_uploader("CSV file", type=["csv"])
    if file:
        df = pd.read_csv(file)
        col = "text" if "text" in df.columns else ("content" if "content" in df.columns else None)
        if col is None:
            st.error("CSV must include a 'text' or 'content' column.")
        else:
            probs = pipe.predict_proba(df[col].astype(str).fillna(""))[:,1]
            labels = np.where(probs >= 0.5, "TRUE", "FAKE")
            out = pd.DataFrame({"text": df[col], "label": labels, "proba_true": probs})
            st.dataframe(out.head(30))
            st.download_button("Download predictions CSV",
                               out.to_csv(index=False).encode("utf-8"),
                               file_name="predictions.csv",
                               mime="text/csv")

# optional: display metrics if present
try:
    metrics = json.load(open("metrics.json", "r", encoding="utf-8"))
    st.caption(f"Model: TF-IDF + Logistic Regression | Metrics (held-out test): {metrics}")
except Exception:
    st.caption("Model: TF-IDF + Logistic Regression")
