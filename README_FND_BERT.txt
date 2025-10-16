
# FND with BERT — Quick Start

## 1) Install requirements
```bash
pip install -r requirements.txt
```

## 2) Run with Kaggle auto-download (Windows example)
```bash
python fnd_bert_pipeline.py --output_dir "C:\Users\JAYPRAKASH\Documents\1 Projects\FND with BERT" --download_kaggle --epochs 1 --max_samples 3000
```

> Tip: use a small `--max_samples` first to verify everything works, then remove it for a full run.

## 3) Or run with local CSVs
If you already have `True.csv` and `Fake.csv`:
```bash
python fnd_bert_pipeline.py --data_dir "C:\path\to\data" --output_dir "C:\Users\JAYPRAKASH\Documents\1 Projects\FND with BERT" --epochs 1
```

## 4) Outputs
- `model.pkl` — pickle with `model_name` + `state_dict` (CPU tensors)
- `submission.csv` — predictions for the test split (`id,label,proba_true`)
- `metrics.json` — test metrics (`accuracy, precision, recall, f1, roc_auc`)
- `model/` — Hugging Face `save_pretrained` folder
