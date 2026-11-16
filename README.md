# Explainable Intrusion Detection for IEC 104 (SCADA) with TabNet

Undergraduate thesis project (Information Security, Can Tho University, 2026): an 8-class network intrusion detection system for IEC 60870-5-104 substation traffic, built on **TabNet** and explained with **attention, SHAP and LIME**, plus an offline **Streamlit dashboard** for operators.

> Thesis title (VI): *Nghiên cứu và xây dựng hệ thống phát hiện xâm nhập mạng sử dụng mô hình TabNet kết hợp Explainable AI*

<!-- TODO: add a screenshot or GIF of the dashboard here -->
<!-- ![dashboard](docs/dashboard.png) -->

## Highlights

- 8-class classification (1 benign + 7 attack types) on ~320k network flows with extreme imbalance (~95% one class).
- Leak-free SMOTE: oversampling is applied to the training part of each fold only.
- Benchmark against 14 baseline models (same split, same 78 features).
- Three-level explainability: TabNet attention (intrinsic), SHAP (global / per-class / local), LIME (local).
- Error analysis: the largest confusion pair (portscan → dos) accounts for 88.6% of all misclassifications.

## Dataset

**SANDI-2024**, IEC 104 part: *Dataset to Train Intrusion Detection Systems based on Machine Learning Models for Electrical Substations* (Gutierrez Mlot et al., Zenodo, DOI [10.5281/zenodo.14096593](https://zenodo.org/records/14096593)). Paper: *Data in Brief*, vol. 57, 111153, 2024.

The data is licensed **CC BY-NC-SA 4.0** and is **not included in this repo**. Download `processed-iec104.tgz` from Zenodo and put the CSVs in `data/` (see [Usage](#usage)).

| Class | Flows |
|---|---|
| dosattack | 304,627 |
| portscanattack | 9,710 |
| ntpddosattack | 2,278 |
| iec104starvationattack | 2,028 |
| fuzzyattack | 939 |
| attackfree | 255 |
| floodattack | 108 |
| mitmattack | 26 |
| **Total** | **319,971** |

Preprocessing: drop `Flow ID`, IPs, `Protocol`, `Timestamp`; IQR clipping; z-score scaling; stratified 70/15/15 split (seed 42). 78 features remain.

## Results

Held-out test set (47,996 flows), single split:

| Model | Accuracy | Macro F1 | Macro AUC |
|---|---|---|---|
| TabNet | 0.9962 | 0.8908 | 0.9995 |
| TabNet + SMOTE | 0.9961 | 0.9019 | 0.9993 |
| Random Forest | 0.9983 | 0.9773 | 1.0000 |
| Decision Tree | 0.9999 | 0.9705 | 0.9998 |
| KNN | 0.9981 | 0.9520 | 0.9995 |
| XGBoost | 0.9992 | 0.9338 | 1.0000 |

5-fold stratified CV (Macro F1, mean ± std): TabNet **0.8749 ± 0.0372**, TabNet + SMOTE **0.8928 ± 0.0377**.

**Honest reading of the numbers**

- TabNet is competitive on accuracy and AUC but **does not beat tree ensembles on Macro F1**. The value of this project is the explainability and error analysis, not a leaderboard win.
- SMOTE raises Macro F1 in 5-fold CV (+0.018), mainly through `floodattack` (F1 0.767 → 0.917, better in all 5 folds). Results for `mitmattack` are inconsistent, and overall accuracy drops and becomes less stable (one fold where `portscanattack` degraded), so SMOTE is a conditional improvement.
- `mitmattack` (26 flows, 4 in the test set) and `floodattack` (108 flows) remain weak (F1 ≈ 0.44–0.50 and ≈ 0.77 in CV) because there is too little data.

## Explainability

| Level | Method | Where |
|---|---|---|
| 1 | TabNet attention masks (global, per step, per class) | `notebooks/06_xai.ipynb`, dashboard page 6 |
| 2 | SHAP: beeswarm / bar, per-class heatmap, waterfall | dashboard pages 2–3 |
| 2 | LIME for single predictions | dashboard page 4 |
| – | SHAP waterfall of the most frequent misclassification | dashboard page 5 |

## Repository layout

```
├── notebooks/                    # run in numeric order
│   ├── 01_merge_csv.ipynb            # merge the 8 IEC 104 CSV files
│   ├── 02_preprocess.ipynb           # clean, scale, split (70/15/15)
│   ├── 03_train_tabnet.ipynb         # baseline TabNet
│   ├── 04_smote_data.ipynb           # SMOTE on the training set only
│   ├── 05_train_tabnet_smote.ipynb   # TabNet + SMOTE
│   ├── 06_baselines.ipynb            # 14 baseline models
│   ├── 07_kfold_tabnet.ipynb         # 5-fold CV, TabNet
│   ├── 08_kfold_tabnet_smote.ipynb   # 5-fold CV, TabNet + SMOTE
│   └── 09_xai.ipynb                  # attention, SHAP, LIME
├── app/dashboard.py             # Streamlit dashboard (offline display of results)
├── results/                     # JSON/CSV metrics and figures
├── requirements.txt
└── README.md
```

## Usage

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. download processed-iec104.tgz from Zenodo and extract the 8 capture104-*.csv files into data/processed/
# 2. run the notebooks in order (01 -> 09) from inside notebooks/; outputs are written to data/processed/
# 3. dashboard (reads the files produced by the notebooks)
python -m streamlit run app/dashboard.py
```

All notebooks read and write `../data/processed/`. The dashboard defaults to `data/processed/`; change it in the sidebar if your files are elsewhere.

## Limitations

- Scaling and IQR clipping were fit on the whole dataset before splitting (mild leakage); SMOTE itself is leak-free.
- `Src Port` is kept as a feature and could act as a shortcut; flows from the same capture are split randomly, so near-duplicates may appear in train and test.
- Training settings differ slightly between some notebooks (`momentum`, `max_epochs` in CV) — see the notebooks before comparing runs directly.
- Very small minority classes make per-class metrics unstable.

## Citation and license

If you use the data, cite the SANDI-2024 dataset and paper above. Code in this repo: MIT (see `LICENSE`). The dataset keeps its own CC BY-NC-SA 4.0 license.

## Author

Lê Thành Duy — Information Security, Can Tho University (2022–2026). <!-- TODO: email / LinkedIn -->
