# ICD-11 Chapter Classifier

A clinical note classifier that predicts which of the 22 ICD-11 
chapters a clinical note belongs to, powered by BioBERT.

## Features
- Single note classification with confidence score
- Batch processing via CSV upload
- Confidence threshold warnings
- Downloadable results

## Model
- Base model: BioBERT (dmis-lab/biobert-base-cased-v1.2)
- Fine-tuned on 11,000 synthetic clinical notes
- 22 ICD-11 chapters covered
- Test accuracy: 100%

## How to Run Locally
git clone https://github.com/yourusername/icd11-classifier.git
cd icd11-classifier
pip install -r requirements.txt
streamlit run app.py

## Disclaimer
For research and educational use only.
Not intended for clinical decision making.
```

---

## 2. .gitignore
Create a file named `.gitignore` and paste this:
```
__pycache__/
*.py[cod]
*.pyo
.env
.venv
env/
venv/
.ipynb_checkpoints/
*.ipynb
*.csv
*.log
*.pt
.DS_Store
Thumbs.db
```

---

## 3. icd11_label_encoder.pickle
This is already on your machine from training. Just copy it into your project folder.

---

## Final File List
```
icd11-classifier/
│
├── app.py                          ✅ done
├── requirements.txt                ✅ done
├── README.md                       ← create now
├── .gitignore                      ← create now
└── icd11_label_encoder.pickle      ← copy from training folder
