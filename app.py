import streamlit as st
import numpy as np
import pandas as pd
import torch
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from io import StringIO
import time
import os
import gdown

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='ICD-11 Chapter Classifier',
    page_icon='🏥',
    layout='wide'
)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_NAME     = 'dmis-lab/biobert-base-cased-v1.2'
CHECKPOINT     = 'biobert_icd11_best.pt'
ENCODER        = 'icd11_label_encoder.pickle'
MAX_LEN        = 256
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Paste your Google Drive File IDs here ─────────────────────────────────────
MODEL_FILE_ID   = '1nWNKgHzwlrOQqWz9GmCCNK6BUkw932jJ'
ENCODER_FILE_ID = '1VvnRyzxoniUzII0Vxxj4FKl1UVvoLM1W'

ICD11_CODES = {
    'Certain infectious or parasitic diseases'                          : '1',
    'Neoplasms'                                                         : '2',
    'Diseases of the blood or blood-forming organs'                     : '3',
    'Diseases of the immune system'                                     : '4',
    'Endocrine, nutritional or metabolic diseases'                      : '5',
    'Mental, behavioural or neurodevelopmental disorders'               : '6',
    'Sleep-wake disorders'                                              : '7',
    'Diseases of the nervous system'                                    : '8',
    'Diseases of the visual system'                                     : '9',
    'Diseases of the ear or mastoid process'                            : '10',
    'Diseases of the circulatory system'                                : '11',
    'Diseases of the respiratory system'                                : '12',
    'Diseases of the digestive system'                                  : '13',
    'Diseases of the skin'                                              : '14',
    'Diseases of the musculoskeletal system or connective tissue'       : '15',
    'Diseases of the genitourinary system'                              : '16',
    'Pregnancy, childbirth or the puerperium'                           : '17',
    'Certain conditions originating in the perinatal period'            : '18',
    'Developmental anomalies'                                           : '19',
    'Symptoms, signs or clinical findings, not elsewhere classified'    : '21',
    'Injury, poisoning or certain other consequences of external causes': '22',
    'Conditions related to sexual health'                               : '23',
}

# ── Download model files from Google Drive if not present ─────────────────────
def download_files():
    if not os.path.exists(CHECKPOINT):
        with st.spinner('Downloading model weights from Google Drive — first run only, please wait...'):
            gdown.download(
                f'https://drive.google.com/uc?id={MODEL_FILE_ID}',
                CHECKPOINT,
                quiet=False
            )
        st.success('✓ Model weights downloaded')

    if not os.path.exists(ENCODER):
        with st.spinner('Downloading label encoder from Google Drive...'):
            gdown.download(
                f'https://drive.google.com/uc?id={ENCODER_FILE_ID}',
                ENCODER,
                quiet=False
            )
        st.success('✓ Label encoder downloaded')

# Run download check on every startup
download_files()

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(ENCODER, 'rb') as f:
        label_encoder = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_encoder.classes_)
    )
    model.load_state_dict(
        torch.load(CHECKPOINT, map_location=DEVICE)
    )
    model = model.to(DEVICE)
    model.eval()

    return model, tokenizer, label_encoder


# ── Prediction function ────────────────────────────────────────────────────────
def predict_note(text, model, tokenizer, label_encoder):
    if not text.strip():
        return None

    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )

    input_ids      = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs   = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    top_idx        = int(np.argmax(probs))
    top_label      = label_encoder.classes_[top_idx]
    top_confidence = float(probs[top_idx]) * 100
    icd_code       = ICD11_CODES.get(top_label, 'N/A')

    return {
        'chapter'    : top_label,
        'icd_code'   : icd_code,
        'confidence' : top_confidence,
    }


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title('🏥 ICD-11 Chapter Classifier')
st.markdown('Powered by **BioBERT** — fine-tuned on all 22 ICD-11 chapters')
st.divider()

# Load model with spinner
with st.spinner('Loading BioBERT model — please wait...'):
    try:
        model, tokenizer, label_encoder = load_model()
        st.success(f'✓ Model ready — running on **{str(DEVICE).upper()}**')
    except Exception as e:
        st.error(f'Failed to load model: {e}')
        st.stop()

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(['📝  Single Note', '📋  Batch Processing'])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE NOTE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader('Type or Paste a Clinical Note')

    note_input = st.text_area(
        label='Clinical note',
        placeholder='e.g. 62yo male smoker, haemoptysis, 8kg weight loss, CXR right hilar mass...',
        height=200,
        label_visibility='collapsed'
    )

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        predict_btn = st.button('🔍 Classify', use_container_width=True, type='primary')
    with col2:
        clear_btn = st.button('🗑️ Clear', use_container_width=True)

    if clear_btn:
        st.rerun()

    if predict_btn:
        if not note_input.strip():
            st.warning('Please enter a clinical note before classifying.')
        else:
            with st.spinner('Analysing note...'):
                result = predict_note(note_input, model, tokenizer, label_encoder)
                time.sleep(0.3)

            st.divider()
            st.subheader('Result')

            conf = result['confidence']
            if conf >= 90:
                badge = '🟢 High confidence'
            elif conf >= 70:
                badge = '🟡 Moderate confidence'
            else:
                badge = '🔴 Low confidence — review recommended'

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric(
                    label='Predicted ICD-11 Chapter',
                    value=result['chapter'],
                )
            with col_b:
                st.metric(
                    label='ICD-11 Code',
                    value=f"Chapter {result['icd_code']}",
                )

            st.progress(conf / 100)
            st.markdown(
                f'**Confidence: {conf:.1f}%** &nbsp;&nbsp; {badge}',
                unsafe_allow_html=True
            )

            if conf < 70:
                st.warning(
                    'Confidence is below 70%. This note may contain ambiguous '
                    'clinical features. Manual review is recommended.'
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader('Batch Processing')
    st.markdown(
        'Upload a **CSV file** with a column named `text` containing one '
        'clinical note per row. The classifier will process all notes and '
        'return a downloadable results file.'
    )

    template_df = pd.DataFrame({'text': [
        '62yo male smoker, haemoptysis, 8kg weight loss, CXR right hilar mass',
        '28F, 32/40 gestation, BP 158/104, urine 3+ protein, headache, MgSO4 commenced',
        'Crushing chest pain, ST elevation V1-V4, troponin 14.2, cath lab activated',
    ]})
    st.download_button(
        label='⬇️ Download CSV Template',
        data=template_df.to_csv(index=False),
        file_name='icd11_template.csv',
        mime='text/csv'
    )

    st.divider()

    uploaded_file = st.file_uploader(
        'Upload your CSV file',
        type=['csv'],
        help='CSV must have a column named "text"'
    )

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            if 'text' not in df.columns:
                st.error(
                    'CSV must contain a column named "text". '
                    'Please check your file or download the template above.'
                )
            else:
                st.success(f'✓ File loaded — {len(df):,} notes detected')
                st.dataframe(df.head(3), use_container_width=True)

                if st.button('🚀 Run Batch Classification', type='primary'):
                    results  = []
                    progress = st.progress(0)
                    status   = st.empty()
                    total    = len(df)

                    for i, row in df.iterrows():
                        status.text(f'Processing note {i+1} of {total}...')
                        result = predict_note(
                            str(row['text']), model, tokenizer, label_encoder
                        )
                        if result:
                            results.append({
                                'note'         : str(row['text'])[:80] + '...',
                                'chapter'      : result['chapter'],
                                'icd_code'     : f"Chapter {result['icd_code']}",
                                'confidence_%' : round(result['confidence'], 1),
                                'flag'         : '⚠️ Review' if result['confidence'] < 70 else '✅ OK'
                            })
                        else:
                            results.append({
                                'note'         : str(row['text'])[:80] + '...',
                                'chapter'      : 'ERROR — empty note',
                                'icd_code'     : 'N/A',
                                'confidence_%' : 0.0,
                                'flag'         : '⚠️ Review'
                            })
                        progress.progress((i + 1) / total)

                    status.text('✓ Classification complete')
                    results_df = pd.DataFrame(results)

                    st.divider()
                    st.subheader('Results')

                    col1, col2, col3 = st.columns(3)
                    col1.metric('Total Notes',        total)
                    col2.metric('Average Confidence', f"{results_df['confidence_%'].mean():.1f}%")
                    col3.metric('Flagged for Review', int((results_df['flag'] == '⚠️ Review').sum()))

                    st.dataframe(results_df, use_container_width=True)

                    st.download_button(
                        label='⬇️ Download Results CSV',
                        data=results_df.to_csv(index=False),
                        file_name='icd11_results.csv',
                        mime='text/csv'
                    )

        except Exception as e:
            st.error(f'Error reading file: {e}')


# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<center><small>ICD-11 Chapter Classifier — BioBERT fine-tuned model — '
    'For research and educational use only</small></center>',
    unsafe_allow_html=True
)
