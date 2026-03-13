import streamlit as st
import numpy as np
import pandas as pd
import torch
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import time
import os
import gdown

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='ICD-11 Chapter Classifier',
    page_icon='🏥',
    layout='wide'
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(102,126,234,0.3);
    }
    .hero h1 {
        color: white;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero p {
        color: rgba(255,255,255,0.85);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 16px rgba(102,126,234,0.25);
    }
    .metric-card .label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.85;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 800;
        margin-top: 0.2rem;
    }

    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        box-shadow: 0 4px 16px rgba(17,153,142,0.3);
        margin-top: 1rem;
    }
    .result-card .chapter {
        font-size: 1.3rem;
        font-weight: 700;
    }
    .result-card .code {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.2rem;
    }
    .result-card .confidence {
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 0.8rem;
    }

    /* Warning result card */
    .result-card-warn {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        box-shadow: 0 4px 16px rgba(247,151,30,0.3);
        margin-top: 1rem;
    }
    .result-card-warn .chapter {
        font-size: 1.3rem;
        font-weight: 700;
    }
    .result-card-warn .code {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.2rem;
    }
    .result-card-warn .confidence {
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 0.8rem;
    }

    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* How to use steps */
    .step {
        background: #1e1e2e;
        border-left: 4px solid #667eea;
        border-radius: 0 8px 8px 0;
        padding: 0.7rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }

    /* Sidebar styles */
    .sidebar-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #2e2e4e;
    }

    /* Chapter list item */
    .chapter-item {
        background: #1e1e2e;
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        margin-bottom: 0.4rem;
        font-size: 0.82rem;
        border-left: 3px solid #667eea;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_NAME     = 'dmis-lab/biobert-base-cased-v1.2'
CHECKPOINT     = 'biobert_icd11_best.pt'
ENCODER        = 'icd11_label_encoder.pickle'
MAX_LEN        = 256
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Paste your Google Drive File IDs here ─────────────────────────────────────
MODEL_FILE_ID   = 'PASTE_YOUR_MODEL_FILE_ID_HERE'
ENCODER_FILE_ID = 'PASTE_YOUR_ENCODER_FILE_ID_HERE'

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

QUICK_EXAMPLES = {
    'STEMI (Circulatory)':
        '75yo male, sudden onset crushing chest pain radiating to left arm, diaphoretic, HR 110, BP 88/60, ECG ST elevation V2-V5, troponin 18.4, cath lab activated, LAD occlusion stented',
    'Pre-eclampsia (Pregnancy)':
        '28F, 36/40 weeks gestation, severe headache and visual disturbances, BP 170/110, urine 3+ protein, brisk reflexes with clonus, platelets 98, MgSO4 loading dose given, emergency LSCS prepared',
    'Depression (Mental health)':
        '34yo male, 6 week history of persistent low mood, unable to get out of bed, lost interest in all activities, early morning wakening, PHQ-9 score 24, passive suicidal ideation, sertraline commenced',
    'Pneumonia (Respiratory)':
        '67yo female, productive cough with green sputum, fever 38.9, right lower lobe consolidation on CXR, WBC 18.4, CRP 280, CURB-65 score 3, IV ceftriaxone commenced, O2 via venturi mask',
    'Neonatal jaundice (Perinatal)':
        'Neonate day 3, jaundice visible to abdomen, SBR 280 umol/L above treatment threshold, breastfeeding encouraged, phototherapy commenced, recheck SBR in 6 hours',
}

# ── Download model files ───────────────────────────────────────────────────────
def download_files():
    if not os.path.exists(CHECKPOINT):
        with st.spinner('Downloading model weights — first run only, please wait...'):
            gdown.download(
                f'https://drive.google.com/uc?id={MODEL_FILE_ID}',
                CHECKPOINT,
                quiet=False
            )

    if not os.path.exists(ENCODER):
        with st.spinner('Downloading label encoder...'):
            gdown.download(
                f'https://drive.google.com/uc?id={ENCODER_FILE_ID}',
                ENCODER,
                quiet=False
            )

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
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    return model, tokenizer, label_encoder

# ── Prediction ─────────────────────────────────────────────────────────────────
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

# ── Load model on startup ──────────────────────────────────────────────────────
with st.spinner('Loading BioBERT model...'):
    try:
        model, tokenizer, label_encoder = load_model()
        model_loaded = True
    except Exception as e:
        model_loaded = False
        model_error  = str(e)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # Model info
    st.markdown('### 📊 Model Information')
    st.markdown(f"""
    <div class="sidebar-card">
        <b>Model:</b> BioBERT<br>
        <b>ICD-11 Chapters:</b> 22<br>
        <b>Training samples:</b> 11,000<br>
        <b>Test accuracy:</b> 100%<br>
        <b>Device:</b> {str(DEVICE).upper()}
    </div>
    """, unsafe_allow_html=True)

    if model_loaded:
        st.success('✅ Model loaded successfully!')
    else:
        st.error('❌ Model failed to load')

    st.divider()

    # Quick examples
    st.markdown('### 🚀 Quick Examples')
    st.markdown('Try example notes:')
    selected_example = st.selectbox(
        'Select an example',
        ['Select an example...'] + list(QUICK_EXAMPLES.keys()),
        label_visibility='collapsed'
    )

    st.divider()

    # File upload
    st.markdown('### 📁 File Upload')
    st.markdown('Upload medical notes (TXT):')
    uploaded_txt = st.file_uploader(
        'Upload TXT',
        type=['txt'],
        label_visibility='collapsed',
        help='Upload a .txt file containing a clinical note'
    )

    st.divider()

    # ICD-11 chapters list
    st.markdown('### 📋 ICD-11 Chapters')
    with st.expander('View all ICD-11 chapters'):
        for chapter, code in ICD11_CODES.items():
            st.markdown(
                f'<div class="chapter-item">Ch.{code} — {chapter}</div>',
                unsafe_allow_html=True
            )

    st.divider()

    # About
    st.markdown('### 💡 About')
    st.markdown(
        'This AI classifier analyses medical notes and predicts '
        'the appropriate ICD-11 chapter using BioBERT, '
        'pretrained on 4.5 billion biomedical words.'
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════

# Hero banner
st.markdown("""
<div class="hero">
    <h1>🏥 Medical Notes ICD-11 Classifier</h1>
    <p>AI-Powered Classification into ICD-11 Chapters using BioBERT</p>
</div>
""", unsafe_allow_html=True)

# ── Two column layout ──────────────────────────────────────────────────────────
left_col, right_col = st.columns([3, 2])

with left_col:

    # Input section
    st.markdown('<div class="section-header">📄 Input Medical Notes</div>',
                unsafe_allow_html=True)
    st.markdown('Paste medical notes here:')

    # Determine text to show — from example, file upload or empty
    default_text = ''
    if selected_example != 'Select an example...':
        default_text = QUICK_EXAMPLES[selected_example]
    elif uploaded_txt is not None:
        default_text = uploaded_txt.read().decode('utf-8')

    note_input = st.text_area(
        label='note',
        value=default_text,
        placeholder='Enter medical notes describing patient symptoms, history, examination findings, and clinical impression...',
        height=220,
        label_visibility='collapsed'
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        classify_btn = st.button(
            '🔍 Classify Notes',
            use_container_width=True,
            type='primary'
        )
    with col2:
        clear_btn = st.button(
            '🗑️ Clear',
            use_container_width=True
        )

    if clear_btn:
        st.rerun()

    # How to use section
    st.divider()
    with st.expander('📖 How to Use This Classifier', expanded=True):
        steps = [
            ('1', 'Upload a text file with medical notes, OR'),
            ('2', 'Paste medical notes directly into the text area, OR'),
            ('3', 'Select an example from the sidebar'),
            ('4', 'Click "Classify Notes" to analyse with BioBERT'),
            ('5', 'Review the ICD-11 chapter classification with confidence score'),
        ]
        for num, text in steps:
            st.markdown(
                f'<div class="step"><b>{num}.</b> {text}</div>',
                unsafe_allow_html=True
            )


with right_col:

    # Model metrics
    st.markdown('<div class="section-header">📈 Model Metrics</div>',
                unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    with m1:
        st.markdown("""
        <div class="metric-card">
            <div class="label">ICD-11 Chapters</div>
            <div class="value">22</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="metric-card">
            <div class="label">Test Accuracy</div>
            <div class="value">100%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    m3, m4 = st.columns(2)
    with m3:
        st.markdown("""
        <div class="metric-card">
            <div class="label">Training Samples</div>
            <div class="value">11K</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown("""
        <div class="metric-card">
            <div class="label">Base Model</div>
            <div class="value" style="font-size:1rem; padding-top:0.4rem">BioBERT</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Result display
    st.markdown('<div class="section-header">🎯 Classification Result</div>',
                unsafe_allow_html=True)

    if classify_btn:
        if not model_loaded:
            st.error('Model not loaded. Please refresh the page.')
        elif not note_input.strip():
            st.warning('Please enter or upload a clinical note first.')
        else:
            with st.spinner('Analysing with BioBERT...'):
                result = predict_note(note_input, model, tokenizer, label_encoder)
                time.sleep(0.4)

            conf = result['confidence']

            if conf >= 70:
                st.markdown(f"""
                <div class="result-card">
                    <div class="chapter">📋 {result['chapter']}</div>
                    <div class="code">ICD-11 Chapter {result['icd_code']}</div>
                    <div class="confidence">Confidence: {conf:.1f}% 🟢</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-warn">
                    <div class="chapter">📋 {result['chapter']}</div>
                    <div class="code">ICD-11 Chapter {result['icd_code']}</div>
                    <div class="confidence">Confidence: {conf:.1f}% ⚠️</div>
                </div>
                """, unsafe_allow_html=True)
                st.warning(
                    'Confidence below 70%. Manual review recommended.'
                )

            # Confidence progress bar
            st.markdown('<br>', unsafe_allow_html=True)
            st.progress(conf / 100)
            st.caption(f'Confidence level: {conf:.1f}%')
    else:
        st.info('Enter a clinical note and click **Classify Notes** to see the result here.')


# ══════════════════════════════════════════════════════════════════════════════
# BATCH PROCESSING SECTION
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown('<div class="section-header">📋 Batch Processing</div>',
            unsafe_allow_html=True)

with st.expander('Process multiple notes at once via CSV upload'):
    st.markdown(
        'Upload a **CSV file** with a column named `text` — '
        'one clinical note per row.'
    )

    template_df = pd.DataFrame({'text': [
        '75yo male, crushing chest pain, ST elevation V2-V5, troponin 18.4, cath lab activated',
        '28F, 36/40 gestation, BP 170/110, urine 3+ protein, MgSO4 commenced',
        '34yo male, low mood 6 weeks, PHQ-9 score 24, sertraline commenced',
    ]})
    st.download_button(
        label='⬇️ Download CSV Template',
        data=template_df.to_csv(index=False),
        file_name='icd11_template.csv',
        mime='text/csv'
    )

    uploaded_csv = st.file_uploader(
        'Upload CSV',
        type=['csv'],
        key='batch_csv'
    )

    if uploaded_csv:
        try:
            df = pd.read_csv(uploaded_csv)
            if 'text' not in df.columns:
                st.error('CSV must have a column named "text".')
            else:
                st.success(f'✓ {len(df):,} notes loaded')
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

                    status.text('✓ Done')
                    results_df = pd.DataFrame(results)

                    c1, c2, c3 = st.columns(3)
                    c1.metric('Total Notes',        total)
                    c2.metric('Avg Confidence',     f"{results_df['confidence_%'].mean():.1f}%")
                    c3.metric('Flagged for Review', int((results_df['flag'] == '⚠️ Review').sum()))

                    st.dataframe(results_df, use_container_width=True)

                    st.download_button(
                        label='⬇️ Download Results CSV',
                        data=results_df.to_csv(index=False),
                        file_name='icd11_results.csv',
                        mime='text/csv'
                    )

        except Exception as e:
            st.error(f'Error: {e}')


# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<center><small>ICD-11 Chapter Classifier — BioBERT fine-tuned — '
    'For research and educational use only</small></center>',
    unsafe_allow_html=True
)
