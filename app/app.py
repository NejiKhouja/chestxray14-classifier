import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
load_dotenv(os.path.join(_HERE, '..', '.env'))

from utils import load_model, preprocess, predict, get_gradcam, explain_with_groq, LABELS

CHECKPOINT = os.path.join(_HERE, '..', 'outputs', 'checkpoints', 'model_epoch5.pth')

# page config
st.set_page_config(page_title='ChestX-ray14 Classifier', layout='wide')
st.title('ChestX-ray14 Multi-label Classifier')
st.caption('Educational/research demo — not for clinical use.')

#sidebar 
with st.sidebar:
    st.header('Settings')
    groq_key  = st.text_input('Groq API Key', value=os.getenv('GROQ_API_KEY', ''), type='password')
    threshold = st.slider('Detection threshold', 0.1, 0.9, 0.5, 0.05)
    st.markdown('---')
    st.markdown('**Model:** DenseNet-121 + clinical features')
    st.markdown('**Classes:** 15 chest pathologies')
    st.markdown('**Checkpoint:** epoch 5 (partial training)')

# load model once 
@st.cache_resource
def get_model():
    if not os.path.exists(CHECKPOINT):
        return None, None, None
    return load_model(CHECKPOINT)

model, epoch, loss = get_model()

if model is None:
    st.error(f'Checkpoint not found at: {CHECKPOINT}')
    st.stop()

st.sidebar.success(f'Loaded: epoch {epoch}, loss {loss:.4f}')

#inputs 
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader('Input')
    uploaded = st.file_uploader('Upload a chest X-ray', type=['png', 'jpg', 'jpeg'])
    age      = st.number_input('Patient Age', min_value=0, max_value=120, value=50)
    gender   = st.selectbox('Patient Gender', ['M', 'F'])
    run_btn  = st.button('Analyze', type='primary', disabled=(uploaded is None))

# inference + display
if run_btn and uploaded is not None:
    image_pil = Image.open(uploaded)
    image_t, age_t, gender_t = preprocess(image_pil, age, gender)

    with st.spinner('Running inference...'):
        probs = predict(model, image_t, age_t, gender_t)

    top_class = int(np.argmax(probs))

    with st.spinner('Computing GradCAM...'):
        cam = get_gradcam(model, image_t, age_t, gender_t, top_class)

    with col2:
        st.subheader('Results')
        tab_cam, tab_probs, tab_llm = st.tabs(['GradCAM', 'Probabilities', 'LLM Explanation'])

        # GradCAM tab
        with tab_cam:
            img_arr  = np.array(image_pil.convert('RGB').resize((224, 224))) / 255.0
            heatmap  = cm.jet(cam)[:, :, :3]
            overlay  = np.clip(0.6 * img_arr + 0.4 * heatmap, 0, 1)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_arr, cmap='gray');  axes[0].set_title('Original');              axes[0].axis('off')
            axes[1].imshow(cam, cmap='jet');        axes[1].set_title(f'GradCAM: {LABELS[top_class]}'); axes[1].axis('off')
            axes[2].imshow(overlay);                axes[2].set_title('Overlay');               axes[2].axis('off')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption(f'GradCAM highlights regions that most influenced the top prediction: **{LABELS[top_class]}**')

        # Probabilities tab
        with tab_probs:
            colors = ['#e74c3c' if p > threshold else '#3498db' for p in probs]
            fig2, ax = plt.subplots(figsize=(10, 5))
            ax.barh(LABELS, probs, color=colors)
            ax.axvline(threshold, color='black', linestyle='--', linewidth=1, label=f'Threshold ({threshold:.0%})')
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probability')
            ax.set_title('Disease Probabilities')
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

            flagged = [(LABELS[i], probs[i]) for i in range(len(LABELS)) if probs[i] > threshold]
            if flagged:
                st.warning('Flagged: ' + ', '.join(f'{l} ({p:.1%})' for l, p in flagged))
            else:
                st.success('No findings above threshold.')

        # LLM Explanation tab
        with tab_llm:
            if not groq_key:
                st.info('Enter a Groq API key in the sidebar to enable LLM explanation.')
            else:
                with st.spinner('Asking Groq...'):
                    try:
                        explanation = explain_with_groq(probs, groq_key)
                        st.markdown(explanation)
                        st.caption('AI-generated explanation for educational purposes only.')
                    except Exception as e:
                        st.error(f'Groq API error: {e}')
