import streamlit as st
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.applications import MobileNetV2
from PIL import Image
import numpy as np
import base64
from utils.preprocess import preprocess_image


# =============================
# Function: Load Model
# =============================
@st.cache_resource
def load_model_with_weights():
    num_classes = 15
    mobilenet_base = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    mobilenet_base.trainable = False
    model = Sequential([
        mobilenet_base,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.load_weights('mobilenetv2_animals_final.keras')
    return model


model = load_model_with_weights()

class_names = [
    'Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin', 'Elephant',
    'Giraffe', 'Horse', 'Kangaroo', 'Lion', 'Panda', 'Tiger', 'Zebra'
]


# =============================
# Embed Background Image (Base64)
# =============================
def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_image = get_base64("myimage.jpg")

st.markdown(f"""
<style>
body, .stApp {{
    background-image: url("data:image/jpg;base64,{bg_image}") !important;
    background-size: cover !important;
    background-position: center !important;
    background-attachment: fixed !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}}

.block-container {{
    background: rgba(0, 0, 0, 0.65);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    max-width: 950px;
    margin: 1rem auto;
    color: white;
}}

.graph-desc {{
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 20px;
    font-size: 1.1rem;
}}

.desc-acc {{
    background: #c6f6d5;
    color: #22543d;
}}

.desc-loss {{
    background: #ffe4ec;
    color: #97266d;
}}

.desc-cm {{
    background: #bee3f8;
    color: #2a4365;
}}

.desc-metrics {{
    background: #fefcbf;
    color: #8a6d3b;
}}

.desc-roc {{
    background: #fbd38d;
    color: #7c4700;
}}

.author-bar {{
    background: linear-gradient(90deg,#86a8e7,#fbc2eb,#fda085);
    color: #22223b;
    font-weight: bold;
    padding: 17px 0;
    border-radius: 15px;
    text-align: center;
    margin-top: 2rem;
    font-size: 1.2rem;
    letter-spacing: 1px;
}}
</style>
""", unsafe_allow_html=True)


# =============================
# Streamlit Tabs and Logic
# =============================
st.title("🐾 Animal Image Classification Dashboard")

tab1, tab2, tab3, tab4 = st.tabs([
    "Upload & Results", "Class Distribution", "Metrics & Charts", "About/Info"
])

if "result_records" not in st.session_state:
    st.session_state["result_records"] = []


# =============================
# TAB 1: Upload & Results
# =============================
with tab1:
    st.header("Upload Animal Images")
    uploaded_files = st.file_uploader(
        "Choose images (JPG/PNG)", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True
    )

    if uploaded_files:
        results = []
        cols = st.columns(3)
        for i, uploaded_file in enumerate(uploaded_files):
            img = Image.open(uploaded_file).convert('RGB')
            img_array = preprocess_image(img)
            preds = model.predict(img_array)[0]
            top_idx = preds.argmax()
            top_conf = preds[top_idx]
            results.append({
                "Filename": uploaded_file.name,
                "Predicted Class": class_names[top_idx],
                "Confidence": float(top_conf)
            })

            with cols[i % 3]:
                st.image(img, width=150, caption=uploaded_file.name)
                st.write(f"**Prediction:** {class_names[top_idx]}")
                st.progress(int(top_conf * 100))

        st.session_state["result_records"] = results
        df_results = pd.DataFrame(results)
        st.dataframe(df_results, use_container_width=True)
        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV of Results", csv, "animal_classification_results.csv", "text/csv")
    else:
        st.write("Upload one or more animal images to begin classification.")


# =============================
# TAB 2: Class Distribution
# =============================
with tab2:
    st.header("Class Prediction Distribution")
    results_df = pd.DataFrame(st.session_state["result_records"])
    if not results_df.empty:
        st.bar_chart(results_df['Predicted Class'].value_counts())
    else:
        st.write("Upload images to see predictions distribution here.")


# =============================
# TAB 3: Metrics & Charts
# =============================
with tab3:
    st.header("Model Performance & Metrics")
    t1, t2, t3, t4 = st.tabs(["Accuracy & Loss", "Confusion Matrix", "Classwise Metrics", "ROC Curves"])

    with t1:
        st.markdown('<div class="graph-desc desc-acc">This plot shows how well the model performs on both the training and validation sets as training progresses. Ideally, both lines should rise and stay close together, which means good generalization without overfitting.</div>', unsafe_allow_html=True)
        st.image("docs/training_validation_accuracy.png", use_container_width=True)
        st.markdown('<div class="graph-desc desc-loss">Lower loss values indicate better performance. Observe the gap between training and validation loss to diagnose overfitting or underfitting.</div>', unsafe_allow_html=True)
        st.image("docs/training_validation_loss.png", use_container_width=True)

    with t2:
        st.markdown('<div class="graph-desc desc-cm">The confusion matrix shows correct and incorrect predictions for each class. High values on the diagonal indicate good classification accuracy per class.</div>', unsafe_allow_html=True)
        st.image("docs/confusion_matrix.png", use_container_width=True)

    with t3:
        st.markdown('<div class="graph-desc desc-metrics">This chart details the precision, recall, and F1-score for each class, highlighting per-class model performance.</div>', unsafe_allow_html=True)
        st.image("docs/classwise_metrics.png", use_container_width=True)

    with t4:
        st.markdown('<div class="graph-desc desc-roc">ROC curves show the trade-off between true positive and false positive rates for each class. An AUC near 1 means excellent classifier quality.</div>', unsafe_allow_html=True)
        st.image("docs/roc_curves.png", use_container_width=True)


# =============================
# TAB 4: About
# =============================
with tab4:
    st.header("About this Dashboard")
    st.markdown("""
- **Dataset:** 15 animal classes, 224x224 RGB JPG images.  
- **Model:** MobileNetV2 based transfer learning, final dense classification head.  
- **Functionality:** Batch image upload, predictions with confidence, downloadable CSV results, prediction distribution charts, and rich model evaluation visuals.  
- **Author:** Swaransh Mishra  
- **Tech:** Python, TensorFlow/Keras, Streamlit
    """)

st.markdown('<div class="author-bar"> Made by Swaransh Mishra </div>', unsafe_allow_html=True)
