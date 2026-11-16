import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


# --------------------------------------------------
# Application configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AgriVision-AI",
    page_icon="🌿",
    layout="wide"
)



# AGRIVISION_DARK_DASHBOARD_THEME_V1
st.markdown(
    """
    <style>
    /* App background */
    .stApp {
        background: #0d1117;
        color: #e6edf3;
    }

    /* Main content */
    .block-container {
        max-width: 1350px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Main text */
    h1, h2, h3 {
        color: #f0f6fc !important;
        letter-spacing: -0.5px;
    }

    h1 {
        font-weight: 800 !important;
    }

    p, li, label {
        color: #c9d1d9;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #111820;
        border-right: 1px solid #263544;
    }

    [data-testid="stSidebar"] > div:first-child {
        background: #111820;
    }

    /* File upload */
    [data-testid="stFileUploader"] {
        background: #161e27;
        border: 1px dashed #2ea66b;
        border-radius: 16px;
        padding: 18px;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #4ade80;
    }

    /* Upload drop zone */
    [data-testid="stFileUploaderDropzone"] {
        background: #111820;
        border: 1px solid #293947;
        border-radius: 12px;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: #161e27;
        border: 1px solid #293947;
        border-radius: 14px;
        padding: 22px;
    }

    [data-testid="stMetricValue"] {
        color: #4ade80;
        font-weight: 800;
    }

    /* Images */
    [data-testid="stImage"] img {
        border-radius: 14px;
        border: 1px solid #293947;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background: #161e27;
        border: 1px solid #293947;
        border-radius: 12px;
        overflow: hidden;
    }

    /* Buttons */
    .stButton > button {
        background: #16834b;
        color: #ffffff;
        border: 1px solid #22a45c;
        border-radius: 10px;
        font-weight: 700;
        padding: 0.65rem 1.2rem;
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        background: #1fa45d;
        color: #ffffff;
        border-color: #4ade80;
    }

    /* Alerts */
    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* Horizontal lines */
    hr {
        border-color: #293947;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background: #161e27;
        border-radius: 8px;
        color: #c9d1d9;
    }

    /* Mobile */
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem;
        }

        h1 {
            font-size: 2rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
DEPLOYMENT_DIR = PROJECT_DIR / "deployment"

MODEL_DIR = (
    DEPLOYMENT_DIR
    / "mobilenetv3large_inference_savedmodel"
)

CLASS_PATH = DEPLOYMENT_DIR / "class_names.json"
METADATA_PATH = DEPLOYMENT_DIR / "model_metadata.json"


# --------------------------------------------------
# Load model and deployment configuration
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading AgriVision-AI model...")
def load_model():
    model = tf.saved_model.load(str(MODEL_DIR))
    return model.signatures["serving_default"]


@st.cache_data
def load_configuration():
    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        class_data = json.load(f)

    if isinstance(class_data, list):
        class_names = class_data
    else:
        class_names = class_data["class_names"]

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    assert len(class_names) == 15
    assert metadata["input"]["shape"] == [None, 224, 224, 3]
    assert metadata["input"]["pixel_value_range"] == [0, 255]

    return class_names, metadata




# AGRIVISION_DISEASE_RECOMMENDATIONS_V1

# --------------------------------------------------
# Disease management knowledge base
# --------------------------------------------------

KNOWLEDGE_PATH = (
    APP_DIR
    / "knowledge_base"
    / "disease_information.json"
)


@st.cache_data
def load_disease_information():
    with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
        information = json.load(f)

    if not isinstance(information, dict):
        raise ValueError("Invalid disease knowledge-base format.")

    return information


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

def preprocess_image(image):
    rgb_image = image.convert("RGB")

    resized_image = rgb_image.resize(
        (224, 224),
        Image.Resampling.BILINEAR
    )

    image_array = np.asarray(
        resized_image,
        dtype=np.float32
    )

    # IMPORTANT:
    # MobileNetV3Large includes its own preprocessing.
    # Do NOT divide pixel values by 255.

    return np.expand_dims(image_array, axis=0)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def predict_image(image, prediction_function, class_names):
    image_batch = preprocess_image(image)

    output = prediction_function(
        plant_image=tf.convert_to_tensor(
            image_batch,
            dtype=tf.float32
        )
    )

    probabilities = next(
        iter(output.values())
    ).numpy()[0]

    assert probabilities.shape == (len(class_names),)
    assert np.all(np.isfinite(probabilities))
    assert np.isclose(probabilities.sum(), 1.0, atol=1e-4)

    top_indices = np.argsort(probabilities)[::-1][:5]

    results = pd.DataFrame({
        "Disease": [
            class_names[int(i)].replace("_", " ")
            for i in top_indices
        ],
        "Confidence (%)": [
            round(float(probabilities[i]) * 100, 2)
            for i in top_indices
        ]
    })

    best_index = int(np.argmax(probabilities))

    return (
        class_names[best_index],
        float(probabilities[best_index]),
        results
    )


# --------------------------------------------------
# User interface
# --------------------------------------------------

st.title("🌿 AgriVision-AI")
st.subheader("AI-Powered Plant Leaf Disease Detection")

st.write(
    "Upload a leaf image to identify potential diseases "
    "using a MobileNetV3Large deep learning model."
)

st.info(
    "Supported plants: Bell Pepper, Potato, and Tomato. "
    "The model recognizes 15 healthy and disease categories."
)

try:
    class_names, metadata = load_configuration()
    disease_information = load_disease_information()
    prediction_function = load_model()

    if set(class_names) != set(disease_information):
        raise ValueError(
            'Disease knowledge base does not match model classes.'
        )

except Exception as exc:
    st.error("Unable to load the deployment model or configuration.")
    st.exception(exc)
    st.stop()


uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image containing a plant leaf."
)


if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file)
        image.load()

    except (UnidentifiedImageError, OSError, ValueError):
        st.error("The uploaded file is not a valid image.")
        st.stop()

    image_column, result_column = st.columns(2)

    with image_column:
        st.image(
            image,
            caption="Uploaded leaf image",
            use_container_width=True
        )

    with result_column:

        with st.spinner("Analyzing leaf image..."):

            try:
                label, confidence, top_results = predict_image(
                    image,
                    prediction_function,
                    class_names
                )

            except Exception as exc:
                st.error("Prediction failed.")
                st.exception(exc)
                st.stop()

        readable_label = label.replace("_", " ")

        if label.lower().endswith("healthy"):
            st.success("🌱 Predicted status: Healthy")
        else:
            st.warning("⚠️ Predicted status: Possible disease")

        st.markdown("### Predicted Class")
        st.write(readable_label)

        st.metric(
            label="Model Confidence",
            value=f"{confidence * 100:.2f}%"
        )

        if confidence < 0.60:
            st.info(
                "The model has relatively low confidence "
                "in this prediction. Consider uploading "
                "a clearer leaf image."
            )

    st.markdown("### 📊 Top Five Predictions")

    st.dataframe(
        top_results,
        hide_index=True,
        use_container_width=True
    )

    chart_data = top_results.set_index("Disease")

    st.bar_chart(
        chart_data["Confidence (%)"]
    )

    st.caption(
        "Confidence values are model softmax scores, "
        "not guarantees of diagnostic accuracy."
    )




    # --------------------------------------------------
    # Disease description and management recommendations
    # --------------------------------------------------

    st.divider()

    st.markdown("## 🌿 Disease Information & Management")

    disease_entry = disease_information.get(label)

    if disease_entry is None:

        st.error(
            "No management information is available "
            "for this predicted class."
        )

    else:

        st.markdown(
            f"### 🌱 {disease_entry['plant']} — "
            f"{disease_entry['disease']}"
        )

        st.markdown("### 🔍 Description")
        st.write(disease_entry["description"])

        if disease_entry["status"] == "healthy":

            st.success(
                "The uploaded leaf was classified as healthy "
                "within the model's supported categories."
            )

        else:

            st.warning(
                "This is an AI-generated disease prediction, "
                "not a confirmed agricultural diagnosis."
            )

        st.markdown("### 🩺 Recommended Management")

        for index, recommendation in enumerate(
            disease_entry["management"],
            start=1
        ):
            st.write(f"{index}. {recommendation}")

        st.markdown("### 🛡️ Prevention Methods")

        for index, prevention_tip in enumerate(
            disease_entry["prevention"],
            start=1
        ):
            st.write(f"{index}. {prevention_tip}")

        st.markdown("### ⚠️ Important Precautions")

        st.info(disease_entry["precautions"])

        st.caption(
            "Recommendations are general educational guidance. "
            "They are associated with the predicted category "
            "and are not personalized treatment prescriptions. "
            "Consult a qualified agricultural professional "
            "before making important crop-treatment decisions."
        )


# --------------------------------------------------
# Model information
# --------------------------------------------------

with st.expander("ℹ️ About the Model"):

    st.write("**Architecture:** MobileNetV3Large")
    st.write("**Dataset:** PlantVillage")
    st.write("**Classes:** 15")
    st.write("**Input size:** 224 × 224 RGB")

    metrics = metadata.get(
        "evaluation", {}
    ).get("recorded_test_metrics", {})

    if "test_accuracy" in metrics:
        st.write(
            "**Recorded test accuracy:** "
            f"{float(metrics['test_accuracy']) * 100:.2f}%"
        )

    if "macro_f1" in metrics:
        st.write(
            "**Recorded macro F1-score:** "
            f"{float(metrics['macro_f1']) * 100:.2f}%"
        )

    st.warning(
        "This model was evaluated on PlantVillage images. "
        "Real-world field performance may differ. "
        "It cannot verify whether an uploaded image "
        "contains a supported plant leaf."
    )


st.divider()

st.caption(
    "AgriVision-AI | AI-assisted plant disease screening. "
    "For important crop treatment decisions, consult "
    "a qualified agricultural professional."
)
