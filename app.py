import os
import csv
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

# ==============================
# PlantHealth AI
# ==============================

st.set_page_config(
    page_title="PlantHealth AI",
    page_icon="🌱",
    layout="centered"
)

# ==============================
# Paths
# ==============================

MODEL_PATH = os.path.join(
    "models",
    "planthealth_mobilenetv2_best.keras"
)

CLASS_NAMES_PATH = os.path.join(
    "models",
    "class_names.csv"
)

IMG_SIZE = (224, 224)

# ==============================
# Load Model
# ==============================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


# ==============================
# Load Class Names
# ==============================

@st.cache_data
def load_class_names():

    class_names = []

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            class_names.append(row["class_name"])

    return class_names


model = load_model()
class_names = load_class_names()

# ==============================
# Header
# ==============================

st.title("🌱 PlantHealth AI")

st.subheader("AI-Powered Plant Disease Detection")

st.write(
    "Upload a plant leaf image and PlantHealth AI "
    "will analyze it using a trained MobileNetV2 "
    "deep learning model."
)

st.divider()

# ==============================
# Upload Image
# ==============================

uploaded_file = st.file_uploader(
    "📷 Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

# ==============================
# Prediction
# ==============================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption=uploaded_file.name,
        use_container_width=True
    )

    st.write(
        f"Uploaded file: `{uploaded_file.name}`"
    )

    if st.button(
        "🔍 Detect Disease",
        use_container_width=True
    ):

        with st.spinner(
            "Analyzing leaf image..."
        ):

            # ==============================
            # EXACT SAME PROCESSING AS
            # SUCCESSFUL DIAGNOSTIC TEST
            # ==============================

            image_resized = image.resize(
                IMG_SIZE
            )

            image_array = np.array(
                image_resized,
                dtype=np.float32
            )

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            # IMPORTANT:
            # Do NOT use preprocess_input here.
            # It is already inside the trained model.
            st.write(
    "Pixel range:",
    float(image_array.min()),
    "to",
    float(image_array.max())
)
            predictions = model.predict(
                image_array,
                verbose=0
            )[0]

            # ==============================
            # Prediction
            # ==============================

            predicted_index = int(
                np.argmax(predictions)
            )

            confidence = float(
                predictions[predicted_index]
            )

            predicted_class = class_names[
                predicted_index
            ]

        # ==============================
        # Result
        # ==============================

        st.success(
            "✅ Analysis completed!"
        )

        st.subheader(
            "🌿 Prediction"
        )

        parts = predicted_class.split(
            "___"
        )

        plant_name = (
            parts[0]
            .replace("_", " ")
        )

        condition = (
            parts[1]
            .replace("_", " ")
            if len(parts) > 1
            else "Unknown"
        )

        st.write(
            f"**Plant:** {plant_name}"
        )

        st.write(
            f"**Condition:** {condition}"
        )

        st.write(
            f"**Confidence:** "
            f"{confidence * 100:.2f}%"
        )

        st.progress(
            confidence
        )

        # ==============================
        # Confidence Message
        # ==============================

        if confidence >= 0.80:

            st.success(
                "The model has high confidence "
                "in this prediction."
            )

        elif confidence >= 0.60:

            st.info(
                "The model has moderate confidence. "
                "Consider checking the image quality "
                "or consulting an agricultural expert."
            )

        else:

            st.warning(
                "The model has low confidence. "
                "Try uploading a clearer leaf image."
            )

        # ==============================
        # Top 3 Predictions
        # ==============================

        st.subheader(
            "📊 Top 3 Predictions"
        )

        top_3_indices = np.argsort(
            predictions
        )[-3:][::-1]

        for rank, index in enumerate(
            top_3_indices,
            start=1
        ):

            class_name = class_names[
                index
            ]

            probability = float(
                predictions[index]
            )

            readable_name = (
                class_name
                .replace("___", " - ")
                .replace("_", " ")
            )

            st.write(
                f"**{rank}. {readable_name}** — "
                f"{probability * 100:.2f}%"
            )

            st.progress(
                probability
            )

# ==============================
# About
# ==============================

st.divider()

with st.expander(
    "ℹ️ About PlantHealth AI"
):

    st.write(
        "PlantHealth AI uses a MobileNetV2-based "
        "deep learning model trained to classify "
        "plant leaf images into 38 plant health "
        "categories."
    )

    st.write(
        "The saved model achieved 92.53% accuracy "
        "on the validation dataset."
    )

# ==============================
# Important Note
# ==============================

with st.expander(
    "⚠️ Important Note"
):

    st.write(
        "This application provides an AI-based "
        "prediction and should not replace "
        "professional agricultural diagnosis."
    )

    st.write(
        "For important crop-management decisions, "
        "consult a qualified agricultural expert."
    )