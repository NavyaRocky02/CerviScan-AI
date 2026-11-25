import os
import json
import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image

# Page configuration
st.set_page_config(
    page_title="CerviScan AI",
    page_icon="🦴",
    layout="wide"
)

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_model.keras"
)

CLASS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "class_names.json"
)

# Load trained model
@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(MODEL_PATH)

    return model


# Load class names
def load_classes():

    if os.path.exists(CLASS_PATH):

        with open(CLASS_PATH, "r") as file:
            return json.load(file)

    return ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]


classes = load_classes()

# Application title
st.title("🦴 CerviScan AI")

st.subheader("Cervical Spine Image Classification Prototype")

st.warning(
    "Research prototype only. This model was trained using "
    "synthetic images, not real CT scans. Predictions have "
    "no diagnostic meaning and must not be used for clinical decisions."
)

st.markdown("---")

st.header("Upload Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg", "bmp"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("L")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    # Image preprocessing
    resized_image = image.resize((224, 224))

    image_array = np.array(
        resized_image,
        dtype=np.float32
    )

    normalized_image = image_array / 255.0

    # Add batch and channel dimensions
    input_image = np.expand_dims(
        normalized_image,
        axis=(0, -1)
    )

    with col2:

        st.image(
            resized_image,
            caption="Preprocessed Image",
            use_container_width=True
        )

    if st.button(
        "Run Experimental Classification",
        type="primary"
    ):

        try:

            if not os.path.exists(MODEL_PATH):
                st.error(
                    "Trained model not found. "
                    "Please train the model first."
                )

            else:

                model = load_model()

                predictions = model.predict(
                    input_image,
                    verbose=0
                )[0]

                predicted_index = int(
                    np.argmax(predictions)
                )

                predicted_class = classes[predicted_index]

                score = float(
                    predictions[predicted_index]
                )

                st.markdown("---")

                st.subheader("Experimental Model Output")

                st.metric(
                    "Predicted Synthetic Class",
                    predicted_class
                )

                st.write(
                    "Model score:",
                    f"{score * 100:.2f}%"
                )

                st.info(
                    "This output represents classification of "
                    "artificial geometric images. It does not "
                    "identify vertebrae or detect fractures."
                )

                st.subheader("Class Scores")

                for i, probability in enumerate(predictions):

                    st.write(classes[i])

                    st.progress(
                        float(probability)
                    )

        except Exception as error:

            st.error(f"Application Error: {error}")


st.markdown("---")

st.caption(
    "CerviScan AI | Experimental Research Prototype | "
    "Not for Medical Diagnosis"
)
