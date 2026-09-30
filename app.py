import os
import tempfile

import streamlit as st
import pandas as pd

from PIL import Image

from src.predict import predict_image, explain_image, load_pipeline
from src.quantum_model import draw_circuit


st.set_page_config(
    page_title="Hybrid QML Image Classifier",
    page_icon="⚛️",
    layout="wide"
)


st.title("⚛️ Hybrid Quantum-Classical Image Classifier")

st.write(
    "ResNet18 → trainable projection (512 → 4 angles) → 4-qubit variational circuit "
    "→ ⟨Z⟩ readout → cat / dog"
)


@st.cache_resource
def _warm_pipeline():
    # Load the model + ResNet once per session.
    return load_pipeline()


pipeline = _warm_pipeline()


uploaded_files = st.file_uploader(
    "Upload up to 2 images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)


if uploaded_files:

    uploaded_files = uploaded_files[:2]

    cols = st.columns(len(uploaded_files))

    for i, uploaded_file in enumerate(uploaded_files):

        with cols[i]:

            st.image(uploaded_file, caption=f"Image {i + 1}", width="stretch")

            image = Image.open(uploaded_file).convert("RGB")

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                temp_path = tmp.name
                image.save(temp_path)

            try:
                if st.button(f"Classify Image {i + 1}", key=f"button_{i}"):

                    with st.spinner("Running quantum-classical pipeline..."):

                        result = predict_image(temp_path)
                        inside = explain_image(temp_path)

                    st.success(f"Prediction: {result['class']}")

                    st.metric(
                        "Confidence",
                        f"{result['confidence'] * 100:.2f}%"
                    )

                    st.subheader("Class Probabilities")
                    st.write(result["probabilities"])

                    st.subheader("Quantum Features  [⟨Z0⟩, ⟨Z1⟩, ⟨Z2⟩, ⟨Z3⟩]")
                    st.write(result["quantum_features"])

                    if inside is not None:
                        with st.expander("Inside the quantum circuit"):
                            model = pipeline["model"]
                            st.caption(
                                "Rotation angles the trained projection sends to the "
                                f"{model.n_qubits} qubits (radians):"
                            )
                            st.write(inside["angles"])

                            st.caption("⟨Z⟩ of every qubit after each circuit layer "
                                       "(the last layer is what the readout sees):")
                            st.bar_chart(pd.DataFrame(
                                inside["trace"].T,
                                index=[f"q{q}" for q in range(model.n_qubits)],
                                columns=[f"after layer {k + 1}" for k in range(model.n_layers)],
                            ))

                            st.caption(f"Circuit ({model.n_layers} layers, "
                                       f"{model.entanglement} entanglement):")
                            st.code(draw_circuit(model.n_qubits, model.n_layers,
                                                 model.entanglement))
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
