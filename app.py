import os
import tempfile

import streamlit as st
import pandas as pd

from PIL import Image

from src.config import CFG
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
    "→ ⟨Z⟩ readout → cat / dog, plus the full noise / mitigation / real-hardware study "
    "(Phases 1-5)."
)


def result_path(name):
    return os.path.join(CFG.results_dir, name)


def show_md(name):
    path = result_path(name)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            st.markdown(fh.read())
    else:
        st.info(f"{name} not found - run the corresponding stage first (see README).")


def show_images(*names, width=None):
    cols = st.columns(len(names))
    for col, name in zip(cols, names):
        path = result_path(name)
        with col:
            if os.path.exists(path):
                st.image(path, width=width or "stretch")
            else:
                st.info(f"{name} not found.")


tab_demo, tab_12, tab_3, tab_4, tab_5 = st.tabs([
    "🖼️ Live Demo", "1-2 · Model", "3 · Noise", "4 · Mitigation", "5 · Real Hardware",
])


# --------------------------------------------------------------------------- Live Demo
with tab_demo:

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


# ------------------------------------------------------------------- Phase 1-2: Model
with tab_12:
    st.header("Phase 1-2 — hybrid model")
    st.write(
        "ResNet18 (frozen) → StandardScaler → trainable Linear(512→4) → BatchNorm → "
        "4-qubit circuit (3 layers, data re-uploading, ring entanglement) → Linear(4→2)."
    )

    metrics_path = result_path("metrics.csv")
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        p12 = df[(df["phase"] == "1-2") & (df["eval"].isin(["cv", "test"]))]
        p12 = p12.sort_values(["circuit", "eval"]).drop_duplicates(
            subset=["circuit", "eval"], keep="last"
        )
        st.subheader("Model comparison (5-fold CV + held-out test)")
        st.dataframe(
            p12[["circuit", "eval", "accuracy", "accuracy_std", "roc_auc"]]
              .rename(columns={"eval": "split"})
              .reset_index(drop=True),
            width="stretch",
        )
        st.caption(
            "`none` = classical baseline (no circuit), `fixed` = frozen random circuit "
            "(Phase 1), `trained` = the trained VQC used everywhere else in this app (Phase 2)."
        )
    else:
        st.info("metrics.csv not found - run `python -m src.evaluate` first.")

    st.subheader("Confusion matrices (held-out test)")
    show_images("confusion_none.png", "confusion_fixed.png", "confusion_trained.png")
    st.subheader("ROC curves (held-out test)")
    show_images("roc_none.png", "roc_fixed.png", "roc_trained.png")


# --------------------------------------------------------------------- Phase 3: Noise
with tab_3:
    st.header("Phase 3 — effect of quantum noise")
    show_md("phase3_summary.md")
    st.subheader("Test accuracy vs noise")
    show_images("phase3_accuracy.png")
    st.subheader("ROC-AUC vs noise")
    show_images("phase3_auc.png")
    st.subheader("Signal strength and prediction imbalance")
    show_images("phase3_signal.png")


# --------------------------------------------------------------- Phase 4: Mitigation
with tab_4:
    st.header("Phase 4 — mitigation and circuit design")
    show_md("phase4_summary.md")
    st.subheader("Accuracy under noise, with each mitigation")
    show_images("phase4_mitigation.png")
    st.subheader("Zero-noise extrapolation: RMSE of ⟨Z⟩ vs the noise-free value")
    show_images("phase4_zne.png")
    st.subheader("Circuit design sweep (depth × entanglement)")
    show_images("phase4_design.png")


# ---------------------------------------------------------- Phase 5: Real Hardware
with tab_5:
    st.header("Phase 5 — real IBM Quantum hardware")
    show_md("phase5_summary.md")
    st.subheader("Ideal vs noisy simulator vs real hardware")
    show_images("phase5_comparison.png")
    st.subheader("Gate-folding ZNE on real hardware")
    show_images("phase5_zne.png")
