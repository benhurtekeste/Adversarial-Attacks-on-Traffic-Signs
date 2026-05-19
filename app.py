import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from PIL import Image

from art.estimators.classification import PyTorchClassifier

from utils.model import GTSRB_CNN
from utils.preprocessing import preprocess_image
from utils.preprocessing import tensor_to_numpy
from utils.attacks import get_attack
from utils.labels import CLASS_NAMES
from utils.visualization import create_heatmap
from utils.visualization import compute_metrics

st.set_page_config(
    page_title="Adversarial Traffic Sign Evaluation",
    page_icon="🚦",
    layout="wide",
)


st.markdown(
    """
    <style>

    .main {
        background-color: #0e1117;
        color: white;
    }

    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-size: 18px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():

    MODEL_PATH = "models/gtsrb_cnn.pth"

    model = GTSRB_CNN(43)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=torch.device("cpu"),
        )
    )

    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 48, 48),
        nb_classes=43,
        clip_values=(0.0, 1.0),
    )

    return model, classifier


model, classifier = load_model()


st.title("🚦 Adversarial Traffic Sign Evaluation Platform")

st.markdown("Real adversarial attacks against a GTSRB CNN classifier using ART.")


with st.sidebar:

    st.header("Attack Configuration")

    attack_name = st.selectbox("Select Attack", ["FGSM", "PGD", "C&W", "DeepFool"])

    epsilon = st.slider(
        "Epsilon",
        0.001,
        0.3,
        0.05,
        0.001,
    )

    uploaded_file = st.file_uploader(
        "Upload Traffic Sign Image", type=["png", "jpg", "jpeg"]
    )


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    image_tensor = preprocess_image(image)

    with torch.no_grad():

        output = model(image_tensor)

        probs = torch.softmax(output, dim=1)

        confidence, pred = torch.max(probs, 1)

    original_class = CLASS_NAMES[pred.item()]

    original_conf = confidence.item()

    attack = get_attack(
        attack_name,
        classifier,
        epsilon,
    )

    x_adv = attack.generate(x=image_tensor.numpy())

    adv_tensor = torch.tensor(x_adv)

    with torch.no_grad():

        adv_output = model(adv_tensor)

        adv_probs = torch.softmax(adv_output, dim=1)

        adv_conf, adv_pred = torch.max(adv_probs, 1)

    adversarial_class = CLASS_NAMES[adv_pred.item()]

    adversarial_conf = adv_conf.item()

    original_np = tensor_to_numpy(image_tensor)

    adversarial_np = tensor_to_numpy(adv_tensor)

    heatmap = create_heatmap(
        original_np,
        adversarial_np,
    )

    metrics = compute_metrics(
        original_np,
        adversarial_np,
    )

    st.header("Attack Results")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader("Original")

        st.image(original_np)

        st.success(f"{original_class} ({original_conf:.2%})")

    with col2:

        st.subheader("Perturbation Heatmap")

        fig, ax = plt.subplots()

        ax.imshow(heatmap, cmap="hot")

        ax.axis("off")

        st.pyplot(fig)

    with col3:

        st.subheader("Adversarial")

        st.image(adversarial_np)

        if original_class != adversarial_class:

            st.error(f"{adversarial_class} ({adversarial_conf:.2%})")

        else:

            st.warning(f"{adversarial_class} ({adversarial_conf:.2%})")

    st.divider()

    st.header("Attack Metrics")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("L2", f"{metrics['L2']:.4f}")

    m2.metric("Linf", f"{metrics['Linf']:.4f}")

    m3.metric("SSIM", f"{metrics['SSIM']:.4f}")

    m4.metric("PSNR", f"{metrics['PSNR']:.2f}")

    st.divider()

    st.header("Confidence Comparison")

    topk = 5

    original_topk = torch.topk(probs, topk)

    adv_topk = torch.topk(adv_probs, topk)

    original_classes = [CLASS_NAMES[i.item()] for i in original_topk.indices[0]]

    original_scores = [x.item() for x in original_topk.values[0]]

    adv_classes = [CLASS_NAMES[i.item()] for i in adv_topk.indices[0]]

    adv_scores = [x.item() for x in adv_topk.values[0]]

    df1 = pd.DataFrame(
        {
            "Class": original_classes,
            "Confidence": original_scores,
        }
    )

    df2 = pd.DataFrame(
        {
            "Class": adv_classes,
            "Confidence": adv_scores,
        }
    )

    c1, c2 = st.columns(2)

    with c1:

        st.subheader("Original Prediction Distribution")

        fig1 = px.bar(
            df1,
            x="Class",
            y="Confidence",
        )

        st.plotly_chart(fig1, use_container_width=True)

    with c2:

        st.subheader("Adversarial Prediction Distribution")

        fig2 = px.bar(
            df2,
            x="Class",
            y="Confidence",
        )

        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    st.header("Attack Summary")

    attack_success = original_class != adversarial_class

    if attack_success:

        st.error("Attack Successful")

    else:

        st.success("Attack Failed")

    st.markdown(f"### Attack Type: {attack_name}")

    st.markdown(f"### Epsilon: {epsilon}")

    st.markdown(f"### Confidence Drop: " f"{(original_conf - adversarial_conf):.4f}")

else:

    st.info("Upload a traffic sign image to begin.")
