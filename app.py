import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10

# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="CNN CIFAR-10 Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ======================================
# CSS
# ======================================

def load_css(file_name):

    with open(file_name) as f:

        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css("style.css")

# ======================================
# LOAD MODEL
# ======================================
import os

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "cnn_model.h5"
)

if not os.path.exists(MODEL_PATH):
    from implementation.train_model import *

model = load_model(MODEL_PATH)
# ======================================
# LOAD DATASET
# ======================================

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train_norm = x_train / 255.0
x_test_norm = x_test / 255.0

# ======================================
# CLASS NAMES
# ======================================

class_names = [

    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"

]

# ======================================
# MODEL EVALUATION
# ======================================

loss, accuracy = model.evaluate(
    x_test_norm,
    y_test,
    verbose=0
)

# ======================================
# HEADER
# ======================================

st.markdown("""

<div class="main-header">

<h1>🧠 CNN CIFAR-10 Dashboard</h1>

<p>
Image Classification using Convolutional Neural Networks
</p>

</div>

""", unsafe_allow_html=True)

# ======================================
# SIDEBAR
# ======================================

st.sidebar.title("⚙️ CNN Hyperparameters")

st.sidebar.info("""

Filters : 32 → 64 → 64

Kernel Size : 3 × 3

Pooling : MaxPooling2D

Activation : ReLU

Optimizer : Adam

Loss : Sparse Categorical Crossentropy

Epochs : 10

Batch Size : 64

Dropout : 0.3

""")

st.sidebar.markdown("---")

prediction_mode = st.sidebar.radio(

    "Prediction Mode",

    [

        "CIFAR Test Image",

        "Upload Image"

    ]

)

# ======================================
# CIFAR TEST IMAGE MODE
# ======================================

if prediction_mode == "CIFAR Test Image":

    index = st.sidebar.slider(

        "Select Test Image",

        0,

        len(x_test)-1,

        0

    )

    image = x_test[index]

    actual_class = class_names[
        y_test[index][0]
    ]

    st.sidebar.image(

        image,

        caption=f"Actual: {actual_class}",

        use_container_width=True

    )

    prediction = model.predict(

        np.expand_dims(
            x_test_norm[index],
            axis=0
        ),

        verbose=0

    )

    pred_class = np.argmax(
        prediction
    )

    confidence = np.max(
        prediction
    )

    st.sidebar.markdown(

        f"""
        <div class="prediction-box">

        <h2>
        Predicted:
        {class_names[pred_class]}
        </h2>

        <h4>
        Confidence:
        {confidence:.2%}
        </h4>

        </div>
        """,

        unsafe_allow_html=True

    )

# ======================================
# UPLOAD IMAGE MODE
# ======================================

else:

    uploaded_file = st.sidebar.file_uploader(

        "Choose Image",

        type=["jpg","jpeg","png"]

    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        )

        st.sidebar.image(

            image,

            caption="Uploaded Image",

            use_container_width=True

        )

        image = image.convert("RGB")

        image = image.resize(
            (32,32)
        )

        image_array = np.array(
            image
        )

        image_array = image_array.astype(
            "float32"
        ) / 255.0

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        prediction = model.predict(
            image_array,
            verbose=0
        )

        pred_class = np.argmax(
            prediction
        )

        confidence = np.max(
            prediction
        )

        st.sidebar.markdown(

            f"""
            <div class="prediction-box">

            <h2>
            Predicted:
            {class_names[pred_class]}
            </h2>

            <h4>
            Confidence:
            {confidence:.2%}
            </h4>

            </div>
            """,

            unsafe_allow_html=True

        )
# ======================================
# DATASET OVERVIEW
# ======================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.subheader("📊 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Training Images",
    x_train.shape[0]
)

c2.metric(
    "Testing Images",
    x_test.shape[0]
)

c3.metric(
    "Classes",
    10
)

c4.metric(
    "Accuracy",
    f"{accuracy:.2%}"
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.subheader("🎯 CIFAR Sample Predictions")

cols = st.columns(4)

for col in cols:

    idx = np.random.randint(
        0,
        len(x_test)
    )

    pred = model.predict(

        np.expand_dims(
            x_test_norm[idx],
            axis=0
        ),

        verbose=0

    )

    pred_class = np.argmax(
        pred
    )

    col.image(
        x_test[idx]
    )

    col.success(
        class_names[pred_class]
    )

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

# ======================================
# SAMPLE IMAGES
# ======================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.subheader("🖼️ CIFAR-10 Sample Images")

fig, axes = plt.subplots(
    3,
    4,
    figsize=(10,7)
)

for i, ax in enumerate(
    axes.flat
):

    ax.imshow(
        x_train[i]
    )

    ax.set_title(
        class_names[
            y_train[i][0]
        ]
    )

    ax.axis("off")

st.pyplot(fig)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

# ======================================
# CLASS DISTRIBUTION
# ======================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.subheader("📈 Class Distribution")

counts = pd.Series(
    y_train.flatten()
).value_counts().sort_index()

fig, ax = plt.subplots(
    figsize=(10,5)
)

sns.barplot(
    x=class_names,
    y=counts.values,
    ax=ax
)

plt.xticks(
    rotation=30
)

st.pyplot(fig)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

# ======================================
# CNN ARCHITECTURE
# ======================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.subheader("🏗️ CNN Architecture")

architecture = pd.DataFrame({

    "Layer":[

        "Conv2D",
        "MaxPooling2D",
        "Conv2D",
        "MaxPooling2D",
        "Conv2D",
        "Flatten",
        "Dense",
        "Dropout",
        "Dense"

    ],

    "Details":[

        "32 Filters (3x3)",
        "Pool Size (2x2)",
        "64 Filters (3x3)",
        "Pool Size (2x2)",
        "64 Filters (3x3)",
        "Flatten Output",
        "64 Neurons",
        "0.3",
        "10 Classes"

    ]

})

st.table(
    architecture
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

# ======================================
# MODEL METRICS
# ======================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.subheader("📊 Model Metrics")

m1, m2 = st.columns(2)

m1.metric(
    "Test Accuracy",
    f"{accuracy:.2%}"
)

m2.metric(
    "Test Loss",
    f"{loss:.4f}"
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

# ======================================
# RANDOM TEST PREDICTIONS
# ======================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.subheader("🔍 Sample Predictions")

indices = np.random.choice(
    len(x_test),
    8,
    replace=False
)

fig, axes = plt.subplots(
    2,
    4,
    figsize=(10,6)
)

for ax, idx in zip(
    axes.flat,
    indices
):

    pred = model.predict(
        np.expand_dims(
            x_test_norm[idx],
            axis=0
        ),
        verbose=0
    )

    pred_class = np.argmax(
        pred
    )

    ax.imshow(
        x_test[idx]
    )

    ax.set_title(
        class_names[pred_class]
    )

    ax.axis("off")

st.pyplot(fig)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)