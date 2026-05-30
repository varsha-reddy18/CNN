import os
import tensorflow as tf

from tensorflow.keras.datasets import cifar10

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

# ======================================
# BASE DIRECTORY
# ======================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

# ======================================
# LOAD DATASET
# ======================================

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# ======================================
# NORMALIZATION
# ======================================

x_train = x_train.astype(
    "float32"
) / 255.0

x_test = x_test.astype(
    "float32"
) / 255.0

# ======================================
# CNN MODEL
# ======================================

model = Sequential([

    Conv2D(
        32,
        (3,3),
        activation="relu",
        input_shape=(32,32,3)
    ),

    MaxPooling2D((2,2)),

    Conv2D(
        64,
        (3,3),
        activation="relu"
    ),

    MaxPooling2D((2,2)),

    Conv2D(
        64,
        (3,3),
        activation="relu"
    ),

    Flatten(),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        10,
        activation="softmax"
    )

])

# ======================================
# MODEL SUMMARY
# ======================================

model.summary()

# ======================================
# COMPILE MODEL
# ======================================

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

# ======================================
# TRAIN MODEL
# ======================================

history = model.fit(

    x_train,

    y_train,

    validation_data=(

        x_test,

        y_test

    ),

    epochs=5,

    batch_size=64,

    verbose=1

)

# ======================================
# EVALUATE MODEL
# ======================================

loss, accuracy = model.evaluate(

    x_test,

    y_test,

    verbose=0

)

print(
    f"Test Accuracy : {accuracy:.4f}"
)

# ======================================
# SAVE MODEL
# ======================================

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "cnn_model.h5"
)

model.save(
    MODEL_PATH
)

print(
    f"Model Saved Successfully at {MODEL_PATH}"
)