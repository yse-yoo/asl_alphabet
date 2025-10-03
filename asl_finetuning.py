import tensorflow as tf
import numpy as np
import os

IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    "asl_finetune_data",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# 正規化
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))