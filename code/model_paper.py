"""
論文のcnnモデル再現
https://www.arxiv-vanity.com/papers/1903.12258/
"""
import tensorflow as tf
import tensorflow.keras.layers as layers


def create_paper_cnn(input_shape=(80, 80, 3), num_classes=3, activation="softmax"):
    inputs = layers.Input(input_shape)
    x = inputs
    for ch in [32, 48]:
        x = layers.Conv2D(ch, 3, padding="same")(x)
        x = layers.ReLU()(x)
        x = layers.MaxPool2D()(x)
    x = layers.Dropout(0.2)(x)
    for ch in [64, 96]:
        x = layers.Conv2D(ch, 3, padding="same")(x)
        x = layers.ReLU()(x)
        x = layers.MaxPool2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(num_classes, activation=activation)(x)
    return tf.keras.models.Model(inputs, x)


if __name__ == "__main__":
    model = create_paper_cnn()
    model.summary()
