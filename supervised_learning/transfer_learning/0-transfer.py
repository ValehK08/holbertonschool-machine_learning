#!/usr/bin/env python3
'''module documented'''
from tensorflow import keras as K


def preprocess_data(X, Y):
    """Preprocess"""
    X_p = X.astype("float32")
    y_p = K.ops.one_hot(Y.squeeze(), 10)
    return X_p, y_p

if __name__ == "__main__":
    def main():
        (X_train, y_train), (X_test, y_test) = K.datasets.cifar10.load_data()
        X_train, y_train = preprocess_data(X_train, y_train)
        X_test, y_test = preprocess_data(X_test, y_test)
        inputs = K.Input(shape=(32, 32, 3))
        x = K.layers.Lambda(
            lambda x: tf.image.resize(x,
                                      (299, 299)))(inputs)
        x = K.applications.inception_resnet_v2.preprocess_input(x)

        base_model = K.applications.InceptionResNetV2(include_top=False,
                                                      weights="imagenet",
                                                      input_tensor=None,
                                                      input_shape=None,
                                                      pooling=None,
                                                      classes=1000,
                                                      classifier_activation="softmax",
                                                      name="inception_resnet_v2",)
        base_model.trainable = False
        x = base_model(x)
        x = K.layers.GlobalAveragePooling2D()(x)
        x = K.layers.Dense(1024, activation="relu")(x)
        preds = K.layers.Dense(10, activation="softmax")(x)
        model = K.models.Model(inputs=inputs, outputs=preds)

        model.compile(optimizer="adam",
                      loss='categorical_crossentropy',
                      metrics=["accuracy"])
        model.fit(X_train, y_train,
                  validation_data=(X_test, y_test),
                  epochs=10,
                  batch_size=32)
        model.evaluate(X_test, y_test)
        model.save("cifar10.h5")
