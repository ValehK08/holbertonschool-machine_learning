#!/usr/bin/env python3
"""
Bayesian Hyperparameter Optimization with GPyOpt
Optimizes a CNN on MNIST using Bayesian Optimization
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import GPyOpt


# Reproducibility
np.random.seed(0)
tf.random.set_seed(0)

# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


def build_model(learning_rate,
                units,
                dropout_rate,
                l2_weight):
    """Build CNN model"""

    model = models.Sequential([
        layers.Conv2D(
            32,
            (3, 3),
            activation='relu',
            input_shape=(28, 28, 1)
        ),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),

        layers.Dense(
            units,
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_weight)
        ),

        layers.Dropout(dropout_rate),

        layers.Dense(10, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def objective(params):
    """
    Objective function for Bayesian Optimization
    """

    learning_rate = float(params[0][0])
    units = int(params[0][1])
    dropout_rate = float(params[0][2])
    l2_weight = float(params[0][3])
    batch_size = int(params[0][4])

    model = build_model(
        learning_rate,
        units,
        dropout_rate,
        l2_weight
    )

    checkpoint_name = (
        f"checkpoint_lr{learning_rate:.5f}"
        f"_u{units}"
        f"_d{dropout_rate:.2f}"
        f"_l2{l2_weight:.5f}"
        f"_b{batch_size}.keras"
    )

    checkpoint = ModelCheckpoint(
        checkpoint_name,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=0
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stop],
        verbose=0
    )

    _, accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    # Minimize negative accuracy
    return -accuracy


# Hyperparameter bounds
bounds = [
    {
        'name': 'learning_rate',
        'type': 'continuous',
        'domain': (1e-4, 1e-2)
    },
    {
        'name': 'units',
        'type': 'discrete',
        'domain': (32, 64, 128, 256)
    },
    {
        'name': 'dropout_rate',
        'type': 'continuous',
        'domain': (0.1, 0.5)
    },
    {
        'name': 'l2_weight',
        'type': 'continuous',
        'domain': (1e-6, 1e-2)
    },
    {
        'name': 'batch_size',
        'type': 'discrete',
        'domain': (32, 64, 128, 256)
    }
]

# Bayesian Optimization
optimizer = GPyOpt.methods.BayesianOptimization(
    f=objective,
    domain=bounds,
    acquisition_type='EI',
    exact_feval=True
)

optimizer.run_optimization(max_iter=30)

# Best result
best_params = optimizer.x_opt
best_score = -optimizer.fx_opt

# Save report
with open('bayes_opt.txt', 'w') as file:
    file.write("Bayesian Optimization Report\n")
    file.write("=" * 40 + "\n\n")

    file.write(f"Best Accuracy: {best_score:.4f}\n\n")

    file.write("Best Hyperparameters:\n")
    file.write(f"Learning Rate: {best_params[0]}\n")
    file.write(f"Units: {int(best_params[1])}\n")
    file.write(f"Dropout Rate: {best_params[2]}\n")
    file.write(f"L2 Weight: {best_params[3]}\n")
    file.write(f"Batch Size: {int(best_params[4])}\n")

# Plot convergence
optimizer.plot_convergence()

plt.savefig("convergence_plot.png")
plt.show()

print("Optimization complete.")
print("Report saved to bayes_opt.txt")
print("Convergence plot saved to convergence_plot.png")#!/usr/bin/env python3
"""
Bayesian Hyperparameter Optimization with GPyOpt
Optimizes a CNN on MNIST using Bayesian Optimization
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import GPyOpt


# Reproducibility
np.random.seed(0)
tf.random.set_seed(0)

# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


def build_model(learning_rate,
                units,
                dropout_rate,
                l2_weight):
    """Build CNN model"""

    model = models.Sequential([
        layers.Conv2D(
            32,
            (3, 3),
            activation='relu',
            input_shape=(28, 28, 1)
        ),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),

        layers.Dense(
            units,
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_weight)
        ),

        layers.Dropout(dropout_rate),

        layers.Dense(10, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def objective(params):
    """
    Objective function for Bayesian Optimization
    """

    learning_rate = float(params[0][0])
    units = int(params[0][1])
    dropout_rate = float(params[0][2])
    l2_weight = float(params[0][3])
    batch_size = int(params[0][4])

    model = build_model(
        learning_rate,
        units,
        dropout_rate,
        l2_weight
    )

    checkpoint_name = (
        f"checkpoint_lr{learning_rate:.5f}"
        f"_u{units}"
        f"_d{dropout_rate:.2f}"
        f"_l2{l2_weight:.5f}"
        f"_b{batch_size}.keras"
    )

    checkpoint = ModelCheckpoint(
        checkpoint_name,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=0
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stop],
        verbose=0
    )

    _, accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    # Minimize negative accuracy
    return -accuracy


# Hyperparameter bounds
bounds = [
    {
        'name': 'learning_rate',
        'type': 'continuous',
        'domain': (1e-4, 1e-2)
    },
    {
        'name': 'units',
        'type': 'discrete',
        'domain': (32, 64, 128, 256)
    },
    {
        'name': 'dropout_rate',
        'type': 'continuous',
        'domain': (0.1, 0.5)
    },
    {
        'name': 'l2_weight',
        'type': 'continuous',
        'domain': (1e-6, 1e-2)
    },
    {
        'name': 'batch_size',
        'type': 'discrete',
        'domain': (32, 64, 128, 256)
    }
]

# Bayesian Optimization
optimizer = GPyOpt.methods.BayesianOptimization(
    f=objective,
    domain=bounds,
    acquisition_type='EI',
    exact_feval=True
)

optimizer.run_optimization(max_iter=30)

# Best result
best_params = optimizer.x_opt
best_score = -optimizer.fx_opt

# Save report
with open('bayes_opt.txt', 'w') as file:
    file.write("Bayesian Optimization Report\n")
    file.write("=" * 40 + "\n\n")

    file.write(f"Best Accuracy: {best_score:.4f}\n\n")

    file.write("Best Hyperparameters:\n")
    file.write(f"Learning Rate: {best_params[0]}\n")
    file.write(f"Units: {int(best_params[1])}\n")
    file.write(f"Dropout Rate: {best_params[2]}\n")
    file.write(f"L2 Weight: {best_params[3]}\n")
    file.write(f"Batch Size: {int(best_params[4])}\n")

# Plot convergence
optimizer.plot_convergence()

plt.savefig("convergence_plot.png")
plt.show()

print("Optimization complete.")
print("Report saved to bayes_opt.txt")
print("Convergence plot saved to convergence_plot.png")
