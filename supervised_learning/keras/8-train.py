#!/usr/bin/env python3
"""8-train.py"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """Train model"""
    callbacks = []
    if early_stopping and validation_data is not None:
        stop = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )
        callbacks.append(stop)

    if learning_rate_decay and validation_data is not None:
        def schedule(epoch, lr):
            """Inverse time decay: new_lr = alpha / (1 + decay_rate * epoch)"""
            new_lr = alpha / (1 + decay_rate * epoch)
            return new_lr

        lr_decay = K.callbacks.LearningRateScheduler(schedule, verbose=1)
        callbacks.append(lr_decay)

    if save_best and filepath is not None:
        checkpoint = K.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor='val_loss',
            mode='min',
            save_best_only=save_best,
        )
        callbacks.append(checkpoint)

    history = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        verbose=verbose,
        shuffle=shuffle,
        callbacks=callbacks
    )

    return history
