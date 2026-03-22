#!/usr/bin/env python3
"""7-train.py"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """train_model"""
    callbacks = []
    if early_stopping and validation_data is not None:
        stop = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )
        callbacks.append(stop)

    if learning_rate_decay and validation_data is not None:
        def scheduler(epoch, lr):
            """Scheduler"""
            new_lr = alpha / (1 + decay_rate * epoch)
            return new_lr

        lr_decay = K.callbacks.LearningRateScheduler(scheduler, verbose=1)
        callbacks.append(lr_decay)

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
