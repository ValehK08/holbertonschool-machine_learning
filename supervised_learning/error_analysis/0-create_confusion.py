#!/usr/bin/env python3
""" 0-create_confusion.py """


def create_confusion_matrix(labels, logits):
    """ CREATE CONFUSION MATRIX """

    confusion = np.zeros((labels.shape[1], labels.shape[1]))
    for i in range(labels.shape[0]):
        true_label_idx = np.argmax(labels[i])
        predicted_label_idx = np.argmax(logits[i])
        confusion[true_label_idx, predicted_label_idx] += 1
    return confusion
