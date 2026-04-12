#!/usr/bin/env python3
"""0-flip.py"""

def flip_image(image):
    """Horizontal Flip Aug"""

    return tf.image.flip_left_right(image)
