import numpy as np

def sigmoid(x):
    x = np.clip(x, -500, 500)  # Clip values to prevent overflow
    return 1 / (1 + np.exp(-x))

# Use this sigmoid function wherever applicable in hand_tracker_nms.py
