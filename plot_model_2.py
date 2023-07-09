import visualkeras
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models, Sequential
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.utils import plot_model

from cifar10_model import Cifar10_Model

model = Cifar10_Model()
plot = visualkeras.layered_view(model.model, to_file='output.png', legend=True)

plot.show()

plot_model(model.model, to_file='model.png', show_shapes=True, show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)