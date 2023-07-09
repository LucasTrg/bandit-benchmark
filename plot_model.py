import matplotlib.pyplot as plt
from tensorflow.keras import models

from cifar10_model import Cifar10_Model

def plot_model_architecture(model):
    plt.figure(figsize=(10, 8))
    layers = model.layers
    y_offset = 0
    for layer in layers:
        if isinstance(layer, models.Sequential):
            y_offset = plot_model_architecture(layer) + 50
        else:
            layer_type = type(layer).__name__
            layer_name = layer.name
            input_shape = layer.input_shape[1:]
            output_shape = layer.output_shape[1:]
            plt.text(0, y_offset + (output_shape[0] - input_shape[0]) // 2, f"{layer_type}\n{layer_name}", ha="center", va="center")
            plt.gca().add_patch(plt.Rectangle((0, y_offset), 50, output_shape[0] - input_shape[0], fill=False))
            y_offset += output_shape[0]

    plt.xlim(-50, 100)
    plt.ylim(0, y_offset)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    plt.savefig('model_architecture.png')
# Create an instance of the Cifar10_Model
cifar10_model = Cifar10_Model()



# Plot the model architecture
plot_model_architecture(cifar10_model.model)
