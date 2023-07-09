    
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models, Sequential

class CNNCifar10_Model(object):
    
    def __init__(self):

        self.model = Sequential()

        self.model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(32,32,3)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPooling2D(pool_size=(2,2)))
        self.model.add(layers.Dropout(0.3))

        self.model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPooling2D(pool_size=(2,2)))
        self.model.add(layers.Dropout(0.5))

        self.model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPooling2D(pool_size=(2,2)))
        self.model.add(layers.Dropout(0.5))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(10, activation='softmax'))    # num_classes = 10
        
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, x_train, y_train, epochs):
        self.model.fit(x_train, y_train, epochs=epochs)

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)
    
    def get_history(self):
        return self.model.history.history

    def get_weights(self):
        return self.model.get_weights()
    
    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_layers(self):
        return self.model.layers