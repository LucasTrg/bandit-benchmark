import tensorflow as tf
from tensorflow.keras.models import load_model

class Mnist_Model(object):
    def __init__(self):
        self.flat=tf.keras.layers.Flatten(input_shape=(28, 28))

        self.l1 = tf.keras.layers.Dense(128, activation='relu')
        self.l2 = tf.keras.layers.Dense(128, activation='relu')
        self.l3 = tf.keras.layers.Dense(10, activation='softmax')
        self.model = tf.keras.models.Sequential([self.flat,self.l1, self.l2, self.l3])
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