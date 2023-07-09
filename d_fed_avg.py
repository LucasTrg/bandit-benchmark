import tensorflow as tf
import random as rand
import numpy as np
import copy

class dFedAvg(object):
    

    def __init__(self, bandit_id, ticket_number, x_train, y_train, x_test, y_test, gamma =0.15, eta=3, seed=42) -> None:
        self.bandit_id = bandit_id
        self.ticket_number = ticket_number
        self.gamma = gamma
        self.eta = eta
        self.ticket_allocation = {}
        self.relationships = {}
        self.loss_history = []
        self.accuracy_history = []
        self.relationship_history = []
        self.communication_history = []
        rand.seed(seed+bandit_id)

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test


    def set_model(self, model):
        self.model = model
    
    def get_weights(self):
        return self.model.get_weights()
    
    def normalize_relationships(self):
        total = 0
        for i in self.relationships:
            total += self.relationships[i]
        for i in self.relationships:
            self.relationships[i] /= total

    # Relation ship is fixed, so we can just update the weights
    def update_relationships(self, other):
        for self_layer, other_layer in zip(self.model.get_layers(), other.model.get_layers()):
            if hasattr(self_layer, 'get_weights'):
                weights = zip(self_layer.get_weights(), other_layer.get_weights())
                updated_weights = [self_weight+1/self.ticket_number*other_weight
                for self_weight, other_weight in weights] 
                self_layer.set_weights(updated_weights)
    
    def scale_model(self):
        for layer in self.model.get_layers():
            if hasattr(layer, 'get_weights'):
                weights = layer.get_weights()
                for i in range(len(weights)):
                    weights[i] *= 1/self.ticket_number

    def evaluate(self, x_test, y_test, write_to_history=False, step='train'):
        if write_to_history:
            loss = self.model.evaluate(x_test, y_test)
            self.loss_history.append((step,loss[0]))
            self.accuracy_history.append((step,loss[1]))
            return loss
        return self.model.evaluate(x_test, y_test)

    def train(self, x_train, y_train, epochs):
        self.model.train(x_train, y_train, epochs=epochs)

    def get_history(self): 
        return self.loss_history

    def get_relationship_history(self):
        return self.relationship_history

    def get_accuracy_history(self):
        return self.accuracy_history

    def get_ticket_allocation(self):
        return self.ticket_allocation

    def get_communication_history(self):
        return self.communication_history

    # gives one ticket for each peer
    def sample_ticket_allocation(self):
        self.ticket_allocation = {}
        for i in self.relationships:
            self.ticket_allocation[i] = 1

        print(f"Node {i} distributed {self.ticket_allocation}")
        self.communication_history.append(copy.deepcopy(self.ticket_allocation))
    