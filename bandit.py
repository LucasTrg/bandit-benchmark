import tensorflow as tf
import random as rand
import numpy as np
import copy

class Bandit(object):
    

    def __init__(self, bandit_id, ticket_number, x_train, y_train, x_test, y_test, gamma =0.5, eta=1) -> None:
        self.bandit_id = bandit_id
        self.ticket_number = ticket_number
        self.gamma = gamma
        self.eta = eta
        self.ticket_allocation = {}
        self.relationships = {}
        self.loss_history = []
        self.accuracy_history = []
        self.relationship_history = []

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


    def utility(self, other, tickets):
        score = -  self.model.evaluate(self.x_test, self.y_test)[1]
        for self_layer, other_layer in zip(self.model.get_layers(), other.model.get_layers()):
            if hasattr(self_layer, 'get_weights'):
                weights = zip(self_layer.get_weights(), other_layer.get_weights())
                updated_weights = [(1-self.gamma)*self_weight+self.gamma*tickets/self.ticket_number*other_weight
                for self_weight, other_weight in weights] 
                self_layer.set_weights(updated_weights)

        score += self.model.evaluate(self.x_test, self.y_test)[1]
        return score
    

    def update_relationships(self, other):
        self.relationships[other.bandit_id] = self.relationships.get(other.bandit_id, 0)*np.exp(self.eta*self.utility(other, self.ticket_allocation[other.bandit_id]))

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

    # Randomly sample ticket allocation from the weighted relationships
    def sample_ticket_allocation(self):
        self.ticket_allocation = {}
        self.normalize_relationships()
        self.relationship_history.append(copy.deepcopy(self.relationships))
        distributed_tickets = 0
        while distributed_tickets < self.ticket_number:
            # uses the relationship weights to randomly distribute a ticket to another bandit
            ticket_rd= rand.random()
            acc=0
            for i in self.relationships:
                acc += self.relationships[i]
                if ticket_rd < acc:
                    self.ticket_allocation[i] = self.ticket_allocation.get(i, 0) + 1
                    distributed_tickets += 1
                    break
        print(f"Node {i} distributed {self.ticket_allocation}")

    