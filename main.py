import sys
import argparse
from mnist_model import Mnist_Model
import tensorflow as tf
from bandit import Bandit
from utils import split_uniform_datasets, split_datasets_by_label
import time
import os
import pickle
# Parse arguments
parser = argparse.ArgumentParser(description='Bandit')
parser.add_argument('--bandit_number', type=int, default=3, help='Number of peers')
parser.add_argument('--ticket_number', type=int, default=3, help='Ticket Number')
parser.add_argument('--rounds', type=int, default=3, help='Number of aggregation rounds')
parser.add_argument('--epochs_per_round', type=int, default=7, help='Number of aggregation rounds')
parser.add_argument('--dirname', type=str, default=str(time.time()).split(".")[0], help='Name for the directory where the results will be stored')
args = parser.parse_args()

local_epochs_per_round = args.epochs_per_round


#load dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Displays a few image and their labels
import matplotlib.pyplot as plt
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
    plt.title(y_train[i])
    print(y_train[i])   
plt.show()

print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)
# splits dataset
#x_train_list, y_train_list, x_test_list, y_test_list = split_uniform_datasets(x_train, x_test, y_train, y_test, args.bandit_number)

x_train_list, y_train_list, x_test_list, y_test_list = split_datasets_by_label(x_train, x_test, y_train, y_test, args.bandit_number, 3)

print(Mnist_Model())


bandits = []

for i in range(args.bandit_number):
    bandits.append(Bandit(i, args.ticket_number, x_train_list[i], y_train_list[i], x_test_list[i], y_test_list[i]))
    bandits[i].set_model( Mnist_Model())

# Set up relationships
for i in range(args.bandit_number):
    for j in range(args.bandit_number):
        if i != j:
            bandits[i].relationships[j] = 1
            
for i in range(args.rounds):
    # Train each bandit for one agg round
    for j in range(args.bandit_number):
        bandits[j].train(bandits[j].x_train, bandits[j].y_train, local_epochs_per_round)
        bandits[j].evaluate(bandits[j].x_test, bandits[j].y_test, True, f"train_round_{i}")
        bandits[j].sample_ticket_allocation()
        print("Bandit ", j, " finished training for round ", i)
        sys.stdout.flush()

    # Evaluate each bandit's contribution
    for i in range(args.bandit_number):
        bandits[i].sample_ticket_allocation()
        for j in bandits[i].get_ticket_allocation():
            
            print("Bandit ", i, " evaluating Bandit ", j, "'s contribution")
            sys.stdout.flush()

            bandits[i].update_relationships(bandits[j])
            print("Bandit ", i, " finished evaluating Bandit ", j, "'s contribution")
            sys.stdout.flush()
        bandits[i].evaluate(bandits[i].x_test, bandits[i].y_test, True, "aggregation_round_" + str(i))



    bandits[i].train(bandits[i].x_train, bandits[i].y_train, local_epochs_per_round)
    bandits[i].evaluate(bandits[i].x_test, bandits[i].y_test, True, f"train_round_{i}")
    bandits[i].sample_ticket_allocation()
    print("Bandit ", i, " finished training")
    sys.stdout.flush()

os.mkdir(args.dirname)

for i in range(args.bandit_number):
    history = bandits[i].get_history()
    print(history)
    with open(args.dirname + "/loss_"+ str(i) + ".txt", "wb") as f:
        pickle.dump(history, f)
        
    with open(args.dirname + "/relationships"+ str(i) + ".txt", "wb") as f:
        pickle.dump(bandits[i].get_relationship_history(), f)

    with open(args.dirname + "/accuracy"+ str(i) + ".txt", "wb") as f:
        pickle.dump(bandits[i].get_accuracy_history(), f)