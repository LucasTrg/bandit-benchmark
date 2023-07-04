import sys
import argparse
from cifar10_model import Cifar10_Model
from CNN_cifar10 import CNNCifar10_Model
from mnist_model import Mnist_Model
import tensorflow as tf
from bandit import Bandit
from utils import split_uniform_datasets, split_datasets_by_label, split_datasets_by_label_fixed
import time
import os
import pickle
import json 
# Parse arguments
parser = argparse.ArgumentParser(description='Bandit')
parser.add_argument('--bandit_number', type=int, default=3, help='Number of peers')
parser.add_argument('--ticket_number', type=int, default=3, help='Maximal number of tickets per round')
parser.add_argument('--rounds', type=int, default=3, help='Number of aggregation rounds')
parser.add_argument('--epochs_per_round', type=int, default=7, help='Number of aggregation rounds')
parser.add_argument('--dirname', type=str, default=str(time.time()).split(".")[0], help='Name for the directory where the results will be stored')
parser.add_argument('--dataset_dist', type=str, default="uniform", help='Dataset distribution to use')
parser.add_argument('--dataset', type=str, default="mnist", help='Dataset to use')
parser.add_argument('--eta', type=int, default=1, help='Eta parameter for the bandit algorithm')
parser.add_argument('--model', type=str, default="dense", help='Model architecture to use')
args = parser.parse_args()

local_epochs_per_round = args.epochs_per_round


#load dataset
if args.dataset == "mnist":
    data = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = data.load_data()

elif args.dataset == "cifar10":
    data = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = data.load_data()
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

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

if args.dataset_dist == "random_labels": 
    x_train_list, y_train_list, x_test_list, y_test_list = split_datasets_by_label(x_train, x_test, y_train, y_test, args.bandit_number, 2)
if ".json" in args.dataset_dist:
    with open(args.dataset_dist) as f:
        distr=json.load(f)
        keys = list(distr.keys())
        for key in keys:
            distr[int(key)] = distr.pop(key)
        print("Loaded dataset distribution: ", distr)
        x_train_list, y_train_list, x_test_list, y_test_list = split_datasets_by_label_fixed(x_train, x_test, y_train, y_test, args.bandit_number, distr)
else:
    x_train_list, y_train_list, x_test_list, y_test_list = split_uniform_datasets(x_train, x_test, y_train, y_test, args.bandit_number)



bandits = []

for i in range(args.bandit_number):
    bandits.append(Bandit(i, args.ticket_number, x_train_list[i], y_train_list[i], x_test_list[i], y_test_list[i], eta=args.eta))
    if args.dataset == "mnist":
        bandits[i].set_model( Mnist_Model())
    elif args.dataset == "cifar10":
        if args.model == "dense":
            bandits[i].set_model(Cifar10_Model())
        elif args.model == "cnn":
            bandits[i].set_model(CNNCifar10_Model())

# Set up relationships
for i in range(args.bandit_number):
    for j in range(args.bandit_number):
        if i != j:
            bandits[i].relationships[j] = 1
            
for r in range(args.rounds):
    # Train each bandit for one agg round
    for j in range(args.bandit_number):
        bandits[j].train(bandits[j].x_train, bandits[j].y_train, local_epochs_per_round)
        bandits[j].evaluate(bandits[j].x_test, bandits[j].y_test, True, f"train_round_{r}")
        bandits[j].sample_ticket_allocation()
        print("Bandit ", j, " finished training for round ", r)
        sys.stdout.flush()

    # Evaluate each bandit's contribution
    for i in range(args.bandit_number):
        bandits[i].sample_ticket_allocation()
        for j in bandits[i].get_ticket_allocation():
            if i == j:
                continue
            
            print("Bandit ", i, " evaluating Bandit ", j, "'s contribution")
            sys.stdout.flush()

            bandits[i].update_relationships(bandits[j])
            print("Bandit ", i, " finished evaluating Bandit ", j, "'s contribution")
            sys.stdout.flush()
        bandits[i].evaluate(bandits[i].x_test, bandits[i].y_test, True, "aggregation_round_" + str(r))


# Train each bandit for one last training round
for i in range(args.bandit_number):
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

