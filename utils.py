import tensorflow as tf
import random as rand
import numpy as np

def split_uniform_datasets(x_train, x_test, y_train, y_test, bandit_number):
    x_train_list = []
    y_train_list = []
    x_test_list = []
    y_test_list = []
   
    for i in range(bandit_number):
        x_train_list.append(x_train[i*len(x_train)//bandit_number:(i+1)*len(x_train)//bandit_number])
        y_train_list.append(y_train[i*len(y_train)//bandit_number:(i+1)*len(y_train)//bandit_number])
        x_test_list.append(x_test[i*len(x_test)//bandit_number:(i+1)*len(x_test)//bandit_number])
        y_test_list.append(y_test[i*len(y_test)//bandit_number:(i+1)*len(y_test)//bandit_number])   

    return x_train_list, y_train_list, x_test_list, y_test_list

# Assigns a subset of the labels to each bandits, and returns the corresponding train and test data as np arrays
def split_datasets_by_label(x_train, x_test, y_train, y_test, bandit_number, labels_per_bandit):
    x_train_list = []
    y_train_list = []
    x_test_list = []
    y_test_list = []
   
    labels_assigned={}

    

    for bandit in range(bandit_number):
        x_train_list.append([])
        y_train_list.append([])
        x_test_list.append([])
        y_test_list.append([])
        tmp = list(range(10))
        rand.shuffle(tmp)
        labels_assigned[bandit] = tmp[:labels_per_bandit]

    x_train_sorted, y_train_sorted = sort_dataset_per_label(x_train, y_train)
    x_test_sorted, y_test_sorted = sort_dataset_per_label(x_test, y_test)

    # Computes the portion each bandit should have depending on how many shared labels they have
    weights = {}
    for val in labels_assigned.values():
        for label in val:
            if label in weights:
                weights[label] += 1
            else:
                weights[label] = 1

    for label in weights:
        weights[label] = 1/weights[label]
    
    allocated={}
    for label in range(10):
        allocated[label] = 0


    # Assigns a portion data to each bandit based on their assigned labels such that there is no overlap
    for bandit in range(bandit_number):
        for label in labels_assigned[bandit]:
            x_train_list[bandit]
            x_train_sorted[label]
            weights[label]
            allocated[label]
            x_train_list[bandit] += x_train_sorted[label][allocated[label]:int(weights[label]*len(x_train_sorted[label]))+allocated[label]]
            y_train_list[bandit] += y_train_sorted[label][allocated[label]:int(weights[label]*len(y_train_sorted[label]))+allocated[label]]
            x_test_list[bandit] += x_test_sorted[label][allocated[label]:int(weights[label]*len(x_test_sorted[label]))+allocated[label]]
            y_test_list[bandit] += y_test_sorted[label][allocated[label]:int(weights[label]*len(y_test_sorted[label]))+allocated[label]]

            allocated[bandit] += int(weights[label]*len(x_train_sorted[label]))

    return np.array([np.array(x) for  x in x_train_list]) , np.array([np.array(y) for  y in y_train_list]), np.array([np.array(x) for  x in x_test_list]), np.array([np.array(y) for  y in y_test_list])

#Sorts x and y by the label in y, and produces a dictionnary of each label and the corresponding data
def sort_dataset_per_label(x,y):
    sorted_train = {}
    sorted_test = {}
    for i in range(len(x)):
        if y[i] not in sorted_train:
            sorted_train[y[i]] = []
            sorted_test[y[i]] = []
        sorted_train[y[i]].append(x[i])
        sorted_test[y[i]].append(y[i])
   
    return sorted_train, sorted_test