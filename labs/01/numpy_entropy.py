#!/usr/bin/env python3
import numpy as np
def entropy(prob):
    entropy = 0
    for p in prob:
        if p != 0:
            entropy+= -p*np.log(p)
    return entropy


def kldiverfence(data, model):
    kldiv = 0
    for i in range(len(data)):
        p_data = data[i]
        p_model = model[i]
        if p_model==0:
            return np.inf
        if p_data==0:
            continue
        kldiv += p_data * np.log(p_data/p_model)
    return kldiv


def cross_entropy(data, model):
    ent = entropy(data)
    kldiv = kldiverfence(data, model)
    return ent + kldiv 


if __name__ == "__main__":
    # Load data distribution, each data point on a line
    data_list = []
    example =  "_2"
    path = "numpy_entropy_eval_examples/"
    #path, example = "",""
    with open(path + "numpy_entropy_data"+example+".txt", "r") as data:
        for line in data:
            line = line.rstrip("\n")
            data_list.append(line)
            # TODO: process the line
    # TODO: Create a NumPy array containing the data distribution
    tmp = {}
    # Load model distribution, each line `word \t probability`, creating
    # a NumPy array containing the model distribution
    with open(path + "numpy_entropy_model"+example+".txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")
            line = line.split("\t")
            tmp[line[0]]= float(line[1])
            # TODO: process the line
    data_distrib, data_model = [], []
    all_values = set(data_list).union(set(tmp.keys()))
    for event in all_values:
        data_distrib.append(data_list.count(event)/len(data_list))
        if event in tmp.keys():
            prob = tmp[event]
        else:
            prob = 0
        data_model.append(prob)
    # print(data_distrib, data_model)
    # TODO: Compute and print entropy H(data distribution)
    entrop = entropy(data_distrib)
    print("{:.2f}".format(entrop))

    # TODO: Compute and print cross-entropy H(data distribution, model distribution)
    # and KL-divergence D_KL(data distribution, model_distribution)
    cross_enropy = cross_entropy(data_distrib, data_model)
    print("{:.2f}".format(cross_enropy))
    kldiv = kldiverfence(data_distrib, data_model)
    print("{:.2f}".format(kldiv))