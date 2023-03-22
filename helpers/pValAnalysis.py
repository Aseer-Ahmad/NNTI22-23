#pValAnalysis.py

from random import sample
import numpy as np
import torch 
import os

from traitlets import observe
from helpers.CustomAudioDataset import CustomAudioDataset

def getDataSet(sr, transform):
    PARENT_PTH = os.getcwd()
    TEST_PTH = os.path.join(PARENT_PTH, 'data', 'test')
    TRAIN_PTH = os.path.join(PARENT_PTH, 'data', 'train')
    
    trainDataSet = CustomAudioDataset(TRAIN_PTH, sr, transform)
    testDataset = CustomAudioDataset(TEST_PTH, sr, transform)

    return trainDataSet, testDataset

def toTensor(X_train):
    tot_samples = X_train.shape[0]
    a, b        = X_train[0].shape
    inp_flat    = np.zeros((tot_samples, a*b))

    for i, item in enumerate(X_train):
        inp_flat[i, :] =  np.array(item).reshape(-1, )

    return torch.from_numpy(inp_flat)

def getBootstrapSample( trainDataSet_1, trainDataSet_2, replace = True):

    trainDataSet_1 = np.array(trainDataSet_1)
    trainDataSet_2 = np.array(trainDataSet_2)    

    trainDataSet_1_x, trainDataSet_1_y = trainDataSet_1[:, 0], trainDataSet_1[:, 1]
    trainDataSet_2_x, trainDataSet_2_y = trainDataSet_2[:, 0], trainDataSet_2[:, 1]

    trainDataSet_1_x = toTensor(trainDataSet_1_x)
    trainDataSet_2_x = toTensor(trainDataSet_2_x)

    x_len            = trainDataSet_1_x.shape[0]

    sample_ind = np.random.choice(x_len, size = x_len, replcae = replace)  

    x_boot_1   = trainDataSet_1_x[sample_ind, :]
    y_boot_1   = trainDataSet_1_y[sample_ind, :]

    x_boot_2   = trainDataSet_2_x[sample_ind, :]
    y_boot_2   = trainDataSet_2_y[sample_ind, :]

    return x_boot_1, y_boot_1, x_boot_2, y_boot_2

def evaluate_Pvalue(model_1, model_2, transform_1, transform_2, opt_1, opt_2, loss, SR, batch_size, epochs, num_runs, device):

    model_empty_1 = model_1
    model_empty_2 = model_2

    trainDataSet_1, testDataset_1 = getDataSet(SR, transform_1)
    trainDataSet_2, testDataset_2 = getDataSet(SR, transform_2)
    
    model_1 = learn(model_1, loss, opt_1, trainDataSet_1, device, epochs)
    model_2 = learn(model_2, loss, opt_2, trainDataSet_2, device, epochs)

    metrics_dict_1 = test(model_1, testDataset_1)
    metrics_dict_2 = test(model_2, testDataset_2)
    
    observe_diff   = metrics_dict_1['accuracy'] - metrics_dict_2['accuracy']

    diffs = []

    for i in range(num_runs):
        x_boot_1, y_boot_1, x_boot_2, y_boot_2  = getBootstrapSample(trainDataSet_1, trainDataSet_2)

        model_1 = learn(model_empty_1, loss, opt_1, x_boot_1, y_boot_1, device, epochs)
        model_2 = learn(model_empty_2, loss, opt_2, x_boot_2, y_boot_2, device, epochs)

        metrics_dict_1 = test(model_1, testDataset_1)
        metrics_dict_2 = test(model_2, testDataset_2)

        sample_diff   = metrics_dict_1['accuracy'] - metrics_dict_2['accuracy']

        diffs.append(sample_diff)

    p_value = (np.sum(diffs >= observe_diff) + np.sum(diffs <= -observe_diff)) / num_runs

    return p_value

    
def test(model, testDataset):
    # can use data loader

    pass

def learn(model, loss, opt, train_x, train_y, device, epochs, batch_size):
    
    train_size 			= train_x.shape[0]
	total_iter 			= int(np.ceil(train_size / batch_size))

    for epoch in epochs:
        for iter_ in range(total_iter):
            

    



# # Define function to compute p-value for performance difference between two models
# def compute_p_value(model1, model2, X_train, y_train, X_test, y_test, num_samples=1000):
#     score1 = train_and_evaluate(model1, X_train, y_train, X_test, y_test)
#     score2 = train_and_evaluate(model2, X_train, y_train, X_test, y_test)
#     observed_diff = score1 - score2

#     # Generate bootstrap samples and compute distribution of differences
#     diffs = []
#     for i in range(num_samples):
#         sample_indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
#         X_sampled = X_train[sample_indices]
#         y_sampled = y_train[sample_indices]
#         score1_sampled = train_and_evaluate(model1, X_sampled, y_sampled, X_test, y_test)
#         score2_sampled = train_and_evaluate(model2, X_sampled, y_sampled, X_test, y_test)
#         diff = score1_sampled - score2_sampled
#         diffs.append(diff)

#     # Compute the p-value for the observed difference
#     p_value = (np.sum(diffs >= observed_diff) + np.sum(diffs <= -observed_diff)) / num_samples
#     return p_value

# # Compare RNN vs. 1D CNN
# rnn_model = create_rnn_model(input_shape)
# cnn_model = create_cnn_model(input_shape)
# p_value = compute_p_value(rnn_model, cnn_model, X_train, y_train, X_test, y_test)
# print("p-value for difference between RNN and 1D CNN:", p_value)

# # Compare RNN vs. audio transformer
# transformer_model = create_transformer_model(input_shape)
# p_value = compute_p_value(rnn_model, transformer_model, X_train, y_train, X_test, y_test)
# print("p-value for difference between RNN and audio transformer:", p_value)

# # Compare 1D CNN vs. audio transformer
# p_value = compute_p_value(cnn_model, transformer_model, X_train, y_train, X_test, y_test)
# print("p-value for difference between 1D CNN and audio transformer:", p_value)