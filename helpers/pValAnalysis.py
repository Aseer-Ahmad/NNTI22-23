from random import sample
import numpy as np
import torch 
import os

from traitlets import observe
from helpers.CustomAudioDataset import CustomAudioDataset

from torch.utils.data import DataLoader

from helpers.metrics import audMetrics


def evaluate_Pvalue(model_1, model_2, transform_1, transform_2, opt_1, opt_2, loss, SR, batch_size, epochs, num_runs, device):

    model_empty_1 = model_1
    model_empty_2 = model_2

    trainDataSet_1, testDataset_1 = getDataSet(SR, transform_1)
    trainDataSet_2, testDataset_2 = getDataSet(SR, transform_2)
    
    trainDataSet_1_x, trainDataSet_1_y = np.array(trainDataSet_1)[:, 0], np.array(trainDataSet_1)[:, 1]
    trainDataSet_2_x, trainDataSet_2_y = np.array(trainDataSet_2)[:, 0], np.array(trainDataSet_2)[:, 1]
    
    print("Training Model-1")

    model_1 = learn(model_1, loss, opt_1, trainDataSet_1_x, trainDataSet_1_y, device, epochs, batch_size)
    metrics_dict_1 = test(model_1, testDataset_1, batch_size, device)
    
    print("Training Model-2")
    
    model_2 = learn(model_2, loss, opt_2, trainDataSet_2_x, trainDataSet_2_y, device, epochs, batch_size)
    metrics_dict_2 = test(model_2, testDataset_2, batch_size, device)
    
    observe_diff   = metrics_dict_1['accuracy'] - metrics_dict_2['accuracy']

    print(f" observe_diff between {model_1.__class__.__name__} and {model_2.__class__.__name__} : {observe_diff}")

    diffs = []
    
    for i in range(num_runs):
        x_boot_1, y_boot_1, x_boot_2, y_boot_2  = getBootstrapSample(trainDataSet_1, trainDataSet_2)

        print(f'Training Bootstapped Model-1 Sample#{i+1}')
        model_1 = learn(model_empty_1, loss, opt_1, x_boot_1, y_boot_1, device, epochs, batch_size)
        metrics_dict_1 = test(model_1, testDataset_1, batch_size, device)

        print(f'Training Bootstapped Model-2 Sample#{i+1}')
        model_2 = learn(model_empty_2, loss, opt_2, x_boot_2, y_boot_2, device, epochs, batch_size)
        metrics_dict_2 = test(model_2, testDataset_2, batch_size, device)

        sample_diff   = metrics_dict_1['accuracy'] - metrics_dict_2['accuracy']

        diffs.append(sample_diff)

        print(f"sample difference between models for bootstrap {i} : {sample_diff}")

    print(diffs)

    p_value = (np.sum(diffs >= observe_diff) + np.sum(diffs <= -observe_diff)) / num_runs

    return p_value

def getDataSet(sr, transform):
    PARENT_PTH = os.getcwd()
    TEST_PTH = os.path.join(PARENT_PTH, 'data', 'test')
    TRAIN_PTH = os.path.join(PARENT_PTH, 'data', 'train')
    
    trainDataSet = CustomAudioDataset(TRAIN_PTH, sr, transform)
    testDataset  = CustomAudioDataset(TEST_PTH, sr, transform)

    return trainDataSet, testDataset

def toTensor(X_train):
    tot_samples = X_train.shape[0]
    tensor_shape = X_train[0].shape
    if len(tensor_shape)==2:
        a,b = tensor_shape
        inp_flat    = np.zeros((tot_samples, a,b))
    if len(tensor_shape)==3:
        a,b,c = tensor_shape
        inp_flat    = np.zeros((tot_samples, a,b,c))

    for i, item in enumerate(X_train):
        inp_flat[i] =  np.array(item)

    return torch.from_numpy(inp_flat)

def getBootstrapSample( trainDataSet_1, trainDataSet_2, replace = True):

    trainDataSet_1 = np.array(trainDataSet_1)
    trainDataSet_2 = np.array(trainDataSet_2)    

    trainDataSet_1_x, trainDataSet_1_y = trainDataSet_1[:, 0], trainDataSet_1[:, 1]
    trainDataSet_2_x, trainDataSet_2_y = trainDataSet_2[:, 0], trainDataSet_2[:, 1]

    trainDataSet_1_x = toTensor(trainDataSet_1_x)
    trainDataSet_2_x = toTensor(trainDataSet_2_x)

    x_len            = trainDataSet_1_x.shape[0]

    sample_ind = np.random.choice(x_len, size = x_len, replace = replace)  

    x_boot_1   = trainDataSet_1_x[sample_ind, :]
    y_boot_1   = trainDataSet_1_y[sample_ind]

    x_boot_2   = trainDataSet_2_x[sample_ind, :]
    y_boot_2   = trainDataSet_2_y[sample_ind]

    return x_boot_1, y_boot_1, x_boot_2, y_boot_2
    
def test(model, testDataset, batch_size, device):

    model.eval()

    testLoader = DataLoader(testDataset, batch_size  = batch_size, shuffle = True)
    tot_batches  = len(testLoader)
    lenDataSet   = len(testLoader)
    np_preds     = np.zeros((len(testDataset)))
    
    metrics_dict = None

    runn_acc    = 0
    runn_prec   = 0
    runn_rec    = 0
    runn_f1     = 0

    with torch.no_grad():
        for i, (aud, labels) in enumerate(testLoader, start=0):
            aud      = aud.to(device)
            labels   = labels.to(device)
            outputs  = model(aud)
            _, preds = torch.max(outputs, 1)

            if i < tot_batches-1:
                np_preds[ batch_size * i : batch_size * (i+1)]  = preds
            else:
                np_preds[ batch_size * i :  ]  = preds

            metrics_dict = audMetrics(labels, preds)
            
            runn_acc    += metrics_dict["accuracy"]
            runn_prec   += metrics_dict["precision"]
            runn_rec    += metrics_dict["recall"]
            runn_f1     += metrics_dict["f1"]

    metrics_dict["accuracy"] = runn_acc/lenDataSet
    metrics_dict["precision"] = runn_prec/lenDataSet
    metrics_dict["recall"] = runn_rec/lenDataSet
    metrics_dict["f1"] = runn_f1/lenDataSet

    print(f"Test model : {model.__class__.__name__} accuracy : {runn_acc/lenDataSet} f1 : {runn_f1/lenDataSet} ")

    return metrics_dict

def learn(model, loss, opt, train_x, train_y, device, epochs, batch_size):
    
    model.train()
    model = model.double()

    train_x             = toTensor(train_x)
    train_y             = torch.from_numpy(train_y.astype(float))
    train_size 			= train_x.shape[0]
    total_iter 			= int(np.ceil(train_size / batch_size))


    for epoch in range(epochs):
        for iter_ in range(total_iter):

            if batch_size * (iter_+1) < train_size :  
                train_batch  = train_x[ iter_ * batch_size : batch_size * (iter_ + 1) , :]
                target_batch = train_y[ iter_ * batch_size : batch_size * (iter_ + 1) ]
            else:
                train_batch  = train_x[ iter_ * batch_size : , :]
                target_batch = train_y[ iter_ * batch_size : ]
                        
            out = model(train_batch)
            loss_val = loss( out, target_batch.long())
            loss_val.backward()
            opt.step()
            opt.zero_grad()
            
            print(f"Train model : {model.__class__.__name__} Epoch {epoch}/{epochs} iter  {iter_}/{total_iter}  training loss {loss_val.item()}")

        # shuffling each epoch
        index = np.arange(0, train_size)
        np.random.shuffle(index)
        train_x, train_y = train_x[index], train_y[index]
        
    return model
