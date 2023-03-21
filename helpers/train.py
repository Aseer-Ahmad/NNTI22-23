from sklearn import metrics
from sklearn.model_selection import train_test_split
import torch.nn as nn
import numpy as np
import torch
from helpers.metrics import audMetrics
from helpers.CustomAudioDataset import CustomAudioDataset
from torch.utils.data import DataLoader
import os

from helpers.tsne import plot2D_tsne

# add summaryWriters for train test val metrics

def train(model, loss, optimizer, scheduler, device, epochs, transform, sr, batch_size, val = True):

    model = model.to(device)
    model = model.double()

    PARENT_PTH = os.getcwd()
    TRAIN_PTH = os.path.join(PARENT_PTH, 'data', 'train')
    VAL_PTH   = os.path.join(PARENT_PTH, 'data', 'dev')

    trainDataSet = CustomAudioDataset(TRAIN_PTH, sr, transform)
    trainLoader = DataLoader(trainDataSet, batch_size  = batch_size, shuffle = True)

    if val:
        valDataSet  = CustomAudioDataset(VAL_PTH, sr, transform)
        valLoader   = DataLoader(valDataSet, batch_size  = len(valDataSet), shuffle = True)

    for epoch in range(1, epochs+1):

        model.train()

        running_loss = 0

        for i, (aud, labels) in enumerate(trainLoader, 1):
    
            aud = aud.to(device)
            labels = labels.to(device)
        
            outputs = model(aud)
            _, preds = torch.max(outputs, 1)
            
            loss_val = loss(outputs, labels)
        
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
                
            running_loss += loss_val.item()

            metrics_dict = audMetrics(labels, preds)
            
            print(f'Epoch {epoch}/{epochs} , Step {i}/{len(trainLoader)} train loss : {running_loss / i} ')
            print(f'accuracy : { metrics_dict["accuracy"] } precision : { metrics_dict["precision"] } recall : { metrics_dict["recall"] } f1 : { metrics_dict["f1"] }\n')

        
        if val:
            model.eval()
            with torch.no_grad():
                for aud, labels in valLoader:
                    aud = aud.to(device)
                    labels = labels.to(device)
                    out  = model(aud)
                    loss_val = loss(out, labels)
                    print(f"val loss : {loss_val.item()}")

    return model

def test(model, TEST_PTH, loss, transform,  device, sr):

    model = model.double()
    model.eval()

    BATCH_SIZE   = 32
    testDataSet  = CustomAudioDataset(TEST_PTH, sr, transform)
    trainLoader  = DataLoader(testDataSet, batch_size  = BATCH_SIZE)
    tot_batches  = len(trainLoader)
    lenDataSet   = len(trainLoader)
    np_preds     = np.zeros((len(testDataSet)))
    runn_loss    = 0
    runn_acc     = 0
    runn_prec    = 0
    runn_rec     = 0
    runn_f1      = 0

    # test data and labels separate for t-sne ; return
    testDataSet_np = np.array(testDataSet)
    test_x, test_y = testDataSet_np[:, 0], testDataSet_np[:, 1]    
            
    for i, (aud, labels) in enumerate(trainLoader, start=0):
        aud      = aud.to(device)
        labels   = labels.to(device)
        outputs  = model(aud)
        _, preds = torch.max(outputs, 1)

        loss_val  = loss(outputs, labels)
        runn_loss += loss_val
        
        if i < tot_batches-1:
            np_preds[ BATCH_SIZE * i : BATCH_SIZE * (i+1)]  = preds
        else:
            np_preds[ BATCH_SIZE * i :  ]  = preds

        metrics_dict = audMetrics(labels, preds)
        
        runn_acc    += metrics_dict["accuracy"]
        runn_prec   += metrics_dict["precision"]
        runn_rec    += metrics_dict["recall"]
        runn_f1     += metrics_dict["f1"]

        print(f"Finished proecessing test batch {i+1}")

    print(f"test loss {runn_loss/lenDataSet} accuracy {runn_acc/lenDataSet} precision {runn_prec/lenDataSet} recall {runn_rec/lenDataSet} f1 {runn_f1/lenDataSet}")

    metrics_dict["accuracy"] = runn_acc/lenDataSet
    metrics_dict["precision"] = runn_prec/lenDataSet
    metrics_dict["recall"] = runn_rec/lenDataSet
    metrics_dict["f1"] = runn_f1/lenDataSet

    return metrics_dict, test_x, test_y, np_preds

        



