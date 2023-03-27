from unittest import TestLoader
from sklearn import metrics
from sklearn.model_selection import train_test_split
import torch.nn as nn
import numpy as np
import torch
from helpers.metrics import audMetrics
from helpers.CustomAudioDataset import CustomAudioDataset
from torch.utils.data import DataLoader
import os
import torchvision

from torch.utils.tensorboard import SummaryWriter


def train(model, loss, optimizer, scheduler, device, epochs, transform, sr, batch_size, val = True):

    model = model.to(device)
    model = model.double()    


    PARENT_PTH = os.getcwd()
    TEST_PTH = os.path.join(PARENT_PTH, 'data', 'test')
    TRAIN_PTH = os.path.join(PARENT_PTH, 'data', 'train')
    VAL_PTH   = os.path.join(PARENT_PTH, 'data', 'dev')

    trainDataSet = CustomAudioDataset(TRAIN_PTH, sr, transform)
    trainLoader = DataLoader(trainDataSet, batch_size  = batch_size, shuffle = True)

    if val:
        valDataSet  = CustomAudioDataset(VAL_PTH, sr, transform)
        valLoader   = DataLoader(valDataSet, batch_size  = batch_size, shuffle = True)

    # tensorboard
    iteration_num = 1
    tb = SummaryWriter()
    aud, labels = next(iter(trainLoader))
    aud = aud.to(device)
    tb.add_graph(model, aud)

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
            
            if scheduler is not None and epoch%3==0:
                scheduler.step()
                
            running_loss += loss_val.item()

            metrics_dict = audMetrics(labels, preds)
            
            print(f'Epoch {epoch}/{epochs} , Step {i}/{len(trainLoader)} train loss : {running_loss / i} ')
            print(f'accuracy : { metrics_dict["accuracy"] } precision : { metrics_dict["precision"] } recall : { metrics_dict["recall"] } f1 : { metrics_dict["f1"] }')


            running_loss_val = 0
            runn_acc_val     = 0
            runn_f1_val    = 0

            # if val:
            #     model.eval()
            #     with torch.no_grad():
            #         for aud, labels in valLoader:
            #             aud = aud.to(device)
            #             labels = labels.to(device)
            #             out  = model(aud)
            #             loss_val = loss(out, labels)
            #             _, preds = torch.max(out, 1)
            #             metrics_dict_val = audMetrics(labels, preds)
            #             running_loss_val += loss_val.item()
            #             runn_acc_val    += metrics_dict_val["accuracy"]
            #             runn_f1_val     += metrics_dict_val["f1"]

            #     print(f"val loss : {running_loss_val/len(valLoader)}")


        # metrics_dict_test, _, _, _ = test(model, TEST_PTH, loss, transform, device, sr )

            # print()
            
            # # scalars for tensorboard
            # tb.add_scalars(f'loss/check_info', {
            #     'train loss ': running_loss/i,
            #     'test loss': metrics_dict_test["test_loss"],
            #     'val loss': running_loss_val/len(valLoader)
            # }, iteration_num)

            # tb.add_scalars(f'accuracy/check_info', {
            #     'train accuracy ': metrics_dict["accuracy"],
            #     'test accuracy': metrics_dict_test["accuracy"],
            #     'val accuracy': runn_acc_val / len(valLoader)
            # }, iteration_num)

            # tb.add_scalars(f'f1/check_info', {
            #     'train accuracy ': metrics_dict["f1"],
            #     'test accuracy': metrics_dict_test["f1"],
            #     'val accuracy': runn_f1_val / len(valLoader)
            # }, iteration_num)

            # iteration_num += 1

            # model.train()

    tb.close()

    return model

def test(model, TEST_PTH, loss, transform,  device, sr):

    model = model.double()
    model.eval()

    BATCH_SIZE   = 32
    testDataSet  = CustomAudioDataset(TEST_PTH, sr, transform)
    testLoader  = DataLoader(testDataSet, batch_size  = BATCH_SIZE)
    tot_batches  = len(testLoader)
    lenDataSet   = len(testLoader)
    np_preds     = np.zeros((len(testDataSet)))
    runn_loss    = 0
    runn_acc     = 0
    runn_prec    = 0
    runn_rec     = 0
    runn_f1      = 0

    with torch.no_grad():
        for i, (aud, labels) in enumerate(testLoader, start=0):
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

        # print(f"Finished proecessing test batch {i+1}")

    print(f"test loss {runn_loss/lenDataSet} accuracy {runn_acc/lenDataSet} precision {runn_prec/lenDataSet} recall {runn_rec/lenDataSet} f1 {runn_f1/lenDataSet}")

    metrics_dict["test_loss"] = runn_loss/lenDataSet
    metrics_dict["accuracy"] = runn_acc/lenDataSet
    metrics_dict["precision"] = runn_prec/lenDataSet
    metrics_dict["recall"] = runn_rec/lenDataSet
    metrics_dict["f1"] = runn_f1/lenDataSet

    # separate test data and labels for t-sne ; return
    testDataSet_np = np.array(testDataSet)
    test_x, test_y = testDataSet_np[:, 0], testDataSet_np[:, 1]    
    
    return metrics_dict, test_x, test_y, np_preds

        



