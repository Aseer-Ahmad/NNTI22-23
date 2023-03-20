#train.py
from sklearn.model_selection import train_test_split
import torch.nn as nn
import tqdm
import torch
from metrics import audMetrics
from CustomAudioDataset import CustomAudioDataset
from torch.utils.data import DataLoader
import os
from preprocessor import transformMelSpecByTruncate1D

def train(model, loss, optimizer, scheduler, device, epochs, val = True):

    model = model.to(device)

    PARENT_PTH = os.getcwd()
    TRAIN_PTH = os.path.join(PARENT_PTH, 'data', 'train')
    VAL_PTH   = os.path.join(PARENT_PTH, 'data', 'dev')

    trainDataSet = CustomAudioDataset(TRAIN_PTH, 8000, transformMelSpecByTruncate1D)
    valDataSet   = CustomAudioDataset(VAL_PTH, 8000, transformMelSpecByTruncate1D)
    
    trainLoader = DataLoader(trainDataSet, batch  = 32, shuffle = True)
    valLoader   = DataLoader(valDataSet, batch  = len(valDataSet), shuffle = True)

    for epoch in range(epochs):

        model.train()

        running_loss = 0

        for i, (aud, labels) in enumerate(tqdm(trainLoader, position=0)):
    
            aud = aud.to(device)
            labels = labels.to(device)
        
            outputs = model(aud)
            _, preds = torch.max(outputs, 1)
            
            loss_val = loss(outputs, labels)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if scheduler is not None:
                scheduler.step()
                
            running_loss += loss_val.item()

            metrics_dict = audMetrics(outputs, labels, preds)
            
        print(f'Epoch {epoch}/{epochs} , Step {i}/{len(trainLoader)} train loss : {running_loss / len(trainLoader)}')

        if val:
            model.eval()
            with torch.no_grad():
                for aud, labels in valLoader:
                    out  = model(aud)
                    loss_val = loss(out, labels)
                    print(f"val loss : {loss_val.item()}")

    return model
