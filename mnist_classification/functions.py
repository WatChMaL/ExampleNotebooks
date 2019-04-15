from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import numpy as np
import torch

def forward(blob,train=True):
    """
       Args: blob should have attributes, net, criterion, softmax, data, label
       
       Returns: a dictionary of predicted labels, softmax, loss, and accuracy
    """
    with torch.set_grad_enabled(train):
        # Prediction
        data = blob.data.cuda()
        prediction = blob.net(data)
        # Training
        loss,acc=-1,-1
        if blob.label is not None:
            label = blob.label.cuda() #label = torch.stack([ torch.as_tensor(l) for l in np.hstack(label) ])
            label.requires_grad = False
            loss = blob.criterion(prediction,label)
        blob.loss = loss
        
        softmax    = blob.softmax(prediction).cpu().detach().numpy()
        prediction = torch.argmax(prediction,dim=-1)
        accuracy   = (prediction == label).sum().item() / float(prediction.nelement())        
        prediction = prediction.cpu().detach().numpy()
        
        return {'prediction' : prediction,
                'softmax'    : softmax,
                'loss'       : loss.cpu().detach().item(),
                'accuracy'   : accuracy}

def backward(blob):
    blob.optimizer.zero_grad()  # Reset gradients accumulation
    blob.loss.backward()
    blob.optimizer.step()

def train_loop(blob,data_loader,max_iteration=2000):
    # Set the network to training mode
    blob.net.train()
    
    # Let's train 2000 steps
    stop_iteration = 2000
    # Loop over data samples and into the network forward function
    for i,data in enumerate(data_loader):
        blob.iteration = i
        # data and label
        blob.data, blob.label = data
        # call forward
        res = forward(blob,True)
        # once in a while, report
        if blob.iteration == 0 or (blob.iteration+1)%100 == 0:
            print('Iteration',blob.iteration,'... Loss',res['loss'],'... Accuracy',res['accuracy'])
        if (blob.iteration+1)==max_iteration:
            break
        backward(blob)

def inference(blob,data_loader):
    # set the network to test (non-train) mode
    blob.net.eval()
    # create the result holder
    accuracy, label, prediction = [], [], []
    confusion_matrix = np.zeros([10,10],dtype=np.int32)
    for i,data in enumerate(data_loader):
        blob.data, blob.label = data
        res = forward(blob,True)
        accuracy.append(res['accuracy'])
        prediction.append(res['prediction'])
        label.append(blob.label)
    # organize the return values
    accuracy   = np.hstack(accuracy)
    prediction = np.hstack(prediction)
    label      = np.hstack(label)
    return accuracy, label, prediction
