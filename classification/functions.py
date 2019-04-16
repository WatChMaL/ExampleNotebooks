from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch

def save_state(blob, prefix='./snapshot'):
    # Output file name
    filename = '%s-%d.ckpt' % (prefix, blob.iteration)
    # Save parameters
    # 0+1) iteration counter + optimizer state => in case we want to "continue training" later
    # 2) network weight
    torch.save({
        'global_step': blob.iteration,
        'optimizer': blob.optimizer.state_dict(),
        'state_dict': blob.net.state_dict()
    }, filename)
    return filename

def restore_state(weight_file, blob):
    # Open a file in read-binary mode
    with open(weight_file, 'rb') as f:
        # torch interprets the file, then we can access using string keys
        checkpoint = torch.load(f)
        # load network weights
        blob.net.load_state_dict(checkpoint['state_dict'], strict=False)
        # if optimizer is provided, load the state of the optimizer
        if blob.optimizer is not None and 'optimizer' in checkpoint.keys():
            blob.optimizer.load_state_dict(checkpoint['optimizer'])
        # load iteration count
        if 'global_step' in checkpoint.keys():
            blob.iteration = checkpoint['global_step']

def forward(blob,train=True):
    """
       Args: blob should have attributes, net, criterion, softmax, data, label
       Returns: a dictionary of predicted labels, softmax, loss, and accuracy
    """
    with torch.set_grad_enabled(train):
        # Prediction
        data = torch.as_tensor(blob.data).cuda()#[torch.as_tensor(d).cuda() for d in blob.data]
        data = data.permute(0,3,1,2)
        prediction = blob.net(data)
        # Training
        loss,acc=-1,-1
        if blob.label is not None:
            label = torch.as_tensor(blob.label).type(torch.LongTensor).cuda()#[torch.as_tensor(l).cuda() for l in blob.label]
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

def train_loop(blob,train_epoch=2.,store_iterations=500,store_prefix='snapshot'):
    import time
    # Define train period. "epoch" = N image consumption where N is the total number of train samples.
    TRAIN_EPOCH=float(train_epoch)
    # Set the network to training mode
    blob.net.train()
    # Check if logger exists
    train_logger = None if not hasattr(blob,'train_log') else blob.train_log
    test_logger  = None if not hasattr(blob,'test_log')  else blob.test_log
    train_loader = blob.train_loader
    test_loader  = None if not hasattr(blob,'test_loader') else blob.test_loader
    # Make sure snapshot directory exists, or attempt to create
    if '/' in store_prefix:
        dirname = store_prefix[store_prefix.rfind('/'):]
        if not os.path.isdir(dirname): os.makedirs(dirname)
    # Set epoch/iteration counter if necessary
    epoch=0.
    if not hasattr(blob,'iteration'):
        blob.iteration = 0
    # Start training
    while int(epoch+0.5) < TRAIN_EPOCH:
        print('Epoch',int(epoch+0.5),'Starting @',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        # Create a progress bar for this epoch
        from utils import progress_bar
        progress = display(progress_bar(0,len(train_loader)),display_id=True)
        # Loop over data samples and into the network forward function
        for i,data in enumerate(train_loader):
            # Data and label
            blob.data,blob.label = data[0:2]
            # Call forward: make a prediction & measure the average error
            res = forward(blob,True)
            # Call backward: backpropagate error and update weights
            backward(blob)
            # Epoch update
            epoch += 1./len(train_loader)
            blob.iteration += 1
            
            #
            # Log/Report
            #
            # Record the current performance on train set
            if train_logger is not None:
                blob.train_log.record(['blob.iteration','epoch','accuracy','loss'],[blob.iteration,epoch,res['accuracy'],res['loss']])
                blob.train_log.write()
            # once in a while, report
            if blob.iteration==1 or blob.iteration%10 == 0:
                message = '... Iteration %d ... Epoch %1.2f ... Loss %1.3f ... Accuracy %1.3f' % (blob.iteration,epoch,res['loss'],res['accuracy'])
                progress.update(progress_bar((i+1),len(train_loader),message))
            # more rarely, run validation
            if test_loader is not None and blob.iteration%100 == 0:
                with torch.no_grad():
                    blob.net.eval()
                    test_data = next(iter(test_loader))
                    blob.data,blob.label = test_data[0:2]
                    res = forward(blob,False)
                    if test_logger is not None:
                        blob.test_log.record(['iteration','epoch','accuracy','loss'],[blob.iteration,epoch,res['accuracy'],res['loss']])
                        blob.test_log.write()
                blob.net.train()
            # store weights
            if blob.iteration%store_iterations == 0:
                fname = store_prefix + '-%07d' % blob.iteration
                save_state(blob,fname)
            if epoch >= TRAIN_EPOCH:
                break
        message = '... Iteration %d ... Epoch %1.2f ... Loss %1.3f ... Accuracy %1.3f' % (blob.iteration,epoch,res['loss'],res['accuracy'])
        progress.update(progress_bar(len(train_loader),len(train_loader),message))
        
    if test_logger is not None: test_logger.close()
    if train_logger is not None: train_logger.close()

def plot_log(train_log_name,test_log_name=None):
    """
    Args: train_log ... string value pointing to the train log csv file
          test_log .... string value pointing to the test log csv file (optional)
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    train_log = pd.read_csv(train_log_name)
    test_log  = None if test_log_name is None else pd.read_csv(test_log_name)

    def moving_average(a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    epoch_average = moving_average(train_log.epoch.values,40)
    loss_average  = moving_average(train_log.loss.values,40)
    accuracy_average = moving_average(train_log.accuracy.values,40)
    
    fig, ax1 = plt.subplots(figsize=(12,8),facecolor='w')
    lines = []
    lines += ax1.plot(train_log.epoch, train_log.loss, linewidth=2, label='Train loss', color='b', alpha=0.3)
    ax1.plot(epoch_average, loss_average, color='blue')
    if test_log is not None:
        lines += ax1.plot(test_log.epoch, test_log.loss, marker='o', markersize=12, linestyle='', label='Test loss', color='blue')
    ax1.set_xlabel('Epoch',fontweight='bold',fontsize=24,color='black')
    ax1.tick_params('x',colors='black',labelsize=18)
    ax1.set_ylabel('Loss', fontsize=24, fontweight='bold',color='b')
    ax1.tick_params('y',colors='b',labelsize=18)
    
    ax2 = ax1.twinx()
    lines += ax2.plot(train_log.epoch, train_log.accuracy, linewidth=2, label='Train accuracy', color='r', alpha=0.3)
    ax2.plot(epoch_average, accuracy_average, color='red')
    if test_log is not None:
        lines += ax2.plot(test_log.epoch, test_log.accuracy, marker='o', markersize=12, linestyle='', label='Test accuracy', color='red')
    
    ax2.set_ylabel('Accuracy', fontsize=24, fontweight='bold',color='r')
    ax2.tick_params('y',colors='r',labelsize=18)
    ax2.set_ylim(0.,1.0)
    
    # added these four lines
    labels = [l.get_label() for l in lines]
    leg    = ax1.legend(lines, labels, fontsize=16, loc=5)
    leg_frame = leg.get_frame()
    leg_frame.set_facecolor('white')
    
    plt.grid()
    plt.show()

def inference(blob,data_loader):
    """
    Args: blob ... needs to have "net" attribute
          data_loader ... DataLoader instance for inference
    Return: accuracy (per-batch), label (per sample), and prediction (per-sample)
    """
    label,prediction,accuracy=[],[],[]
    # set the network to test (non-train) mode
    blob.net.eval()
    # create the result holder
    index,label,prediction = [],[],[]
    for data in data_loader:
        blob.data, blob.label = data[0:2]
        res = forward(blob,True)
        accuracy.append(res['accuracy'])
        prediction.append(res['prediction'])
        label.append(blob.label)
    accuracy   = np.array(accuracy,dtype=np.float32)
    label      = np.hstack(label)
    prediction = np.hstack(prediction)        
    return accuracy, label, prediction

