#!/usr/bin/env python
# coding: utf-8

# In[60]:


import torch
from  torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import os
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import itertools
writer = SummaryWriter()


# In[61]:


class stl10_dataset(torch.utils.data.Dataset):

    def __init__(self, text_file, root_dir, transform):
        """
        Args:
            text_file(string): path to text file
            root_dir(string): directory with all train images
        """
        self.name_frame = pd.read_csv(text_file,sep=" ",usecols=range(1))         
        self.label_frame = pd.read_csv(text_file,sep=" ",usecols=range(1,2))       
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.name_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.name_frame.iloc[idx, 0])
        image = Image.open(img_name)
        image = self.transform(image)
        labels = self.label_frame.iloc[idx, 0]

        return image,labels

data_transform = transforms.Compose([ transforms.Resize((32,32)), transforms.ToTensor(), 
                                     transforms.Normalize(mean=[0,0,0], std=[1,1,1])])

trainSet = stl10_dataset(text_file ='splits/train.txt', root_dir = '', transform=data_transform)
stl10_TrainLoader = torch.utils.data.DataLoader(trainSet, batch_size=16, shuffle=True)


# In[62]:


# Find mean and standard deviation for the data.

def compute_mean_std(image_set, image_loader):

    psum    = torch.tensor([0.0, 0.0, 0.0])             
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    # loop through images and find mean 
    for inputs,_ in tqdm(image_loader):
        psum    += inputs.sum(axis = [0, 2, 3])     
        psum_sq += (inputs ** 2).sum(axis = [0, 2, 3])

    # pixel count in a batch
    count = len(image_set) * 32 * 32

    # mean and std dev calculations
    total_mean = psum / count
    total_var  = (psum_sq / count) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)

    # print data stats 
    print('mean: '  + str(total_mean))
    print('std:  '  + str(total_std))
    return total_mean, total_std
   
# compute mean and standard deviation
train_mean, train_std = compute_mean_std(trainSet, stl10_TrainLoader)


# In[63]:


#re-load the train test and val data with newly calculated mean and std dev

data_transform2 = transforms.Compose([ transforms.Resize((32,32)), transforms.ToTensor(), 
                                     transforms.Normalize(mean=train_mean, std=train_std)])


trainSet = stl10_dataset(text_file ='splits/train.txt', root_dir = '', transform=data_transform2)
testSet = stl10_dataset(text_file ='splits/test.txt', root_dir = '', transform=data_transform2)
valSet = stl10_dataset(text_file ='splits/val.txt', root_dir = '', transform=data_transform2)

stl10_TrainLoader = torch.utils.data.DataLoader(trainSet, batch_size=128, shuffle=True)
stl10_TestLoader = torch.utils.data.DataLoader(testSet, batch_size=1, shuffle=True)
stl10_ValLoader = torch.utils.data.DataLoader(valSet, batch_size=1, shuffle=True)

   


# In[64]:


#plotting sample image
dataiter = iter(stl10_TrainLoader)
images, labels = dataiter.next()
plt.imshow(np.transpose(images[0].numpy(), (1, 2, 0)))
print(labels)


# In[65]:


#lenet5 with batch normalization 

class LeNet5_bn(nn.Module):
    def __init__(self):
        super(LeNet5_bn, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, stride = 1, kernel_size=(5,5))
        self.conv1_bn = nn.BatchNorm2d(6)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, stride = 1, kernel_size=(5,5))
        self.conv2_bn = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(400, 120)
        self.fc1_bn = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.fc2_bn = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84,10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv1_bn(x)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.conv2_bn(x)
        x = self.pool2(x)
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = self.fc1_bn(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_bn(x)
        x = self.fc3(x)
        return x                   


# In[66]:


#lenet5 
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, stride = 1, kernel_size=(5,5))
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, stride = 1, kernel_size=(5,5))
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(400, 120)             
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,10)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# In[67]:


# multi-class accuracy function 

def multi_accuracy(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc * 100)
    return acc


# In[68]:


def train(model, epochs, criterion, optimizer, scheduler,model_name=""):
    model.train()
    min_valid_loss = float("inf")
    
    for epoch in range(1,epochs+1):  # loop over the dataset multiple times
        running_loss = 0.0
        running_accuracy = 0
        
        for i, data in enumerate(stl10_TrainLoader, 0):
            batch_loss = 0
            batch_accuracy = 0
            
            inputs, labels = data    # get the inputs; data is a list of [inputs, labels]
            
            optimizer.zero_grad()    # zero the parameter gradients

            
            outputs = model(inputs)  # forward + backward + optimize
            loss = criterion(outputs, labels) # calculate loss
            loss.backward()                   # accumulate gradient
            optimizer.step()                  # update weights

            # add statistics calculated
            running_loss += loss.item()
            running_accuracy += multi_accuracy(outputs,labels)
            batch_loss += loss.item()
            batch_accuracy += multi_accuracy(outputs,labels)
            
            # log statistics on tensorboard
            writer.add_scalars(model_name+' batch_step_loss', {'training loss':(batch_loss/len(data))}, epoch)
            writer.add_scalars(model_name+' batch_step_accuracy', {'training accuracy':(batch_accuracy/len(data))}, epoch)
            
        
        # advance the scheduler for lr decay
        scheduler.step()
        
        
        # calculate loss and accuracy on validation set
        r_valid_loss = 0.0
        r_valid_accuracy = 0
        model.eval()
        for data, labels in stl10_ValLoader:
            target = model(data)
            loss = criterion(target,labels)
            r_valid_loss += loss.item() 
            r_valid_accuracy += multi_accuracy(target, labels) 
            
        
        #print Loss and Accuracy every 5 epochs    
        if epoch%5 ==0:   
            print("Epoch", epoch," Training Loss", running_loss/len(stl10_TrainLoader)," Training Accuracy", running_accuracy/len(stl10_TrainLoader))
            print("Epoch", epoch," Validation Loss", r_valid_loss/len(stl10_ValLoader)," Validation Accuracy", r_valid_accuracy/len(stl10_ValLoader))
       
        # log training vs validation loss and accuracy 
        writer.add_scalars(model_name+' loss', {'training loss':(running_loss/len(stl10_TrainLoader)),'validation loss':(r_valid_loss/len(stl10_ValLoader))}, epoch)
        writer.add_scalars(model_name+' accuracy', {'training accuracy':(running_accuracy/len(stl10_TrainLoader)),'validation accuracy':(r_valid_accuracy/len(stl10_ValLoader))}, epoch)
        
    print('Finished Training')
    return model


# In[69]:


# print class wise accuracy 

def print_class_wise_acc(cm, model_name):
    print("Accuracy by each class for {}".format(model_name))
    accuracy_multiclass = []
    index = 0
    for row in cm:
        accuracy_multiclass.append([index+1, (row[index]/sum(row))*100])
        index += 1
    df = pd.DataFrame(accuracy_multiclass, columns=['class', 'accuracy'])
    print(df)


# In[70]:


def evaluate_model(model,model_name=""):
    model.eval()

    y_pred_list = []
    y_test_list = []
    correct_pred = 0
    failed_dict = {}

    with torch.no_grad():
        for X_batch, y_batch in stl10_TestLoader:            # loop through the test batch
            X_batch = X_batch.type(torch.FloatTensor)        # convert to tensor
            y_batch.type(torch.FloatTensor)
            y_test_pred = model(X_batch)                     # get prediction
            
            y_pred_softmax = torch.log_softmax(y_test_pred, dim = 1)
            _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    # get class tag
            
            y_pred_list.append(y_pred_tags.item())           
            
            y_test_list.append(y_batch.item())
            if y_batch != y_pred_tags:                       # save one example of each wrongly predicted class 
                if y_batch.item() not in failed_dict:
                    failed_dict[y_batch.item()] = (X_batch, y_pred_tags)
            
            correct_pred += multi_accuracy(y_test_pred, y_batch)
        print(correct_pred.item()/len(stl10_TestLoader))
        
        
        y_test_list, y_pred_list = np.array(y_test_list), np.array(y_pred_list) 
        y_unique = np.array([1,2,3,4,5,6,7,8,9,10])          # unique label names for confusion matrix plot
        cm = confusion_matrix(y_test_list, y_pred_list)      # sklearn's confusion matrix
        
        plot_confusion_matrix(cm, y_unique,title=model_name) # plot confusion matrix
        print_class_wise_acc(cm, model_name)                 # print classwise accuracy 
        
        # plot misclassified images 
        
        i=0
        fig, axs = plt.subplots(2, 5, figsize=(25, 10))
        failed_dict = dict(sorted(failed_dict.items()))
        for label, (image, pred) in failed_dict.items():
            image = torch.squeeze(image, axis=0)            
            
            if i<5:  
                axs[0, i].imshow(np.transpose(image.numpy(), (1, 2, 0)))
                axs[0, i].set_title("label:"+str(label+1)+" pred:"+str(pred.item()+1))
            else:
                axs[1, i-5].imshow(np.transpose(image.numpy(), (1, 2, 0)))
                axs[1, i-5].set_title("label:"+str(label+1)+" pred:"+str(pred.item()+1))
                
            i+=1

        


# In[71]:


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Purples')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


# # lenet5 main experiment

# In[72]:


lenet = LeNet5()  
EPOCHS = 100
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lenet.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
lenet = train(lenet, EPOCHS, criterion, optimizer, scheduler,model_name="lenet5")
writer.flush()


# In[73]:


evaluate_model(lenet, model_name="lenet5")


# # lenet5 with l2 regularization

# In[74]:


lenet_l2 = LeNet5()  
EPOCHS = 100
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lenet_l2.parameters(), lr=0.001, weight_decay=0.01)
scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
lenet_l2 = train(lenet_l2, EPOCHS, criterion, optimizer, scheduler, model_name ="lenet5_l2_regularization")
writer.flush()


# In[75]:


evaluate_model(lenet_l2, model_name ="lenet5_l2_regularization")


# # lenet5 with batch normalization

# In[76]:


lenet_bn = LeNet5_bn()
EPOCHS = 100
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lenet_bn.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
lenet = train(lenet_bn, EPOCHS, criterion, optimizer, scheduler, model_name="lenet5_batch_norm")
writer.flush()


# In[77]:


evaluate_model(lenet_bn, model_name="lenet5_batch_norm")


# # Accuracies for each model 
# 
# Lenet5 Model:
# 48.32966593318664
# 
# Lenet5 l2 Regularization Model:
# 48.60972194438888
# 
# Lenet5 Batch Normalization Model:
# 47.289457891578316

# In[ ]:




