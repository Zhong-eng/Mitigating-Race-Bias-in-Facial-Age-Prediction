import numpy as np
import matplotlib.pyplot as plt
import os

import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
from sklearn.model_selection import train_test_split
import time

class ImageDataset(Dataset):
    def __init__(self, img, age):
        self.img = img
        self.age = age
    def __len__(self):
        return len(self.age)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #   idx = idx.tolist()
        # print(self.img)
        # print(idx)
        img = torch.tensor(self.img[idx], dtype=torch.float32).permute(2, 0, 1)
        age = torch.tensor(self.age[idx])

        return img, age

def get_performance(net, data_loader, device):
    """
    Evaluate model performance on validation set or test set.
    Input:
        - net: model
        - data_loader: data to evaluate, i.e. val or test
        - device: device to use
    Return:
        - loss: loss on validation set
    """
    net.eval()
    total_loss = [] # loss for each batch

    with torch.no_grad():
        for img, age in data_loader:
            loss = None # loss for this batch
            pred = None # predictions for this battch
            ######## TODO: calculate loss, get predictions #########
            pred = net(img.to('cuda')).reshape(-1)
            gt = age.to(torch.float32).to('cuda')
            loss = F.mse_loss(pred.to(device), gt.to(device))
            ###################### End of your code ######################
            total_loss.append(loss.item())
    total_loss = sum(total_loss) / len(total_loss)
    
    
    return total_loss

def get_performance_test(net, data_loader, device):
    """
    Evaluate model performance on validation set or test set.
    Input:
        - net: model
        - data_loader: data to evaluate, i.e. val or test
        - device: device to use
    Return:
        - loss: loss on validation set
    """
    net.eval()
    total_loss = [[] for i in range(8)] # loss for each batch

    with torch.no_grad():
        for img, age in data_loader:
            loss = None # loss for this batch
            pred = None # predictions for this battch
            ######## TODO: calculate loss, get predictions #########
            pred = net(img.to(device)).reshape(-1)
            gt = age.to(torch.float32).to(device)
            loss = F.mse_loss(pred.to(device), gt.to(device))
            ###################### End of your code ######################
            idx = age[0].item() // 10
            if idx >= 7:
                idx = 7
            total_loss[idx].append(loss.item())
    total_loss = [np.mean(each) for each in total_loss]
    
    return np.mean(total_loss)


def train(model, epoch, train_loader, val_loader):
    
    train_loss = []
    best_model, best_loss = None, 100000000

    print('------------------------ Start Training ------------------------')
    t_start = time.time()
    num_itr = 0
    for epoch in range(epoch):
        # Training:
        num_itr += 1
        model.train()
        optimizer = optim.Adam(params=model.parameters(), lr=1e-4, weight_decay=0.1)
        for img, age in train_loader:
            loss = None

            ############ TODO: calculate loss, update weights ############            
            optimizer.zero_grad()
            logits = model(img.to('cuda')).reshape(-1)
            labels = age.to(torch.float32).to('cuda')
            loss = F.mse_loss(logits.to('cuda'), labels)
            
            loss.backward()
            optimizer.step()
            ###################### End of your code ######################
            if num_itr % 30 == 0:  # Data collection cycle
                train_loss.append(loss.item())
        print('Epoch No. {0}--Iteration No. {1}-- batch loss = {2:.4f}'.format(
            epoch + 1,
            num_itr,
            loss.item()
        ))

        # Validation:
        val_loss = get_performance(model, val_loader, device)
        print("Validation loss: {:.4f}".format(val_loss))
        if val_loss < best_loss:
          best_model = model

    t_end = time.time()

    return best_model

def get_performance_group(net_list, data_loader, device):
    """
    Evaluate model performance on validation set or test set.
    Input:
        - net: model
        - data_loader: data to evaluate, i.e. val or test
        - device: device to use
    Return:
        - loss: loss on validation set
    """
    # net.eval()
    total_loss = [] # loss for each batch

    with torch.no_grad():
        for img, age in data_loader:
            loss = None # loss for this batch
            pred = None # predictions for this battch
            ######## TODO: calculate loss, get predictions #########
            pred = [net(img.to('cuda')).reshape(-1) for net in net_list]
            pred = torch.stack(pred)
            pred = torch.mean(pred, dim=0)
            gt = age.to(torch.float32).to('cuda')
            loss = F.mse_loss(pred.to(device), gt.to(device))
            ###################### End of your code ######################
            total_loss.append(loss.item())
    total_loss = sum(total_loss) / len(total_loss)
    return total_loss

def get_performance_group_test(net_list, data_loader, device):
    """
    Evaluate model performance on validation set or test set.
    Input:
        - net: model
        - data_loader: data to evaluate, i.e. val or test
        - device: device to use
    Return:
        - loss: loss on validation set
    """
    # net.eval()
    total_loss = [[] for i in range(8)] # loss for each batch

    with torch.no_grad():
        for img, age in data_loader:
            loss = None # loss for this batch
            pred = None # predictions for this battch
            ######## TODO: calculate loss, get predictions #########
            pred = [net(img.to('cuda')).reshape(-1) for net in net_list]
            pred = torch.stack(pred)
            pred = torch.mean(pred, dim=0)
            gt = age.to(torch.float32).to('cuda')
            loss = F.mse_loss(pred.to(device), gt.to(device))
            ###################### End of your code ######################
            idx = age[0].item() // 10
            if idx >= 7:
                idx = 7
            total_loss[idx].append(loss.item())
    total_loss = [np.mean(each) for each in total_loss]
    
    return np.mean(total_loss)





if __name__ == "__main__":
    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using the GPU!")
    else:
        print("WARNING: Could not find GPU! Using CPU only. If you want to enable GPU, please to go Edit > Notebook Settings > Hardware Accelerator and select GPU.")

    PATH = 'UTKFace'
    filenames = set()
    for filename in os.listdir(PATH):
        filenames.add(filename)
    filenames.remove('61_1_20170109150557335.jpg.chip.jpg')
    filenames.remove('61_1_20170109142408075.jpg.chip.jpg')
    filenames.remove('39_1_20170116174525125.jpg.chip.jpg')
    print(f"There are {len(filenames)} images in total")


    images = []
    ages = []
    races = []
    genders = []
    filenames = sorted(filenames)
    for filename in filenames:
        input_path = os.path.join(PATH, filename)
        image = cv2.imread(input_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        label = filename.split("_")
        ages.append(int(label[0]))
        genders.append(int(label[1]))
        races.append(int(label[2]))
        images.append(image)
    
    images = np.array(images)
    ages = np.array(ages)
    races = np.array(races)
    genders = np.array(genders)

    train_img, test_img, train_age, test_age, train_races, test_race, train_gender, test_gender = train_test_split(images, ages, races, genders, test_size=0.2, random_state=5) 
    train_img, val_img, train_age, val_age, train_races, val_race, train_gender, val_gender = train_test_split(train_img, train_age, train_races, train_gender, test_size=0.125, random_state=5) 
    

    
    train_dataset = ImageDataset(train_img, train_age)
    val_dataset = ImageDataset(val_img, val_age)

    train_loader = DataLoader(train_dataset, batch_size=32)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
    model.classifier[6] = nn.Linear(in_features=4096, out_features=1, bias=True)
    model = model.to('cuda')

    best_model = train(model, 50, train_loader, val_loader)
    torch.save(best_model, 'checkpoint/regression.pt')

    
    loss = []
    for i in range(5):
        idx = test_race == i
        print(sum(idx))
        img = test_img[idx]
        label = test_age[idx]
        dataset = ImageDataset(img, label)
        data_loader = DataLoader(dataset, batch_size=1)

        loss.append(get_performance_test(model, data_loader, device))
    print(loss)
    print(np.var(loss/np.sum(loss)))

    # ensemble Learning1
    model_list = []
    for i in range(5):
        idx = np.random.choice(train_img.shape[0], train_img.shape[0], replace=True)
        sample_img = train_img[idx]
        sample_age = train_age[idx]

        sample_set = ImageDataset(sample_img, sample_age)
        sample_loader = DataLoader(sample_set, batch_size=32)
        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
        model.classifier[6] = nn.Linear(in_features=4096, out_features=1, bias=True)
        model = model.to('cuda')
        best_model = train(model, 30, sample_loader, val_loader)
        model_list.append(best_model)
        torch.save(best_model, 'checkpoint/regression'+str(i)+'.pt')

    
    loss = []
    for i in range(5):
        idx = test_race == i
        print(sum(idx))
        img = test_img[idx]
        label = test_age[idx]
        dataset = ImageDataset(img, label)

        data_loader = DataLoader(dataset, batch_size=1)

        loss.append(get_performance_group_test(model_list, data_loader, 'cuda'))
    
    print(loss)
    print(np.var(loss/np.sum(loss)))


    