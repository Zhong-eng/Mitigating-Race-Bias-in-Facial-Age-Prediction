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
    def __init__(self, img, age, race):
        self.img = img
        self.age = age
        self.race = race
    def __len__(self):
        return len(self.age)

    def __getitem__(self, idx):
        img = torch.tensor(self.img[idx], dtype=torch.float32).permute(2, 0, 1)
        age = torch.tensor(self.age[idx])
        race = torch.tensor(self.race[idx])
        return img, age, race
    
class RaceIndifferenceModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.shared = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
        self.Relu = nn.ReLU()
        self.age_pred = nn.Linear(in_features=1000, out_features=1, bias=True)
        self.race_class = nn.Linear(in_features=1000, out_features=5, bias=True)
        
    def forward(self, x):

        x_1 = self.Relu(self.shared(x))
        x_2 = x_1
        x_1 = self.age_pred(x_1)
        x_2 = self.race_class(x_2)

        return x_1, x_2

def train_indiff(model, epoch, train_loader, val_loader):
    
    train_loss = []
    best_model, best_loss = None, 100000000

    print('------------------------ Start Training ------------------------')
    t_start = time.time()
    num_itr = 0
    for epoch in range(epoch):
        # Training:
        num_itr += 1
        model.train()
        optimizer = optim.Adam(params=model.parameters(), lr=1e-4, weight_decay=1e-5)
        for img, age, race in train_loader:
            loss = None

            ############ TODO: calculate loss, update weights ############            
            optimizer.zero_grad()

            output = model(img.to('cuda'))
            output_1, output_2 = output
            output_1 = output_1.reshape(-1)
            output_2 = F.softmax(output_2, dim=1)

            label_1 = age.to(torch.float32).to('cuda')
            criteron = nn.CrossEntropyLoss()
            label_2 = race.to('cuda')
            loss = F.mse_loss(output_1.to('cuda'), label_1)*0.05-criteron(output_2.to('cuda'), label_2)
            
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
        val_loss, race_loss = get_performance_train_indiff(model, val_loader, device)
        print("Validation age loss: {0:.4f}, race_loss: {1:.4f}".format(val_loss, race_loss))
        if val_loss < best_loss:
          best_model = model

    t_end = time.time()

    return best_model

def get_performance_train_indiff(net, data_loader, device):
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
    race_loss = []
    with torch.no_grad():
        for img, age, race in data_loader:
            loss = None # loss for this batch
            pred = None # predictions for this battch
            ######## TODO: calculate loss, get predictions #########
            pred, fake = net(img.to('cuda'))
            pred = pred.reshape(-1)
            gt = age.to(torch.float32).to('cuda')
            loss = F.mse_loss(pred.to(device), gt.to(device))
            ###################### End of your code ######################
            total_loss.append(loss.item())
            criteron = nn.CrossEntropyLoss()
            fake = F.softmax(fake, dim=1)
            race_loss.append(criteron(fake.to(device), race.to(device)))
    total_loss = sum(total_loss) / len(total_loss)
    race_loss = sum(race_loss) / len(race_loss)
    return total_loss, race_loss


def get_performance_indiff_test(net, data_loader, device):
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
        for img, age, _ in data_loader:
            loss = None # loss for this batch
            pred = None # predictions for this battch
            ######## TODO: calculate loss, get predictions #########
            pred, _ = net(img.to(device))
            pred = pred.reshape(-1)
            gt = age.to(torch.float32).to(device)
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
        # if len(filenames) > 100:
        #     break
    
    ### Deleted manually
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
    
    ## TODO:Fake Train_race
    race_number = train_img.shape[0] // 5
    fake_train_races = [0] * race_number + [1] * race_number + [2] * race_number + [3] * race_number + [4] * (train_img.shape[0] - 4 * race_number)

    np.random.shuffle(fake_train_races)
    train_dataset = ImageDataset(train_img, train_age, train_races)
    val_dataset = ImageDataset(val_img, val_age, val_race)

    train_loader = DataLoader(train_dataset, batch_size=32)
    val_loader = DataLoader(val_dataset, batch_size=32)


    ####### RaceIndifferenceModel Model#######
    model_ = RaceIndifferenceModel().to(device)

    best_mode_ = train_indiff(model_, 80, train_loader, val_loader)
    torch.save(best_mode_, 'checkpoint/RaceIndifferenceModel.pt')

    loss = []
    for i in range(5):
        idx = test_race == i
        print(sum(idx))
        img = test_img[idx]
        label = test_age[idx]
        dataset = ImageDataset(img, label, test_race[idx])
        data_loader = DataLoader(dataset, batch_size=1)

        loss.append(get_performance_indiff_test(best_mode_, data_loader, 'cuda'))
    print(loss)
    print(np.var(loss/np.sum(loss)))
    