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
            loss_ = None # loss for this batch
            pred = None # predictions for this battch
            ######## TODO: calculate loss, get predictions #########
            pred = net(img.to(device)).reshape(-1)
            gt = age.to(torch.float32).to(device)
            loss_ = F.mse_loss(pred.to(device), gt.to(device))
            ###################### End of your code ######################
            idx = age[0].item() // 10
            if idx >= 7:
                idx = 7
            total_loss[idx].append(loss_.item())
    total_loss = [np.mean(each) for each in total_loss]
    
    return np.mean(total_loss)

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
            pred = net(img.to(device)).reshape(-1)
            gt = age.to(torch.float32).to(device)
            loss = F.mse_loss(pred.to(device), gt.to(device))
            ###################### End of your code ######################
            total_loss.append(loss.item())
        total_loss = sum(total_loss) / len(total_loss)
    
    
    return total_loss


def train_epoch(model, epoch, train_loader, val_loader):
    

    t_start = time.time()
    # Training:
    model.train()
    optimizer = optim.Adam(params=model.parameters(), lr=1e-4, weight_decay=1e-5)
    for img, age in train_loader:
        loss = None

        ############ TODO: calculate loss, update weights ############            
        optimizer.zero_grad()
        logits = model(img.to(device)).reshape(-1)
        labels = age.to(torch.float32).to(device)
        loss = F.mse_loss(logits.to(device), labels)
        
        loss.backward()
        optimizer.step()
        ###################### End of your code ######################
    print('Epoch No. {0}-- train loss = {1:.4f}'.format(
        epoch + 1,
        loss.item()
    ))

    # Validation:
    val_loss = get_performance(model, val_loader, device)
    print("Validation loss: {:.4f}".format(val_loss))
    best_model = model

    t_end = time.time()

    return best_model
class ImageDataset(Dataset):
    def __init__(self, img, age):
        self.img = img
        self.age = age
    def __len__(self):
        return len(self.age)

    def __getitem__(self, idx):
        img = torch.tensor(self.img[idx], dtype=torch.float32).permute(2, 0, 1)
        age = torch.tensor(self.age[idx])

        return img, age

class Adaptive_clf_Abernethy:
    def __init__(self, classifier, train_feature, train_label, train_group, val_feature, val_label, val_group):
        
        self.model = classifier
        self.pool_img, self.train_img, self.pool_label, self.train_label, self.pool_races, self.train_race = train_test_split(train_feature, train_label, train_group, test_size=2000, random_state=42) 
        
        self.val_feature = val_feature
        self.val_label = val_label 
        self.val_group = val_group
        
        self.fairness_violation = []
        self.train_loss = []
    
    
    def train(self, p=0.5, T = 16000):
        """
        train model according to the fairness metric.
            T: sample budget
            p: prob of choosing next sample from the whole population
        """
        train_dataset = ImageDataset(self.train_img, self.train_label)
        val_dataset = ImageDataset(self.val_feature, self.val_label)
        # train and record loss
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        self.model = train_epoch(self.model, 0, train_loader, val_loader)
        best_loss = 100000
        best_model = self.model
        for j in range(1, T + 1, 20):
            loss = []
            for i in range(5):
                idx = self.val_group == i
                img = self.val_feature[idx]
                label = self.val_label[idx]
                dataset = ImageDataset(img, label)
                data_loader = DataLoader(dataset, batch_size=1)
                loss.append(get_performance_test(self.model, data_loader, device))
            disadv_group = np.argmax(loss)
            print(f'fairness violation:{np.var(loss/np.sum(loss))}; disadvantaged group:{disadv_group}')
            if np.mean(loss) < best_loss:
                best_loss = np.mean(loss)
                best_model = self.model
            # choose where to sample 
            if np.random.random_sample() < p:
                print('select from the whole pool')
                index = np.random.randint(0, len(self.pool_label), size = 20)
                self.train_img = np.concatenate([self.train_img, self.pool_img[index]], axis=0)
                self.train_label = np.append(self.train_label, self.pool_label[index])
                self.train_race = np.append(self.train_race, self.pool_races[index])
                
                # self.pool_img = np.delete(self.pool_img, index, axis=0)
                # self.pool_label = np.delete(self.pool_label, index, axis=0)
                # self.pool_races = np.delete(self.pool_races, index, axis=0)
            else:
                print('select from disadvantaged group')
                img_shop = [self.pool_img[self.pool_races == i] for i in range(5)]
                label_shop = [self.pool_label[self.pool_races == i] for i in range(5)]
                race_shop = [self.pool_races[self.pool_races == i] for i in range(5)]

                index = np.random.randint(0, len(race_shop[disadv_group]), size=20)

                self.train_img = np.concatenate([self.train_img, img_shop[disadv_group][index]], axis = 0)
                self.train_label = np.append(self.train_label, label_shop[disadv_group][index])
                self.train_race = np.append(self.train_race, race_shop[disadv_group][index])
                
                # img_shop[disadv_group] = np.delete(img_shop[disadv_group], index, 0)
                # label_shop[disadv_group] = np.delete(label_shop[disadv_group], index, 0)
                # race_shop[disadv_group] = np.delete(race_shop[disadv_group], index, 0)
                
                self.pool_img = np.concatenate(img_shop, axis=0)
                self.pool_label = np.concatenate(label_shop, axis=0)
                self.pool_races = np.concatenate(race_shop, axis=0)
                np.random.shuffle(self.pool_img)
                np.random.shuffle(self.pool_label)
                np.random.shuffle(self.pool_races)
            # train again using new training set
            self.model = train_epoch(self.model, j, train_loader, val_loader)
        self.model = best_model


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
        # if len(filenames) > 500:
        #     break
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
    train_img, val_img, train_age, val_age, train_races, val_race, train_gender, val_gender = train_test_split(train_img, train_age, train_races, train_gender, test_size=0.3, random_state=5) 
    
    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
    model.classifier[6] = nn.Linear(in_features=4096, out_features=1, bias=True)
    model = model.to(device)
    
    adaptive_sampler = Adaptive_clf_Abernethy(model, train_img, train_age, train_races, val_img, val_age, val_race)
    adaptive_sampler.train()
    
    loss = []
    for i in range(5):
        idx = test_race == i
        print(sum(idx))
        img = test_img[idx]
        label = test_age[idx]
        dataset = ImageDataset(img, label)
        data_loader = DataLoader(dataset, batch_size=1)

        loss.append(get_performance_test(adaptive_sampler.model, data_loader, device))
    print(loss)
    print(np.var(loss/np.sum(loss)))