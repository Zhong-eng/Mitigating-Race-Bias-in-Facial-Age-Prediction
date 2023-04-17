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
from scipy.stats import norm

def PDF(age,variances):
    ages = [list(range(1,117)) for _ in age]
    result = []
    for a1,a2 in zip(age,ages):
        result.append(norm.pdf(a2,a1,variances[a1-1])/sum(norm.pdf(a2,a1,variances[a1-1])))
    result = np.array(result)
    return result

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
            pred = net(img.to('cuda'))
            pred = F.log_softmax(pred, dim=1)
            gt = age.to('cuda')
            kl_loss = nn.KLDivLoss(reduction="batchmean")
            loss = kl_loss(pred.to('cuda'), gt)
            # loss = F.kl_div(pred.to(device), gt.to(device))
            ###################### End of your code ######################
            idx = torch.argmax(age[0]).item() // 10
            if idx >= 7:
                idx = 7
            total_loss[idx].append(loss.item())
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
            pred = net(img.to('cuda'))
            pred = F.log_softmax(pred, dim=1)
            gt = age.to('cuda')
            kl_loss = nn.KLDivLoss(reduction="batchmean")
            loss = kl_loss(pred.to('cuda'), gt)
            # loss = F.kl_div(pred.to(device), gt.to(device))
            ###################### End of your code ######################
            total_loss.append(loss.item())
    if len(total_loss) > 0:
      total_loss = sum(total_loss) / len(total_loss)
    else:
      return 0.
    
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
        logits = model(img.to(device))
        logits = F.log_softmax(logits, dim=1)
        labels = age.to(torch.float32).to(device)
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        loss = kl_loss(logits.to('cuda'), labels)
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
        torch.save(self.model, 'checkpoint/adaptive_fuzzy.pt')


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
    
    variances = [0.18307534268723666, 0.8575262230710946, 1.4759993067071526, 2.230822529884938, 6.649663793711493, 2.515879907526478, 3.089804728942829, 3.450409802673334, 5.378499087419674, 7.3403557231145475, 4.316240471593102, 3.9717547076188247, 4.71750174051434, 5.639121512248396, 6.335008995858168, 4.487735015692444, 6.412114289036048, 6.007844575603222, 5.2830414197111075, 5.036368655585688, 5.660433440187571, 3.563586563300856, 3.8391469787987402, 3.9130159710354735, 4.172653613342597, 5.363474477838178, 5.691438469031803, 4.258333159944988, 5.241676951138231, 5.117169275026069, 5.564173378956198, 6.485681144482222, 7.165389872592505, 6.854551453304662, 10.521665822475457, 9.984825800681437, 12.601421903098684, 15.243072542316353, 14.925115152170523, 18.705826923330932, 20.727869836916827, 18.51435104124787, 23.815570525593404, 22.292855949246587, 23.02207815944548, 16.827907646799147, 18.628691653278977, 22.524312290781157, 24.454528714075042, 27.262928976888695, 24.988085937916846, 23.328170815060115, 27.414943061832663, 29.62284827045376, 23.112990274452486, 22.701961991575644, 20.108095482902616, 26.11081481196221, 16.920671168427216, 25.579270652614117, 30.33191257353101, 29.573796858540497, 24.937558151024508, 20.87356444451234, 23.312352329321516, 26.399453922725133, 26.651204347377714, 32.721362711880644, 22.311954887671554, 26.450237424983257, 19.32760057511645, 20.927159745735004, 22.195454650861524, 16.835712041860543, 23.680013305544726, 29.43671835635599, 22.287304153335135, 33.60332037049843, 21.28317927563258, 31.141837132627895, 30.256606632379572, 15.100100485617023, 17.3387691437942, 36.81011687697413, 29.51547107350869, 24.399201459915137, 24.468125512514234, 23.86446722469175, 57.20617961759491, 34.537026539486234, 6.922665562538896, 7.887746018933185, 89.54267290447994, 0.0, 41.66917212589225, 41.119323088607615, 0.0, 0.0, 51.64723604379403, 64.32901710921487, 1.9045819379389286, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 14.591357069599326, 0.0, 0.0, 0.0, 0.0, 101.41100699533449, 77.25483910009886]
    variances = np.array(variances)

    variances[variances == 0] = np.mean(variances[variances != 0])
    train_age_pdf = PDF(train_age,variances)
    val_age_pdf = PDF(val_age,variances)
    test_age_pdf = PDF(test_age,variances)

    train_dataset = ImageDataset(train_img, train_age_pdf)
    val_dataset = ImageDataset(val_img, val_age_pdf)

    train_loader = DataLoader(train_dataset, batch_size=32)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
    model.classifier[6] = nn.Linear(in_features=4096, out_features=116, bias=True)
    model = model.to(device)
    
    adaptive_sampler = Adaptive_clf_Abernethy(model, train_img, train_age_pdf, train_races, val_img, val_age_pdf, val_race)
    adaptive_sampler.train(p=0.7)
    
    loss = []
    for i in range(5):
        idx = test_race == i
        print(sum(idx))
        img = test_img[idx]
        label = test_age_pdf[idx]
        dataset = ImageDataset(img, label)
        data_loader = DataLoader(dataset, batch_size=1)

        loss.append(get_performance_test(adaptive_sampler.model, data_loader, device))
    print(loss)
    print(np.var(loss/np.sum(loss)))