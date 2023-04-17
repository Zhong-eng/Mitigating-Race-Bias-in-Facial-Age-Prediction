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
        self.age_pred = nn.Linear(in_features=1000, out_features=116, bias=True)
        self.race_class = nn.Linear(in_features=1000, out_features=5, bias=True)
        
    def forward(self, x):
        x_1 = self.Relu(self.shared(x))
        x_2 = x_1
        x_1 = self.age_pred(x_1)
        x_2 = self.race_class(x_2)

        return x_1, x_2


def PDF(age,variances):
    ages = [list(range(1,117)) for _ in age]
    result = []
    for a1,a2 in zip(age,ages):
        result.append(norm.pdf(a2,a1,variances[a1-1])/sum(norm.pdf(a2,a1,variances[a1-1])))
    result = np.array(result)
    return result

def get_variance(net, data_loader, device):
    net.eval()
    total_variance = [[] for _ in range(116)] 

    with torch.no_grad():
        for img, age in data_loader:
            pred = None # predictions for this batch
            ######## TODO: calculate loss, get predictions #########
            pred = net(img.to('cuda')).reshape(-1)
            total_variance[age.item() - 1].append(pred.item())
            ###################### End of your code ######################

    return [np.var(li) if len(li) > 0 else 0. for li in total_variance]

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
            gt = age.to(torch.float32).to('cuda')
            pred = F.log_softmax(pred, dim=1)
            criteron_age = nn.KLDivLoss(reduction="batchmean")
            loss = criteron_age(pred.to(device), gt.to(device))
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
            pred = F.log_softmax(pred, dim=1)
            gt = age.to(torch.float32).to(device)
            criteron_age = nn.KLDivLoss(reduction="batchmean")
            loss = criteron_age(pred.to(device), gt.to(device))
            ###################### End of your code ######################
            idx = torch.argmax(age[0]) // 10
            if idx >= 7:
                idx = 7
            total_loss[idx].append(loss.item())
    total_loss = [np.mean(each) for each in total_loss]
    
    return np.mean(total_loss)



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
            output_1 = F.log_softmax(output_1, dim=1)
            output_2 = F.softmax(output_2, dim=1)

            label_1 = age.to('cuda')
            criteron_race = nn.CrossEntropyLoss()
            criteron_age = nn.KLDivLoss(reduction="batchmean")
            label_2 = race.to('cuda')
            loss = 0.1*criteron_age(output_1.to('cuda'), label_1)-0.2*criteron_race(output_2.to('cuda'), label_2)
            
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
    filenames.remove('61_1_20170109150557335.jpg.chip.jpg')
    filenames.remove('61_1_20170109142408075.jpg.chip.jpg')
    filenames.remove('39_1_20170116174525125.jpg.chip.jpg')
    print(f"There are {len(filenames)} images in total")

    np.random.seed(42)

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
    
    age_group = []
    image_group = []
    for i in range(12):
        temp_age = []
        temp_img = []
        for j in range(len(train_age)):
            if 10*i <= train_age[j] < 10*(i+1):
                temp_age.append(train_age[j])
                temp_img.append(train_img[j])
        age_group.append(temp_age)
        image_group.append(temp_img)

    idx = np.random.choice(len(image_group[0]), 1500, replace=False)
    age_group[0] = np.array(age_group[0])[idx]
    image_group[0] = np.array(image_group[0])[idx]

    idx = np.random.choice(len(image_group[1]), 1500, replace=True)
    age_group[1] = np.array(age_group[1])[idx]
    image_group[1] = np.array(image_group[1])[idx]

    idx = np.random.choice(len(image_group[2]), 1500, replace=False)
    age_group[2] = np.array(age_group[2])[idx]
    image_group[2] = np.array(image_group[2])[idx]

    idx = np.random.choice(len(image_group[3]), 1500, replace=False)
    age_group[3] = np.array(age_group[3])[idx]
    image_group[3] = np.array(image_group[3])[idx]

    idx = np.random.choice(len(image_group[6]), 1500, replace=True)
    age_group[6] = np.array(age_group[6])[idx]
    image_group[6] = np.array(image_group[6])[idx]

    idx = np.random.choice(len(image_group[7]), 1000, replace=True)
    age_group[7] = np.array(age_group[7])[idx]
    image_group[7] = np.array(image_group[7])[idx]

    idx = np.random.choice(len(image_group[8]), 1000, replace=True)
    age_group[8] = np.array(age_group[8])[idx]
    image_group[8] = np.array(image_group[8])[idx]


    idx = np.random.choice(len(image_group[9]), 1000, replace=True)
    age_group[9] = np.array(age_group[9])[idx]
    image_group[9] = np.array(image_group[9])[idx]

    idx = np.random.choice(len(image_group[10]), 1000, replace=True)
    age_group[10] = np.array(age_group[10])[idx]
    image_group[10] = np.array(image_group[10])[idx]

    idx = np.random.choice(len(image_group[11]), 1000, replace=True)
    age_group[11] = np.array(age_group[11])[idx]
    image_group[11] = np.array(image_group[11])[idx]

    image_group = [np.array(each) for each in image_group]
    train_img = np.concatenate(image_group)
    age_group = [np.array(each) for each in age_group]
    train_age = np.concatenate(age_group)

    p = np.random.permutation(len(train_age))
    train_age = train_age[p]
    train_img = train_img[p]

    variances = [0.18307534268723666, 0.8575262230710946, 1.4759993067071526, 2.230822529884938, 6.649663793711493, 2.515879907526478, 3.089804728942829, 3.450409802673334, 5.378499087419674, 7.3403557231145475, 4.316240471593102, 3.9717547076188247, 4.71750174051434, 5.639121512248396, 6.335008995858168, 4.487735015692444, 6.412114289036048, 6.007844575603222, 5.2830414197111075, 5.036368655585688, 5.660433440187571, 3.563586563300856, 3.8391469787987402, 3.9130159710354735, 4.172653613342597, 5.363474477838178, 5.691438469031803, 4.258333159944988, 5.241676951138231, 5.117169275026069, 5.564173378956198, 6.485681144482222, 7.165389872592505, 6.854551453304662, 10.521665822475457, 9.984825800681437, 12.601421903098684, 15.243072542316353, 14.925115152170523, 18.705826923330932, 20.727869836916827, 18.51435104124787, 23.815570525593404, 22.292855949246587, 23.02207815944548, 16.827907646799147, 18.628691653278977, 22.524312290781157, 24.454528714075042, 27.262928976888695, 24.988085937916846, 23.328170815060115, 27.414943061832663, 29.62284827045376, 23.112990274452486, 22.701961991575644, 20.108095482902616, 26.11081481196221, 16.920671168427216, 25.579270652614117, 30.33191257353101, 29.573796858540497, 24.937558151024508, 20.87356444451234, 23.312352329321516, 26.399453922725133, 26.651204347377714, 32.721362711880644, 22.311954887671554, 26.450237424983257, 19.32760057511645, 20.927159745735004, 22.195454650861524, 16.835712041860543, 23.680013305544726, 29.43671835635599, 22.287304153335135, 33.60332037049843, 21.28317927563258, 31.141837132627895, 30.256606632379572, 15.100100485617023, 17.3387691437942, 36.81011687697413, 29.51547107350869, 24.399201459915137, 24.468125512514234, 23.86446722469175, 57.20617961759491, 34.537026539486234, 6.922665562538896, 7.887746018933185, 89.54267290447994, 0.0, 41.66917212589225, 41.119323088607615, 0.0, 0.0, 51.64723604379403, 64.32901710921487, 1.9045819379389286, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 14.591357069599326, 0.0, 0.0, 0.0, 0.0, 101.41100699533449, 77.25483910009886]
    variances = np.array(variances)

    variances[variances == 0] = np.mean(variances[variances != 0])
    train_age_pdf = PDF(train_age,variances)
    val_age_pdf = PDF(val_age,variances)
    test_age_pdf = PDF(test_age,variances)

    ## TODO:Fake Train_race
    race_number = train_img.shape[0] // 5
    fake_train_races = [0] * race_number + [1] * race_number + [2] * race_number + [3] * race_number + [4] * (train_img.shape[0] - 4 * race_number)

    np.random.shuffle(fake_train_races)
    train_dataset = ImageDataset(train_img, train_age_pdf, train_races)
    val_dataset = ImageDataset(val_img, val_age_pdf, val_race)

    train_loader = DataLoader(train_dataset, batch_size=32)
    val_loader = DataLoader(val_dataset, batch_size=32)


    ####### RaceIndifferenceModel Model#######
    model_ = RaceIndifferenceModel().to(device)

    best_mode_ = train_indiff(model_, 80, train_loader, val_loader)
    torch.save(best_mode_, 'checkpoint/RaceIndifferenceModel_fuzzy.pt')

    loss = []
    for i in range(5):
        idx = test_race == i
        print(sum(idx))
        img = test_img[idx]
        label = test_age_pdf[idx]
        dataset = ImageDataset(img, label, test_race[idx])
        data_loader = DataLoader(dataset, batch_size=1)

        loss.append(get_performance_indiff_test(best_mode_, data_loader, 'cuda'))
    print(loss)
    print(np.var(loss/np.sum(loss)))
    