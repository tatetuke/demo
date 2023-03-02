'''
reference
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
import torchvision.transforms as transforms
import scipy.io.wavfile
import torch.nn as nn
import torch.nn.functional as F
import librosa
import librosa.display
import numpy as np
import pandas as pd
import argparse
import torch.optim as optim
from tqdm import tqdm
import setting

#データセットのファイル名
TRAIN_DATA_FILE_NAME ="TrainSet.csv"
NET_PATH = "network.pth"
SUB_DATA=setting.SUB_DATA


#ネットワークのパラメーター
LABEL_SIZE=4
BATCH_SIZE = 4
EPOCH=30
INF=100000000000000



#subdata:30
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.fc1 = nn.Linear(1500, 150)#入力サイズを変えた
        self.fc3 = nn.Linear(150, LABEL_SIZE)#出力サイズを変えた

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x
      

class MyDataset(torch.utils.data.Dataset):

    def __init__(self,data_size=INF,file_path=None, transform=None,out1vec=True):

        df = pd.read_csv(TRAIN_DATA_FILE_NAME)
        labels = df.iloc[:,0].to_numpy()
        audio_data = df.iloc[:,2:]
        # 音データ変換
        fft_data, _ = FILE2DATA(audio_data)
        self.transform = transform
        self.data_num = len(audio_data)
        self.data = []
        self.label = []
        for data in fft_data:
            self.data.append(data)
        for label in labels:
            # ohv=[0 for _ in range(LABEL_SIZE)]
            # ohv[label]=1
            # self.label.append(np.array(ohv))
            self.label.append(label)
            
        print("データのサイズ",np.shape(self.data))

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.label[idx]
        
        if self.transform:
            out_data = self.transform(out_data)

        return out_data, out_label
    
    

def FILE2DATA(files):
    datas = []
    rates = []
    for i in range(len(files)):
        amps = []
        for file in files.iloc[i, :]:
            rate, data = scipy.io.wavfile.read(file)
            data = data / 32768
            fft_size = 1024
            hop_length = int(fft_size / 4)
            amplitude = np.abs(librosa.core.stft(
                data, n_fft=fft_size, hop_length=hop_length))
            amps.append(np.array(amplitude))
        
        #HACK:NCHWをNHWCに書き替え
        indata=[]
        for i in range(len(amps[0])):
            indata2=[]
            for j in range(len(amps[0][0])-SUB_DATA):
                indata3=[]
                for c in range(len(amps)):
                    indata3.append(amps[c][i][j])
                indata2.append(indata3)
            indata.append(indata2)
        datas.append(indata)
        rates.append(rate)

    return np.array(datas), rates[0]


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--train', action='store_false')
    parser.add_argument('--dataset', choices=['normal','mixup', 'bilinear', 'cgan'],default='normal')
    

    args=parser.parse_args()

    net = Net()
    classes = ('左下','左上','右下','右上')
    

    if args.train:

        # ----------
        #  Training
        # ----------

        print("Load Train Data")
        trainset=MyDataset(file_path=TRAIN_DATA_FILE_NAME,
                                transform=transforms.ToTensor(),out1vec=False)
            
        trainloader=torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=0)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        net.to(device)
        
        #損失関数の設定（平均二乗誤差）
        criterion = nn.CrossEntropyLoss()


        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        print("Start Training")

        for epoch in tqdm(range(EPOCH),desc="Epoch"):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels] 
                inputs, labels = data[0].to(device),data[1].to(device)

                #データがdouble型になっているのでfloat型に変換
                inputs=inputs.float()
                # labels=labels.float()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

        torch.save(net.state_dict(), NET_PATH)

    else : 
        print("Load NetWork")
        net.load_state_dict(torch.load(NET_PATH))
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        net.to(device)



