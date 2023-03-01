'''
references
mixup:https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py
intermix:https://github.com/scaleracademy/InterMix-ICASSP/blob/main/main.py
speechMix:https://github.com/midas-research/speechmix
'''
from __future__ import print_function

import numpy as np
import torch
import librosa.display
import librosa
import pandas as pd
import scipy.io.wavfile
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import randint
import setting
import math

# CSVファイルの場所
CSV_PATH = "../../"
TRAIN_DATA_FILE_NAME = setting.TRAIN_DATA_FILE_NAME
TEST_DATA_FILE_NAME = setting.TEST_DATA_FILE_NAME


ALPHA = 0.25 # 細かい分け方
DIST=setting.DIST  #座標一単位当たりの長さ(cm)
RATE = 44100
INF=100000000
EPS=0.01

#データを削る
SUB_DATA=setting.SUB_DATA


# 座標ラベルの最小距離
MIN_DIST = setting.MIN_DIST
MIN_DIST2=setting.MIN_DIST2
Y_MAX=setting.Y_MAX
X_MAX=setting.X_MAX

#引数はフーリエ変換済みのデータ
def showgraph(data):
    # 縦軸（振幅）の配列を作成   #16bitの音声ファイルのデータを-1から1に正規化
    #rt_data = data / 32768
    # フレーム長
    fft_size = 1024
    # フレームシフト長
    hop_length = int(fft_size/4)
    log_power = librosa.core.amplitude_to_db(data)
    print(np.shape(log_power))
    librosa.display.specshow(
        log_power, sr=RATE, hop_length=hop_length, x_axis='time', y_axis='hz', cmap='magma')
    plt.title('スペクトログラム', fontname="MS Gothic")
    plt.pause(3)
    plt.cla()

#音声データをSTFTする
def STFT(data):
    stft_data=[]
 
    fft_size = 1024# フレーム長
    hop_length = int(fft_size / 4)# フレームシフト長
    for i in range(len(data)):
        mic_pair=[]
        for wav in data[i]:
            wav = wav / 32768# 縦軸（振幅）の配列を作成   #16bitの音声ファイルのデータを-1から1に正規化
            amplitude = np.abs(librosa.core.stft(
                wav, n_fft=fft_size, hop_length=hop_length))# 短時間フーリエ変換
            mic_pair.append(np.array(amplitude))
        stft_data.append(mic_pair)
    return stft_data

#時間方向にデータを削減する
def SubData(data):
    ret_data=[]
    for files in data:
        resized_files=[]
        for file in files:
            resized_file=[]
            for i in range(len(file)):
                raw_data=[]
                for j in range(len(file[0])-SUB_DATA):
                    raw_data.append(file[i][j])
                resized_file.append(raw_data)
            resized_files.append(resized_file)
        ret_data.append(resized_files)

    return ret_data

#ファイルの読み込み
def LoadData(files):
    datas = []
    rates = []
    for i in tqdm(range(len(files))):
        tmp = []
        for file in files.iloc[i, :]:
            rate, data = scipy.io.wavfile.read(file)
            tmp.append(np.array(data))
            rates.append(rate)
        datas.append(tmp)

    return np.array(datas), rates[0]

#HACK:NCHWをNHWCに書き替える。もっと簡潔な書き方がある(はず)
#BUG:憎きtransformerを使う必要がなくなったので、こちらも必要なくなった。transormerに関わる実装のBUGが取り切れていない。
def NCHW2NHWC(data):
    resized_stft_data=[]
    for amps in data:
        indata=[]
        for i in range(len(amps[0])):
                indata2=[]
                for j in range(len(amps[0][0])):
                    indata3=[]
                    for c in range(len(amps)):
                        indata3.append(amps[c][i][j])
                    indata2.append(indata3)
                indata.append(indata2)
        resized_stft_data.append(indata)
    
    return resized_stft_data

#データを一次元化
def FlattenData(data):
    flatten_datas=[]
    for files in range(len(data)):
        flatten_data = np.concatenate([np.array(files).flatten()], 0)
        flatten_datas.append(flatten_data)
    return flatten_datas

#csvファイルからデータをロード
def LoadFilesFromCSV(csv_file_path):
    datas=[]
    labels=[]
    if csv_file_path:
        df = pd.read_csv(csv_file_path,header=None)
    else:
        df = pd.read_csv(TRAIN_DATA_FILE_NAME,header=None)
    labels = df.iloc[:, 0:2].to_numpy()
    datas = df.iloc[:, 2:]
    
    return datas,labels
    
    
def LOAD_STFT(files,out1vec):
    datas = []
    #音声データをwavファイルからロード
    datas,rate=LoadData(files)
    #音声データをSTFT
    datas=STFT(datas)
    #音声データを時間方向に削減
    datas=SubData(datas)
    #音声データの配列を変形する
    #datas=NCHW2NHWC(datas)
    #SVR用に一次元配列を変換
    if out1vec:
        datas=FlattenData(datas)
                
    return np.array(datas), rate

    
def Mixup(x1, y1, x2, y2, alpha=0.1,l2=False,isp=False):
        
    def G(x):
        return 10*math.log10(x)
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    X = []
    Y = []
    
    mixed_y = lam * y1 + (1 - lam) * y2
    Y.append(mixed_y)
    if isp:
        mixed_x=x1
        for c in range(len(x1)):
            for i in range(len(x1[c])):
                for j in range(len(x1[c][i])):
                    p=1/(1+10**((G(x1[c][i][j])-G(x2[c][i][j]))/20)*(1-lam)/lam)
                    mixed_x[c][i][j]= (p*x1[c][i][j]+(1-p)*x2[c][i][j])/math.sqrt(p**2+(1-p)**2)
        X.append(mixed_x) 
    else:              
        mixed_x = lam * x1 + (1 - lam) * x2
        if l2:
            div=math.sqrt((lam)**2+(1-lam)**2)
            mixed_x/=div
        X.append(mixed_x)
    
        
    return X, Y

   

class MyDataset(torch.utils.data.Dataset):

    def __init__(self,file_path,data_size=INF, transform=None,out1vec=True):
        #CSVファイルから音データを読み込み
        audio_data,xy_data=LoadFilesFromCSV(file_path)
        # 音データ変換
        fft_data, _RATE = LOAD_STFT(audio_data,out1vec)

        self.transform = transform
        self.data_num = len(audio_data)
        
        self.data = []
        self.label = []
        
        # HACK:ラベルごとにデータを分類する。（クラス内法表記とかもっと簡潔な書き方がありそう）
        unique_label = np.unique(xy_data, axis=0)
        label_data = []  # ラベルごとに分類したデータ。i番目のデータのラベルはuniue_label[i]
        for target_label in unique_label:
            push_data = []
            for i, label in enumerate(xy_data):
                if not all(label == target_label):continue
                push_data.append(i)
            label_data.append(push_data)
        
        for i in range(len(unique_label)):
            idx_perm=torch.randperm(len(label_data[i]))#ランダムに選ぶ
            for k in range(min(len(idx_perm),data_size)):
                # idx=label_data[i][idx_perm[k]]
                idx=label_data[i][k]
                self.data.append(fft_data[idx])
                self.label.append(np.array(xy_data[idx])*DIST)
                
        print("データのサイズ",np.shape(self.data))

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.label[idx]
        
        # if self.transform:
        #     out_data = self.transform(out_data)

        return out_data, out_label
    
class RawAudioDataset(torch.utils.data.Dataset):

    def __init__(self,file_path,data_size=INF, transform=None,out1vec=True):
        #CSVファイルから音データを読み込み
        audio_data,xy_data=LoadFilesFromCSV(file_path)
        
        # 音データ読み込み
        fft_data, _RATE = LoadData(audio_data)

        self.transform = transform
        self.data_num = len(audio_data)
        
        self.data = []
        self.label = []
        
        # HACK:ラベルごとにデータを分類する。（クラス内法表記とかもっと簡潔な書き方がありそう）
        unique_label = np.unique(xy_data, axis=0)
        label_data = []  # ラベルごとに分類したデータ。i番目のデータのラベルはuniue_label[i]
        for target_label in unique_label:
            push_data = []
            for i, label in enumerate(xy_data):
                if not all(label == target_label):continue
                push_data.append(i)
            label_data.append(push_data)
        
        for i in range(len(unique_label)):
            idx_perm=torch.randperm(len(label_data[i]))
            for k in range(min(len(idx_perm),data_size)):
                idx=label_data[i][idx_perm[k]]
                self.data.append(fft_data[idx])
                self.label.append(np.array(xy_data[idx])*DIST)
        print("データのサイズ",np.shape(self.data))

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.label[idx]
        
        # if self.transform:
        #     out_data = self.transform(out_data)

        return out_data, out_label

class MixRawAudioDataset(torch.utils.data.Dataset):
    def __init__(self,trainset=None,file_path=None,data_size=INF, transform=None,out1vec=True,num=1,augmentation_method='None',l2=False):
        #検証用
        if trainset==None:
            print("error")
            if file_path:
                trainset=RawAudioDataset(file_path=file_path,out1vec=out1vec)
            else:
                trainset =  RawAudioDataset(file_path=TRAIN_DATA_FILE_NAME,out1vec=out1vec)
        
        #Mixup or BilinearMixup でデータ拡張
        #通常とは異なり、この時点でデータセット中のデータは音響信号(波形)
        if augmentation_method=='mixup':
            self.data,self.label=MixupGenerator(trainset,num=num,l2=l2)
        elif augmentation_method=='bilinear':
            self.data,self.label=BilinearMixupGenerator(trainset,num=num,l2=l2)
        elif augmentation_method=='multi':
            self.data,self.label=MultiMixGenerator(trainset,num=num,l2=l2)
        else:assert True,"unexpected method in MixRawAudio Class"
        
        self.transform = transform
        self.data_num = len(self.label)
        #音声データをSTFT
        self.data=STFT(self.data)
        #音声データを時間方向に削減
        self.data=SubData(self.data)
        #音声データの配列を変形する
        # self.data=NCHW2NHWC(self.data)
        #SVR用に一次元配列を変換
        if out1vec:
            self.data=FlattenData(self.data)
        self.data=np.array(self.data)                    
        
    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.label[idx]
        
        # if self.transform:
        #     out_data = self.transform(out_data)

        return out_data, out_label



#MixUpしたデータを含むデータセット
class MyMixUpData(torch.utils.data.Dataset):

    def __init__(self, trainDatasets=None,file_path=None, transform=None,out1vec=True,num=1,l2=False,isp=False):

        self.data=[]
        self.label=[]

        #検証用
        if trainDatasets:
            self.data,self.label = MixupGenerator(trainDatasets,num=num,l2=l2,isp=isp)
        else:
            if file_path:
                self.data,self.label = MixupGenerator(MyDataset(file_path=file_path,out1vec=out1vec),num=num,l2=l2,isp=isp)
            else:
                self.data,self.label =  MixupGenerator(MyDataset(file_path=TRAIN_DATA_FILE_NAME,out1vec=out1vec),num=num,l2=l2,isp=isp)
        
        self.transform = transform
        self.data_num = len(self.label)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.label[idx]
        
        # if self.transform:
        #     out_data = self.transform(out_data)

        return out_data, out_label


#MixUpしたデータを含むデータセット
class BilinearMixUpDataset(torch.utils.data.Dataset):

    def __init__(self, trainDatasets=None,file_path=None, transform=None,out1vec=True,num=1,l2=False):

        self.data=[]
        self.label=[]
        if trainDatasets:
            self.data,self.label = BilinearMixupGenerator(trainDatasets,num=num,l2=l2)
        else:
            if file_path:
                self.data,self.label = BilinearMixupGenerator(MyDataset(file_path=file_path,out1vec=out1vec),num=num,l2=l2)
            else:
                self.data,self.label =  BilinearMixupGenerator(MyDataset(file_path=TRAIN_DATA_FILE_NAME,out1vec=out1vec),num=num,l2=l2)
        print("データ拡張後のサイズ",np.shape(self.data))
        self.transform = transform
        self.data_num = len(self.label)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.label[idx]
        
        # if self.transform:
        #     out_data = self.transform(out_data)

        return out_data, out_label


#MixUpしたデータを含むデータセット
class MultiMixDataset(torch.utils.data.Dataset):

    def __init__(self, trainDatasets=None,file_path=None, transform=None,out1vec=True,num=1,l2=False):

        self.data=[]
        self.label=[]
        if trainDatasets:
            self.data,self.label = MultiMixGenerator(trainDatasets,num=num,l2=l2)
        else:
            if file_path:
                self.data,self.label = MultiMixGenerator(MyDataset(file_path=file_path,out1vec=out1vec),num=num,l2=l2)
            else:
                self.data,self.label =  MultiMixGenerator(MyDataset(file_path=TRAIN_DATA_FILE_NAME,out1vec=out1vec),num=num,l2=l2)
        print("データ拡張後のサイズ",np.shape(self.data))
        self.transform = transform
        self.data_num = len(self.label)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.label[idx]
        
        # if self.transform:
        #     out_data = self.transform(out_data)

        return out_data, out_label

class MiddleLayerMixDataset(torch.utils.data.Dataset):

    def __init__(self, trainDatasets=None,file_path=None, transform=None,out1vec=True,num=1,algo=None):
        self.data=[]
        self.label=[]
        self.rate=[]
        if trainDatasets:
            if algo=='bilinear':
                self.data,self.label,self.rate = MidBilinearGenerator(trainDatasets,num=num)
            elif algo=='mixup':
                self.data,self.label,self.rate = MidMixupGenerator(trainDatasets,num=num)
            elif algo=='multi':
                self.data,self.label,self.rate = MidMultiGenerator(trainDatasets,num=num)
        else:
            if file_path:
                if algo=='bilinaer':
                    self.data,self.label,self.rate = MidBilinearGenerator(MyDataset(file_path=file_path,out1vec=out1vec),num=num)
                elif algo=='mixup':
                      self.data,self.label,self.rate = MidMixupGenerator(MyDataset(file_path=file_path,out1vec=out1vec),num=num)
                elif algo=='multi':
                     self.data,self.label,self.rate = MidMultiGenerator(MyDataset(file_path=file_path,out1vec=out1vec),num=num)
            else:
                self.data,self.label,self.rate =  MidBilinearGenerator(MyDataset(file_path=TRAIN_DATA_FILE_NAME,out1vec=out1vec),num=num)
                print("error:データセットとして読み込むcsvファイルが指定されていません")
        print("データ拡張後のサイズ",np.shape(self.data))
        self.transform = transform
        self.data_num = len(self.label)
        

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.label[idx]
        out_r=self.rate[idx]
        return out_data, out_label,out_r


# マンハッタン距離がMIN_DISTであるとき生成
def MixupGenerator(trainset,num=10,l2=False,isp=False):
    # HACK:ラベルごとにデータを分類する。（クラス内法表記とかもっと簡潔な書き方がありそう）
    unique_label = np.unique(trainset.label, axis=0)
    label_data = []  # ラベルごとに分類したデータ。i番目のデータのラベルはuniue_label[i]

    for target_label in unique_label:
        push_data = []
        for i, label in enumerate(trainset.label):
            if not all(label == target_label):continue
            push_data.append(i)
        label_data.append(push_data)

    # ラベルの組み合わせごとに生成
    # ラベル（xy座標）のマンハッタン距離が0より大きい最小距離であればミックスアップする
    unique_idx_pair=[]
    for i, label1 in enumerate(unique_label):
        for j, label2 in enumerate(unique_label):
            if i < j:
                break  # ラベルの組み合わせの重複を消す
            # if np.sum(np.abs(label1-label2)) > MIN_DIST+EPS:
            #     continue
            # unique_idx_pair.append([i,j])
            if np.sum(np.square(np.abs(label1-label2)))==MIN_DIST2:
                unique_idx_pair.append([i,j])
            if np.sum(np.square(np.abs(label1-label2)))==MIN_DIST*MIN_DIST:
                unique_idx_pair.append([i,j])
    
    data = []
    label = []
    for _ in range(num):
        #mixupするデータの組み合わせをランダムに決める
        idx_pair=unique_idx_pair[np.random.randint(0,len(unique_idx_pair))]
        i=idx_pair[0]
        j=idx_pair[1]
        label1=unique_label[i]
        label2=unique_label[j]
        #ラベルi,jのデータの中からそれぞれランダムにmixupするデータを取得する
        data1_idx=label_data[i][np.random.randint(0,len(label_data[i]))]
        data2_idx=label_data[j][np.random.randint(0,len(label_data[j]))]
        data1=trainset.data[data1_idx]
        data2=trainset.data[data2_idx]
        
        mixup_data, mixup_label = Mixup(
            data1, label1, data2, label2, alpha=0.1,l2=l2,isp=isp)
        
        data.extend(mixup_data)
        label.extend(mixup_label)
        
        #######################################################
        #              mixup　のテスト
        #######################################################
        if __name__=='__main__':
          #元のデータA
          tmpdA=[]
          print("元データA")
          for i in range(len(data1)):
            tmpdB=[]
            for j in range(len(data1[0])):
              tmpdB.append(data1[i][j][0])
            tmpdA.append(tmpdB)
          print(np.shape(tmpdA))
          showgraph(tmpdA)
          
          #元のデータB
          tmpdA=[]
          print("元データB")
          for i in range(len(data2)):
            tmpdB=[]
            for j in range(len(data2[0])):
              tmpdB.append(data2[i][j][0])
            tmpdA.append(tmpdB)
          print(np.shape(tmpdA))
          showgraph(tmpdA)
          
          #mixupデータ
          # for c in range(len(mixup_data[0][0][0])):
          print("mixupデータ")
          tmpdA=[]
          for i in range(len(mixup_data[0])):
            tmpdB=[]
            for j in range(len(mixup_data[0][0])):
              tmpdB.append(mixup_data[0][i][j][0])
            tmpdA.append(tmpdB)
          showgraph(tmpdA)
        #######################################################
        #              mixup　のテスト
        #######################################################
        
    for i in range(len(trainset.data)):
        data.append(trainset.data[i])
        label.append(trainset.label[i])
    
    return data, label


# 座標をランダムに生成し、生成した座標に近い4点を混ぜ合わせる
def BilinearMixupGenerator(trainset,num=1,l2=False):
    
    def BilinearMixup(y,datas,idx_dict,l2):
        Y=[]
        dxy=[[0,0],[1,0],[0,1],[1,1]]
        
        for xy in dxy:
            Y.append([(y[0]//MIN_DIST+xy[0])*MIN_DIST,
                    (y[1]//MIN_DIST+xy[1])*MIN_DIST])
            
        idx=[idx_dict[tuple(Y[i])] for i in range(len(Y))]
        data=[datas[i][randint(0,len(datas[i])-1)] for i in idx]
        Data=np.array(datas[0][0])
        Data=Data-np.array(datas[0][0])
        div=0
        for i in range(len(Y)):
            Data=Data+np.array(data[i])*(MIN_DIST-abs(Y[i][0]-y[0]))/MIN_DIST*(MIN_DIST-abs(Y[i][1]-y[1]))/MIN_DIST
            div+=((MIN_DIST-abs(Y[i][0]-y[0]))/MIN_DIST*(MIN_DIST-abs(Y[i][1]-y[1]))/MIN_DIST)**2
        #エネルギーを考慮して、√r1^2+r2^2+r3^2+r^4で割る
        if l2:
            Data/=math.sqrt(div)
        
        return Data
        
        
    print("BileanearMixupで生成したデータ",num,"個")

    # HACK:ラベルごとにデータを分類する。（クラス内法表記とかもっと簡潔な書き方がありそう）
    unique_label = np.unique(trainset.label, axis=0)
    label_data = []  # ラベルごとに分類したデータ。i番目のデータのラベルはuniue_label[i]
    idx_dict={}
    for i, label in enumerate(unique_label):
        idx_dict[tuple(label)]=i

    for target_label in unique_label:
        push_data = []
        for i, label in enumerate(trainset.label):
            if not all(label == target_label):continue
            push_data.append(trainset.data[i])
        label_data.append(push_data)
        
    data = []
    label = []
    for _ in range(num):
        Y=np.random.rand(2)*np.array([X_MAX,Y_MAX])
        X=BilinearMixup(y=Y,datas=label_data,idx_dict=idx_dict,l2=l2)
        data.append(X)
        label.append(Y)
    
    for i in range(len(trainset.data)):
        data.append(trainset.data[i])
        label.append(trainset.label[i])
    data,label=np.array(data),np.array(label)
    return data, label

##3つのデータで混合
def MultiMixGenerator(trainset,num=1,l2=False):
    
    def CalRate(X,Y,x1,x2,x3,y1,y2,y3):
        # if x1-x3==0:
        #     b=(X-x3)/(x2-x3)
        # else:
        b=((Y-y3)*(x1-x3)-(X-x3)*(y1-y3))/(-(x2-x3)*(y1-y3)+(y2-y3)*(x1-x3))
        # if x2-x3==0:
        #     a=(X-x3)/(x1-x3)
        # else:
        a=((Y-y3)*(x2-x3)-(X-x3)*(y2-y3))/(-(x1-x3)*(y2-y3)+(y1-y3)*(x2-x3))
        return a,b
        
    def MultiMix(x1,x2,x3,a,b,l2=False):
        x1=np.array(x1)
        x2=np.array(x2)
        x3=np.array(x3)
        
        X = x1*a+x2*b+(1-a-b)*x3
        if l2:
            div=math.sqrt(a**2+b**2+(1-a-b)**2)
            X/=div
        return X
            
        
    print("MultiMixで生成したデータ",num,"個")
    
    # HACK:ラベルごとにデータを分類する。（クラス内法表記とかもっと簡潔な書き方がありそう）
    unique_label = np.unique(trainset.label, axis=0)
    label_data = []  # ラベルごとに分類したデータ。i番目のデータのラベルはuniue_label[i]

    for target_label in unique_label:
        push_data = []
        for i, label in enumerate(trainset.label):
            if not all(label == target_label):continue
            push_data.append(trainset.data[i])
        label_data.append(push_data)
    
    data = []
    label = []
    for _ in range(num):
        Y=np.random.rand(2)*np.array([X_MAX,Y_MAX])
        y1,y2,y3=-1,-1,-1
        y1_dist,y2_dist,y3_dist=INF,INF,INF
        for i,lab in enumerate(unique_label):
            dist=math.sqrt((lab[0]-Y[0])**2+(lab[1]-Y[1])**2)
            if y1_dist>dist:
                y3_dist=y2_dist
                y2_dist=y1_dist
                y1_dist=dist
                y3=y2
                y2=y1
                y1=i
            elif y2_dist>dist:
                y3_dist=y2_dist
                y2_dist=dist
                y3=y2
                y2=i
            elif y3_dist>dist:
                y3_dist=dist
                y3=i
        x1=label_data[y1][np.random.randint(0,len(label_data[y1]))]        
        x2=label_data[y2][np.random.randint(0,len(label_data[y2]))]        
        x3=label_data[y3][np.random.randint(0,len(label_data[y3]))]        
        y1=unique_label[y1]
        y2=unique_label[y2]
        y3=unique_label[y3]
        a,b=CalRate(X=Y[0],Y=Y[1],x1=y1[0],x2=y2[0],x3=y3[0],y1=y1[1],y2=y2[1],y3=y3[1])
        X=MultiMix(x1,x2,x3,a,b,l2=l2)
        data.append(X)
        label.append(Y)
   
    
    for i in range(len(trainset.data)):
        data.append(trainset.data[i])
        label.append(trainset.label[i])
    data,label=np.array(data),np.array(label)
    return data, label



def MidMixupGenerator(trainset,num=1):
    
    def MidMixup(y1, y2, alpha=0.1):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        mixed_y = lam * y1 + (1 - lam) * y2
        r=[lam,1-lam]
        return r, mixed_y
        
        
    # HACK:ラベルごとにデータを分類する。（クラス内法表記とかもっと簡潔な書き方がありそう）
    unique_label = np.unique(trainset.label, axis=0)
    label_data = []  # ラベルごとに分類したデータ。i番目のデータのラベルはuniue_label[i]

    for target_label in unique_label:
        push_data = []
        for i, label in enumerate(trainset.label):
            if not all(label == target_label):continue
            push_data.append(i)
        label_data.append(push_data)

    # ラベルの組み合わせごとに生成
    # ラベル（xy座標）のマンハッタン距離が0より大きい最小距離であればミックスアップする
    unique_idx_pair=[]
    for i, label1 in enumerate(unique_label):
        for j, label2 in enumerate(unique_label):
            if i < j:
                break  # ラベルの組み合わせの重複を消す
            # if np.sum(np.abs(label1-label2)) > MIN_DIST+EPS:
            #     continue
            # unique_idx_pair.append([i,j])
            if np.sum(np.square(np.abs(label1-label2)))==MIN_DIST2:
                unique_idx_pair.append([i,j])
            if np.sum(np.square(np.abs(label1-label2)))==MIN_DIST*MIN_DIST:
                unique_idx_pair.append([i,j])
    
    data = []
    label = []
    rate = []
           
    for i in range(len(trainset.data)):
        data.append([trainset.data[i],trainset.data[i]])
        
        label.append(trainset.label[i])
        rate.append(np.array([1,0]))
        
    for _ in range(num):
        #mixupするデータの組み合わせをランダムに決める
        idx_pair=unique_idx_pair[np.random.randint(0,len(unique_idx_pair))]
        i=idx_pair[0]
        j=idx_pair[1]
        label1=unique_label[i]
        label2=unique_label[j]
        #ラベルi,jのデータの中からそれぞれランダムにmixupするデータを取得する
        data1_idx=label_data[i][np.random.randint(0,len(label_data[i]))]
        data2_idx=label_data[j][np.random.randint(0,len(label_data[j]))]
        data1=trainset.data[data1_idx]
        data2=trainset.data[data2_idx]
        
        r, mixup_label = MidMixup(
            label1,label2, alpha=0.1)
        
        data.append([data1,data2])
        label.append(mixup_label)
        rate.append(r)
    print(np.shape(data))
    
 
    data,label,rate=np.array(data),np.array(label),np.array(rate)
    
    return data, label,rate


# 座標をランダムに生成し、生成した座標に近い4点を混ぜ合わせる
def MidBilinearGenerator(trainset,num=1):
    
    def BilinearMixup(y,datas,idx_dict):
        Y=[]
        R=[]
        dxy=[[0,0],[1,0],[0,1],[1,1]]
        
        for xy in dxy:
            Y.append([(y[0]//MIN_DIST+xy[0])*MIN_DIST,
                    (y[1]//MIN_DIST+xy[1])*MIN_DIST])
        
        idx=[idx_dict[tuple(Y[i])] for i in range(len(Y))]
        data=[datas[i][randint(0,len(datas[i])-1)] for i in idx]
        for i in range(len(Y)):
            R.append((MIN_DIST-abs(Y[i][0]-y[0]))/MIN_DIST*(MIN_DIST-abs(Y[i][1]-y[1]))/MIN_DIST)
        return data,R
    print("BileanearMixupで生成したデータ",num,"個")

    # HACK:ラベルごとにデータを分類する。（クラス内法表記とかもっと簡潔な書き方がありそう）
    unique_label = np.unique(trainset.label, axis=0)
    label_data = []  # ラベルごとに分類したデータ。i番目のデータのラベルはuniue_label[i]
    idx_dict={}
    for i, label in enumerate(unique_label):
        idx_dict[tuple(label)]=i

    for target_label in unique_label:
        push_data = []
        for i, label in enumerate(trainset.label):
            if not all(label == target_label):continue
            push_data.append(trainset.data[i])
        label_data.append(push_data)
        
    data = []
    label = []
    rate =[]
    for i in range(len(trainset.data)):
        data.append([trainset.data[i]]*4)
        label.append(trainset.label[i])
        rate.append([1,0,0,0])
    for _ in range(num):
        Y=np.random.rand(2)*np.array([X_MAX,Y_MAX])
        X,r=BilinearMixup(y=Y,datas=label_data,idx_dict=idx_dict)
        data.append(X)
        label.append(Y)
        rate.append(r)
    
    data,label,rate=np.array(data),np.array(label),np.array(rate)
    print(np.shape(data))
    return data, label,rate

def MidMultiGenerator(trainset,num=1):
    
    def CalRate(X,Y,x1,x2,x3,y1,y2,y3):
        b=((Y-y3)*(x1-x3)-(X-x3)*(y1-y3))/(-(x2-x3)*(y1-y3)+(y2-y3)*(x1-x3))
        a=((Y-y3)*(x2-x3)-(X-x3)*(y2-y3))/(-(x1-x3)*(y2-y3)+(y1-y3)*(x2-x3))
        return a,b
            
        
    print("MultiMixで生成したデータ",num,"個")
    
    # HACK:ラベルごとにデータを分類する。（クラス内法表記とかもっと簡潔な書き方がありそう）
    unique_label = np.unique(trainset.label, axis=0)
    label_data = []  # ラベルごとに分類したデータ。i番目のデータのラベルはuniue_label[i]

    for target_label in unique_label:
        push_data = []
        for i, label in enumerate(trainset.label):
            if not all(label == target_label):continue
            push_data.append(trainset.data[i])
        label_data.append(push_data)
    
    data = []
    label = []
    rate=[]
    for _ in range(num):
        Y=np.random.rand(2)*np.array([X_MAX,Y_MAX])
        y1,y2,y3=-1,-1,-1
        y1_dist,y2_dist,y3_dist=INF,INF,INF
        for i,lab in enumerate(unique_label):
            dist=math.sqrt((lab[0]-Y[0])**2+(lab[1]-Y[1])**2)
            if y1_dist>dist:
                y3_dist=y2_dist
                y2_dist=y1_dist
                y1_dist=dist
                y3=y2
                y2=y1
                y1=i
            elif y2_dist>dist:
                y3_dist=y2_dist
                y2_dist=dist
                y3=y2
                y2=i
            elif y3_dist>dist:
                y3_dist=dist
                y3=i
        x1=label_data[y1][np.random.randint(0,len(label_data[y1]))]        
        x2=label_data[y2][np.random.randint(0,len(label_data[y2]))]        
        x3=label_data[y3][np.random.randint(0,len(label_data[y3]))]        
        y1=unique_label[y1]
        y2=unique_label[y2]
        y3=unique_label[y3]
        a,b=CalRate(X=Y[0],Y=Y[1],x1=y1[0],x2=y2[0],x3=y3[0],y1=y1[1],y2=y2[1],y3=y3[1])
        data.append([x1,x2,x3])
        label.append(Y)
        rate.append([a,b,1-a-b])
   
    
    for i in range(len(trainset.data)):
        data.append([trainset.data[i]]*3)
        label.append(trainset.label[i])
        rate.append([1,0,0])
    data,label,rate=np.array(data),np.array(label),np.array(rate)
    print(np.shape(data))
    return data, label,rate

#ミックスアップしたデータと訓練データを合わせる
def MixUpData(trainDataset=None,file_path=None,num=4,data_size=INF):
    data = []
    label = []
    if trainDataset:
        data, label = MixupGenerator(trainset=trainDataset,num=num)
    else:
        trainset = MyDataset(data_size=data_size,file_path=file_path)
        data, label = MixupGenerator(trainset=trainset,num=num)

    # data.extend(trainset.data)
    # data.extend(gen_data)
    # label.extend(trainset.label)
    # label.extend(gen_label)
    return data, label


def BilinearData(trainDataset=None,file_path=None,num=4,data_size=INF):
    data = []
    label = []
    if trainDataset:
        data, label = BilinearMixupGenerator(trainset=trainDataset,num=num)
    else:
        trainset = MyDataset(data_size=data_size,file_path=file_path)
        data, label = BilinearMixupGenerator(trainset=trainset,num=num)

    return data, label

#テストデータだけ
def TestData(file_path=TEST_DATA_FILE_NAME):
    testset = MyDataset(file_path=file_path)
    return testset.data,testset.label

#訓練データだけ
def TrainData(file_path=TRAIN_DATA_FILE_NAME):
    trainset=MyDataset(file_path=file_path)
    return trainset.data,trainset.label


if __name__ == '__main__':

    print("データ生成")
    #trainset = MyDataset(transforms.ToTensor())
    # trainset = MyDataset(CSV_PATH+TRAIN_DATA_FILE_NAME,out1vec=False)
    trainset = MyDataset(file_path=TRAIN_DATA_FILE_NAME,out1vec=False)

    gen_data, gen_label = MixupGenerator(trainset,num=10)

    # 生成されたラベルの表示
    unique_gen_label = np.unique(gen_label, axis=0)
    print(unique_gen_label)







##########################################
# 　　　＿＿_
# 　　 |＼ 　＼
# 　　 | |￣￣｜
# 　　 | | 先 ｜
# 　　 | | 祖 ｜
# 　　 | | 代 ｜
# 　 ＿| | 々 ｜
# 　|＼＼|＿＿亅＼
# 　 ＼匚二二二二]
############################################

# マンハッタン距離がMIN_DISTであるとき生成
# def MixupGeneratorBalance(trainset,num=1):
#     #NOTE:
#     #座標はランダムでMixupする（本来のmixup）
#     def Mixup2(x1, y1, x2, y2, alpha=0.1):
#         '''Returns mixed inputs, pairs of targets, and lambda'''
#         if alpha > 0:
#             lam = np.random.beta(alpha, alpha)
#         else:
#             lam = 1
#         X = []
#         Y = []
#         mixed_x = lam * x1 + (1 - lam) * x2
#         mixed_y = lam * y1 + (1 - lam) * y2
#         X.append(mixed_x)
#         Y.append(mixed_y)


#         return X, Y

#     #等間隔でmixupするデータを生成する
#     def Mixup(x1, y1, x2, y2, alpha=0.1):
#         '''Returns mixed inputs, pairs of targets, and lambda'''
#         X = []
#         Y = []
#         for lam in np.arange(alpha, 1, alpha):
#             mixed_x = lam * x1 + (1 - lam) * x2
#             mixed_y = lam * y1 + (1 - lam) * y2
#             X.append(mixed_x)
#             Y.append(mixed_y)

#         return X, Y
    
#     # HACK:ラベルごとにデータを分類する。（クラス内法表記とかもっと簡潔な書き方がありそう）
#     unique_label = np.unique(trainset.label, axis=0)
#     label_data = []  # ラベルごとに分類したデータ。i番目のデータのラベルはuniue_label[i]

#     for target_label in unique_label:
#         push_data = []
#         for i, label in enumerate(trainset.label):
#             if not all(label == target_label):continue
#             push_data.append(i)
#         label_data.append(push_data)
#     data = []
#     label = []
#     # ラベルの組み合わせごとに生成
#     # ラベル（xy座標）のマンハッタン距離が0より大きい最小距離であればミックスアップする
#     for i, label1 in enumerate(unique_label):
#         for j, label2 in enumerate(unique_label):
#             if i < j:
#                 break  # ラベルの組み合わせの重複を消す
#             if np.sum(np.abs(label1-label2)) > MIN_DIST+EPS:
#                 continue
#             idx_perm_1=torch.randperm(len(label_data[i]))
#             idx_perm_2=torch.randperm(len(label_data[j]))
            
#             #mixupするデータの組み合わせをランダムに決める
#             for _ in range(num):
#                 # data1_idx=label_data[i][idx_perm_1[k]]
#                 # data2_idx=label_data[j][idx_perm_2[k]]
#                 data1_idx=label_data[i][randint(0,len(idx_perm_1)-1)]
#                 data2_idx=label_data[j][randint(0,len(idx_perm_2)-1)]
                
#                 #CHANGE ミックスアップのアルゴリズムが異なる
#                 # mixup_data, mixup_label = Mixup(
#                 #     trainset.data[data1_idx], label1, trainset.data[data2_idx], label2, ALPHA)
#                 mixup_data, mixup_label = Mixup2(
#                     trainset.data[data1_idx], label1, trainset.data[data2_idx], label2, alpha=0.1)
#                 data.extend(mixup_data)
#                 label.extend(mixup_label)

#             #CHANGE 
#             #mixupするデータの組み合わせをpermの順で決める
#             # for k in range(min(len(idx_perm_1),len(idx_perm_2),num)):
#                 # data1_idx = np.random.randint(0, len(label_data[i]))
#                 # data1_idx = label_data[i][data1_idx]
#                 # data2_idx = np.random.randint(0, len(label_data[j]))
#                 # data2_idx = label_data[j][data2_idx]

#                 # mixup_data, mixup_label = my_mixup_data(
#                 #     trainset.data[data1_idx], label1, trainset.data[data2_idx], label2, ALPHA)
#                 # data.extend(mixup_data)
#                 # label.extend(mixup_label)
#     for i in range(len(trainset.data)):
#         data.append(trainset.data[i])
#         label.append(trainset.label[i])
    
#     return data, label
# #HACK　showグラフと合わせてクラスにしたい
# #短時間フーリエ変換部分とファイル読み込み部分を分けたい
# def FILE2DATA(files,out1vec):
#     datas = []
#     rates = []
#     for i in tqdm(range(len(files))):
#         amps = []
#         for file in files.iloc[i, :]:
#             rate, data = scipy.io.wavfile.read(file)
#             # 縦軸（振幅）の配列を作成   #16bitの音声ファイルのデータを-1から1に正規化
#             data = data / 32768
#             # フレーム長
#             fft_size = 1024
#             # フレームシフト長
#             hop_length = int(fft_size / 4)
#             # 短時間フーリエ変換
#             amplitude = np.abs(librosa.core.stft(
#                 data, n_fft=fft_size, hop_length=hop_length))
            
#             #デシベル変換する。
#             #WARNING 
#             # これを追加した後、結果がnanになる現象が発生する。
#             # pytorch内で何かが起きている可能性があり、直接の関係性は不明。
#             # amplitude = librosa.core.amplitude_to_db(amplitude)
#             #WARNING
            
#             # NOTE:out1vec=false:グラフ表示のために一次元配列の圧縮を解除
#             # if out1vec:
#             #     indata = np.array(amplitude).flatten()
#             #     amps.append(indata)
#             # else:
#             amps.append(np.array(amplitude))

#         # if out1vec:
#         #     indata=[]
#         #     for i in range(len(amps[0])):
#         #         indata2=[]
#         #         for j in range(len(amps[0][0])-SUB_DATA):
#         #             indata3=[]
#         #             for c in range(len(amps)):
#         #                 indata3.append(amps[c][i][j])
#         #             indata2.append(indata3)
#         #         indata.append(indata2)

#         #     datas.append(indata)
            
#         #     ampin = np.concatenate([np.array(amps).flatten()], 0)
#         #     datas.append(ampin)

#         #         indata2=[]
#         #         for j in range(len(amps[0][0])-SUB_DATA):
#         #             indata3=[]
#         #             for c in range(len(amps)):
#         #                 indata3.append(amps[c][i][j])
#         #             indata2.append(indata3)
#         #         indata.append(indata2)

#         #     datas.append(indata)
#         #     # datas.append(amps)

#         #HACK:NCHWをNHWCに書き替える。もっと簡潔な書き方がある
#         indata=[]
        
#         for i in range(len(amps[0])):
#             indata2=[]
#             for j in range(len(amps[0][0])-SUB_DATA):
#                 indata3=[]
#                 for c in range(len(amps)):
#                     indata3.append(amps[c][i][j])
#                 indata2.append(indata3)
#             indata.append(indata2)

        
#         # datas.append(amps)
        
#         if out1vec:
#           ampin = np.concatenate([np.array(indata).flatten()], 0)
#           datas.append(ampin)
#         else:
#           datas.append(indata)
        
#         rates.append(rate)

#     return np.array(datas), rates[0]
     # else:
#         #     #HACK:NCHWをNHWCに書き替える。もっと簡潔な書き方がある
#         #     indata=[]
            
#         #     for i in range(len(amps[0])):
#    
# class MiddleLayerMixupDataset(torch.utils.data.Dataset):

#     def __init__(self, trainDatasets=None,file_path=None, transform=None,out1vec=True,num=1,algo='mixup'):

#         self.data=[]
#         self.label=[]
#         self.rate=[]

        
#         if trainDatasets:
#             self.data,self.label,self.rate = MidMixupGenerator(trainDatasets,num=num)
#         else:
#             if file_path:
#                 self.data,self.label,self.rate = MidMixupGenerator(MyDataset(file_path=file_path,out1vec=out1vec),num=num)
#             else:
#                 self.data,self.label,self.rate =  MidMixupGenerator(MyDataset(file_path=TRAIN_DATA_FILE_NAME,out1vec=out1vec),num=num)
#         print("データ拡張後のサイズ",np.shape(self.data))
#         self.transform = transform
#         self.data_num = len(self.label)
        

#     def __len__(self):
#         return self.data_num

#     def __getitem__(self, idx):
#         out_data = self.data[idx]
#         out_label = self.label[idx]
#         out_r=self.rate[idx]
#         return out_data, out_label,out_r
