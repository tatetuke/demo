import numpy as np
import scipy.io.wavfile
import torch
import torchvision.transforms as transforms
import scipy.io.wavfile
import librosa
import librosa.display
import torchvision.transforms as transforms
import pyaudio
import wave
import torch
import setting
import train as ftrain

MUSIC_FILES=['img/audio/a/ベース2.wav','img/audio/a/ピアノ.wav','img/audio/a/ギター5.wav','img/audio/a/シンバル1.wav']


FORMAT = pyaudio.paInt16
chunk = 1024
RATE = 44100
RECORD_SECONDS = 0.25
DEVICE_INDEX = 1
WRITE_CSV_FILE="fes.csv"
NET_PATH = "fes.pth"
threshold = 0.1#閾値
CHANNEL = 3#入力チャネル
INF=1000000000000000000
SUB_DATA=setting.SUB_DATA

BATCH_SIZE=ftrain.BATCH_SIZE
FILES=['AUDIO_C0.wav','AUDIO_C1.wav','AUDIO_C2.wav']

class MyDataset(torch.utils.data.Dataset):

    def __init__(self,data_size=INF,file_path=None, transform=None,out1vec=True):

        xy_data = -1
        audio_data = FILES
        # 音データ変換
        fft_data, _RATE = FILE2DATA(audio_data)
        self.transform = transform
        self.data_num = len(fft_data)
        self.data = []
        self.label = []
        for data in fft_data:
            self.data.append(data)
        self.label.append(np.array(xy_data))

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.label[idx]
        
        if self.transform:
            out_data = self.transform(out_data)

        return out_data, out_label

def audiostart(channel):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                        rate=RATE,
                        channels=channel,
                        input_device_index=DEVICE_INDEX,
                        input=True,
                        frames_per_buffer=chunk,

                        )
    return audio, stream



def audiostop(audio, stream):
    stream.stop_stream()
    stream.close()
    audio.terminate()

def extract_mic(data, s,record_start=-1,record_end=INF):
    ret = []
    for i in range(len(data)):
        if i<record_start:continue
        if i>len(data)-record_end:break

        if i % (2*CHANNEL) != 2*s and i % (2*CHANNEL) != 2*s+1:
            continue
        ret.append(data[i])
    return bytes(ret)

def max_vol(data):
    ret_i=0
    for i in range(len(data)):
        if data[i]>=threshold:
            ret_i=i
            break
    start=(ret_i//CHANNEL)*CHANNEL*2
    endsub=(len(data)-(ret_i//CHANNEL)*CHANNEL)*2
    return start,endsub

def FILE2DATA(files):
    datas = []
    rates = []
    amps = []
    for file in files:
        rate, data = scipy.io.wavfile.read(file)
        data = data / 32768
        fft_size = 1024
        hop_length = int(fft_size / 4)
        amplitude = np.abs(librosa.core.stft(
            data, n_fft=fft_size, hop_length=hop_length))
        amps.append(np.array(amplitude))
    
    #HACK:NCHWをNHWCに書き替える。もっと簡潔な書き方があるlinearprobing
    indata=[]
    for c in range(len(amps)):
        indata2=[]
        for i in range(len(amps[0])):
            indata3=[]
            for j in range(len(amps[0][0])-SUB_DATA):
                indata3.append(amps[c][i][j])
            indata2.append(indata3)
        indata.append(indata2)
    datas.append(indata)
    rates.append(rate)
    # indata=[]
    #     for i in range(len(amps[0])):
    #         indata2=[]
    #         for j in range(len(amps[0][0])-SUB_DATA):
    #             indata3=[]
    #             for c in range(len(amps)):
    #                 indata3.append(amps[c][i][j])
    #             indata2.append(indata3)
    #         indata.append(indata2)
    #     datas.append(indata)
    #     rates.append(rate)

    # return np.array(datas), rates[0]
    return np.array(datas), 1




if __name__ == '__main__':
    # load model
    net=ftrain.Net()
    
    print("Load NetWork")
    net.load_state_dict(torch.load(NET_PATH))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    # 入力待ち
    print("準備完了:マイクを接続して下さい ")
    t = ""
    while (t != "q"):
        t = input()

    # リアルタイムで録音して識別
    (audio, stream) = audiostart(CHANNEL)
    cnt = 0
    while True:
        try:
            # 音の最大値を求める
            rt_data = stream.read(chunk)
            x = np.frombuffer(rt_data, dtype="int16") / 32768.0
            # 音の最大値がある閾値(threshold)を超えたら録音開始
            if x.max() < threshold:continue
    
    
    #---------
    #Sampling
    #---------
    
    
            print("録音開始")
            # 音データを入れる配列を初期化
            all = []
            all.append(rt_data)
            record_start,record_end=max_vol(x)
            for i in range(0, int(RATE / chunk * RECORD_SECONDS)):
                rt_data = stream.read(chunk)
                all.append(rt_data)
            all = b''.join(all)
            files_now = []
            for i in range(CHANNEL):
                x = extract_mic(all, i,record_start,record_end)
                # データを保存
                filename2 = "./AUDIO_C"+str(i)+".wav"
                files_now.append(filename2)
                out = wave.open(filename2, 'w')
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(RATE)
                out.writeframes(x)
                out.close()

            con = 0
            for filename in files_now:
                rate, rt_data = scipy.io.wavfile.read(filename)
                rt_data = rt_data / 32768
                fft_size = 1024
                hop_length = int(fft_size / 4)
                amplitude = np.abs(librosa.core.stft(
                    rt_data, n_fft=fft_size, hop_length=hop_length))
  
    #--------------
    #show image
    #--------------
  
  
                # if (con == 0):
                #     log_power = librosa.core.amplitude_to_db(amplitude)
                #     librosa.display.specshow(
                #         log_power, sr=rate, hop_length=hop_length, x_axis='time', y_axis='hz', cmap='magma')
                #     plt.title('スペクトログラム', fontname="MS Gothic")
                #     plt.pause(1)
                #     plt.cla()
                #     con += 1
                    
    # ----------
    #  Test
    # ----------
            testset=MyDataset(transform=transforms.ToTensor())
            # testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
            #                             shuffle=False, num_workers=2,persistent_workers=True)
            # testloader = DataLoader(testset, batch_size=BATCH_SIZE,
            #                             shuffle=False)
            
            with torch.no_grad():
                # for data in testloader:
                    # inputs, labels = data[0].to(device),data[1].to(device)
                    # print(inputs)
                inputs=torch.tensor(np.array([testset.data[0],testset.data[0],testset.data[0],testset.data[0]]), dtype=torch.float64)

                

                #データがdouble型になっているのでfloat型に変換
                inputs=inputs.float()
                
                # labels=labels.float()
                
                
                # calculate outputs by running images through the network
                outputs = net(inputs)
                # the class with the highest energy is what we choose as prediction
                _, predicted=torch.max(outputs.data,1)
                
                # print("録音終了")
                # print("識別結果:", predicted)
                # impath = "img/"+"predImg/"+str(predicted[0])+".png"
                # im = cv2.imread(impath)
                # cv2.imshow('test', im)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
                print("音の再生開始")
                file = wave.open(MUSIC_FILES[predicted[0]], mode='rb')
                p = pyaudio.PyAudio() #pyaudioのインスタンス化
                streamPlay = p.open(
                    format = p.get_format_from_width(file.getsampwidth()),
                    channels = 1,
                    output_device_index=3,#3
                    rate = file.getframerate(),
                    output = True
                    )
                playSound = file.readframes(chunk)
                cnt=0
                while playSound:
                    #音の再生
                    streamPlay.write(playSound)
                    playSound = file.readframes(chunk)
                    cnt+=1
                    
                    if predicted[0]==2:
                        if cnt==50: break
                    # elif predicted[0]==3:
                    #     if cnt==60: break
                        
                print("再生終了")
                

        except KeyboardInterrupt:
            break

    audiostop(audio, stream)
