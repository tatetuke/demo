###
# 複数のチャンネルの音を別々のマイクに分けずに録音する
###
import struct
import pyaudio
import csv
import numpy as np
import librosa
import librosa.display
from datetime import datetime
import wave


chunk = 1024#録音する秒数に関わる
FORMAT = pyaudio.paInt16
RATE = 44100
RECORD_SECONDS = 0.25
DEVICE_INDEX = 1

INF=100000000

LABEL = 3



RECORD_NUM = 100

# 閾値
threshold = 0.05
WRITE_CSV_FILE = "TrainSet.csv"
SAVE_AUDIO_FOLDER = "./data/TrainingData"


# 入力チャネル
CHANNEL =3

#　音の読み込みを開始


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

# 音の読み込みを中断


def audiostop(audio, stream):
    stream.stop_stream()
    stream.close()
    audio.terminate()


# 複数のマイクデータから一つのマイクを抽出する
# data:音データ、channel:チャンネル数
def extract_mic(data, s,record_start=-1,record_end=INF):
    ret = []
    for i in range(len(data)):
        if i<record_start:continue
        if i>len(data)-record_end:break

        if i % (2*CHANNEL) != 2*s and i % (2*CHANNEL) != 2*s+1:
            continue
        ret.append(data[i])
    print(len(data))
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


if __name__ == '__main__':
    (audio, stream) = audiostart(CHANNEL)
    cnt = 0
    print("x,y:",LABEL)

    while True:
        try:
            # 音の最大値を求める
            data = stream.read(chunk)
            x = np.frombuffer(data, dtype="int16") / 32768.0

            # 音の最大値がある閾値(threshold)を超えたら録音開始
            if x.max() > threshold:
                record_start,record_endsub=max_vol(x)
                filename = datetime.today().strftime("%Y%m%d_%H%M%S")
                print(filename)
                # 音データを入れる配列を初期化
                all = []
                all.append(data)

                for i in range(0, int(RATE / chunk * RECORD_SECONDS)):
                    data = stream.read(chunk)
                    all.append(data)
                files = []
                all = b''.join(all)
                for i in range(CHANNEL):
                    x = extract_mic(all, i,record_start,record_endsub)

                    # 録音
                    # NOTE 順繰りに録音
                    filename2 = SAVE_AUDIO_FOLDER+"/"+datetime.today().strftime("%Y%m%d_%H%M%S") + \
                        "_"+str(cnt)+"_C"+str(i)+".wav"

                    files.append(filename2)
                    out = wave.open(filename2, 'w')
                    out.setnchannels(1)
                    # 1サンプル2byte=16bitのデータhttps://qiita.com/Dsuke-K/items/2ad4945a81644db1e9ff
                    out.setsampwidth(2)
                    out.setframerate(RATE)
                    out.writeframes(x)
                    out.close()

                with open(WRITE_CSV_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    written_data = [LABEL,-1]
                    for filen in files:
                        written_data.append(filen)
                    writer.writerows([written_data])

                cnt += 1
                if cnt == RECORD_NUM:
                    break
                print(cnt)


        except KeyboardInterrupt:
            break

    audiostop(audio, stream)
