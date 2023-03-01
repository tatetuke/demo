#データセットのファイル名
# TRAIN_DATA_FILE_NAME = "XYdata.csv"
# TEST_DATA_FILE_NAME = "XYdata_test.csv"
TRAIN_DATA_FILE_NAME = "train2.csv"
# TRAIN_DATA_FILE_NAME = "train2_8cm.csv"
# TEST_DATA_FILE_NAME = "test_random.csv"
TEST_DATA_FILE_NAME = "test_random2.csv"
# TEST_DATA_FILE_NAME = "test_grid_point.csv"
# TEST_DATA_FILE_NAME = "test_grid_point2.csv"

NET_PATH = './CNN.pth'

ALPHA = 0.25 # 細かい分け方
DIST=2  # 何センチ間隔でデータを取ったか
RATE = 44100

# 座標ラベルの最小距離
MIN_DIST = 4 #MixupとBilinearMixupに影響する4
MIN_DIST2 = 32 #MixupとBilinearMixupに影響する4

X_MAX=8
Y_MAX=8

# SUB_DATA=20
SUB_DATA=25




