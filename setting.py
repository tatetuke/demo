#チューニング時に叩くマスのラベル。
CLASS_LABEL=0

#record.pyを実行して録るデータの数
RECORD_NUM=100

#デモで流れる楽器の効果音のファイル。左からラベル0,1,2,3に紐づけされている。
I_FILES=['audio/ベース2.wav','audio/ピアノ.wav','audio/ギター5.wav','audio/シンバル1.wav']

#デモでマイクが頻繁に誤反応する場合、値を上げる。(0~1の範囲、デフォルト=0.1)
THRESHOLD=0.1