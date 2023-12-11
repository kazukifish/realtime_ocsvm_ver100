# The notation of this file conforms to the snake case
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
import cv2
from pyzbar.pyzbar import decode
from pyzbar.pyzbar import ZBarSymbol

# データ取得（実際に学習させるときは，ストレス負荷なしのデータを使用）
input_file = []
with open("分析したファイル名","r") as f:
    input_file = f.readlines()
x1 = np.empty(0)
x2 = np.empty(0)
for i in input_file:
    x1 = np.append(x1,float(i[6:11])) # (389,)配列
    x2 = np.append(x2,float(i[15:])) # (389,)配列

# データフレームの作成（angle and rc）
df = pd.DataFrame({"angle":x1,"rc":x2}) 
df_norm = df.query("0.90 < angle < 2.20 & 0.60 < rc < 1.10") # 正常値

# データに関しては要検討
X_train = df_norm.values # 学習用データ（360,2）の正常データのみ

# インスタンス化
ocsvm = OneClassSVM(kernel='rbf',gamma='auto',nu=1e-3,verbose=False)

# 実行部分
ocsvm.fit(X_train) # トレーニング

# 2要素計算用関数
def calc(recieve,count,mean,base):
    ag = recieve[count-1]-recieve[count-2]
    ag = np.arctan2(ag[0],ag[1])
    rc = mean/base
    return (ag,rc)

# 異常検知モニタリング関数
def video_capture(id):
    
    # カメラ関連
    font = cv2.FONT_HERSHEY_SIMPLEX
    cap = cv2.VideoCapture(id)
    
    # 異常検知関連
    qrlist = []
    c = 0
    
    # 検知用プロパティ
    cnt = 0
    mean_data = 0
    std_time = 25 # 基本QRコード表示間隔
    max_time = 30 # 最大QRコード表示間隔
    base_val = np.empty(0) # ベース値用numpy配列
    dataset = np.empty(0) # データ格納用numpy配列（n,2）の配列を生成
    agrc = np.empty(0) # 角度と変化率のセットを格納する配列
    
    while True:
        ret,frame = cap.read()
        if ret:
            value = decode(frame, symbols=[ZBarSymbol.QRCODE])
            if value:
                for qrcode in value:
                    x, y, w, h = qrcode.rect
                    dec = qrcode.data.decode('utf-8') # 'str'
                    
                    # dec_infを格納するlist
                    if not qrlist:
                        qrlist.append(dec)
                        c += 1
                    elif qrlist[c-1] == dec:
                        continue # passではNG
                    else:
                        qrlist.append(dec)
                        c += 1
                    dec_inf = qrlist[c-1] # 'str'
                    print('dec : ',dec_inf)
                    # qrlistを使用した異常検知
                    if cnt == 0:
                        data = dec_inf[11:17],dec_inf[17:23],dec_inf[23:29],dec_inf[29:35],dec_inf[35:41]
                        for j in data:
                            if j[0] == '0':
                                mean_data += float(j[1:])/100 # 1QRコードデータの合計値を格納
                            else:
                                mean_data += float(j[1:])/(-100)
                        # 次回ループ用の初期化とカウンタの更新
                        cnt += 1
                        base_val = np.append(base_val,(cnt,mean_data/5))
                        mean_data = mean_data/5
                        dataset = np.append(dataset,(cnt,mean_data)).reshape(cnt,2)
                        mean_data = 0
                        old_date = int(dec_inf[5:7])*3600+int(dec_inf[7:9])*60+int(dec_inf[9:11])
                    else:
                        new_date = int(dec_inf[5:7])*3600+int(dec_inf[7:9])*60+int(dec_inf[9:11])
                        if (new_date - old_date) > max_time:
                            loop = int(((new_date - old_date)//std_time)-1) # ロストデータの数（推定）
                            for _ in range(loop):
                                cnt += 1
                                # ここにロストデータが来た時の処理を記述する
                                dataset = np.append(dataset,(cnt,-1)).reshape(cnt,2)
                                # agとrcの算出
                                result = calc(recieve=dataset,count=cnt,mean=mean_data,base=base_val[1])
                                agrc = np.append(agrc,result).reshape(cnt-1,2) #角度と変化率の配列を更新（追加） 
                                # ocsvmとの適合
                                if ocsvm.predict([agrc[cnt-2]]) == np.array([1]):
                                    print('正常')
                                else:
                                    print('異常：',dec_inf)
                            data = dec_inf[11:17],dec_inf[17:23],dec_inf[23:29],dec_inf[29:35],dec_inf[35:41]
                            for j in data:
                                if j[0] == '0':
                                    mean_data += float(j[1:])/100
                                else:
                                    mean_data += float(j[1:])/(-100)
                            mean_data = mean_data/5 # 分析に用いる形に加工したQRコードデータ
                            # 次回ループ用の初期化とカウンタの更新
                            cnt += 1
                            dataset = np.append(dataset,(cnt,mean_data)).reshape(cnt,2)
                            # agとrcの算出
                            result = calc(recieve=dataset,count=cnt,mean=mean_data,base=base_val[1])
                            agrc = np.append(agrc,result).reshape(cnt-1,2) #角度と変化率の配列を更新（追加） 
                            # ocsvmとの適合
                            if ocsvm.predict([agrc[cnt-2]]) == np.array([1]):
                                print('正常')
                            else:
                                print('異常：',dec_inf)
                            mean_data = 0
                            old_date = new_date # 時刻データの更新
                        
                        # ロストデータがなかった場合（合格）   
                        else:
                            data = dec_inf[11:17],dec_inf[17:23],dec_inf[23:29],dec_inf[29:35],dec_inf[35:41]
                            for k in data:
                                if k[0] == '0':
                                    mean_data += float(k[1:])/100
                                else:
                                    mean_data += float(k[1:])/(-100)
                            mean_data = mean_data/5 # 分析に用いる形に加工したQRコードデータ
                            # カウンタの更新
                            cnt += 1
                            dataset = np.append(dataset,(cnt,mean_data)).reshape(cnt,2)
                            # agとrcの算出
                            result = calc(recieve=dataset,count=cnt,mean=mean_data,base=base_val[1])
                            agrc = np.append(agrc,result).reshape(cnt-1,2) #角度と変化率の配列を更新（追加） 
                            # ocsvmとの適合
                            if ocsvm.predict([agrc[cnt-2]]) == np.array([1]):
                                print('正常')
                            else:
                                print('異常：',dec_inf)
                            # 次回ループ用の初期化
                            mean_data = 0
                            old_date = new_date # 時刻データの更新
                    
                    # 可視化処理
                    frame = cv2.putText(frame, dec, (x, y-6), font, .3, (255, 0, 0), 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
                    
                    # モニタリングデータの書き込み
                    with open('書き込み先のファイル名', 'a') as f:
                        print(dec_inf, file=f)
        # 画像の出力
        cv2.imshow("frame", frame)
        
        # 終了操作
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()


video_capture(0)