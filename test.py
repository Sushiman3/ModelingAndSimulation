import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures # 多項式回帰向けデータ変換用
from sklearn.metrics import mean_squared_error # 誤差和計算用ライブラリ

try:
    file=pd.read_csv('/data/train_data.csv') #トレーニングデータの読み込み
except FileNotFoundError:
    file=pd.read_csv('week6/data/train_data.csv')
X = np.array(file['x'], dtype=float).reshape(-1,1) #reshape(-1,1)はXを2次元配列の1列の縦ベクトルとする意味
y_train = np.array(file['y'], dtype=float)
poly=PolynomialFeatures(3) # 多項式の次数
X_poly=poly.fit_transform(X) # 多項式用にXの形式を変換

try:
    file=pd.read_csv('/data/validation_data.csv') #検証データの読み込み
except FileNotFoundError:
    file=pd.read_csv('week6/data/validation_data.csv')
y_valid = np.array(file['y_valid'], dtype=float)
#(Xはトレーニングデータ，検証データとも同じ(1～15)なのでy_validのみ読み込む)

# LinearRegression by sklearn
from sklearn.linear_model import LinearRegression # scikit-learnのLinearRegression(線形回帰)ライブラリを読み込み
lr = LinearRegression()
lr.fit(X_poly,y_train) # トレーニングデータに対して多項式回帰を実行
y_pred = lr.predict(X_poly) # 回帰曲線によるyの予測値をy_predに格納

print('train data: S=',mean_squared_error(y_pred, y_train)) # yの予測値とトレーニングデータとの二乗誤差和
print('validation data: S=',mean_squared_error(y_pred, y_valid)) # yの予測値と検証データとの二乗誤差和

# Plot Data and Regression line（グラフ作成）
plt.scatter(X, y_train, color='blue', label='training') # 散布図にトレーニングデータをプロット(青)
plt.scatter(X, y_valid, color='orange', label='validation') # 散布図に検証データをプロット（橙）
xline=np.linspace(1,15).reshape(-1,1) # 回帰曲線の描画用(以下3行)
xline_poly=poly.fit_transform(xline)
y0_pred = lr.predict(xline_poly)
plt.plot(xline, y0_pred, color='blue', label='regression') # 得られた回帰曲線を描画
plt.legend()
plt.xlabel('x'); plt.ylabel('y')
plt.show()