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

try:
    file=pd.read_csv('/data/validation_data.csv') #検証データの読み込み
except FileNotFoundError:
    file=pd.read_csv('week6/data/validation_data.csv')
y_valid = np.array(file['y_valid'], dtype=float)
#(Xはトレーニングデータ，検証データとも同じ(1～15)なのでy_validのみ読み込む)
train_error = []
valid_error = []

for n in range(1,10):
    poly=PolynomialFeatures(degree=n, include_bias=True) # degree=nはn次多項式回帰
    X_poly = poly.fit_transform(X) # Xをn次多項式回帰用に変換
    theta = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y_train # 正規方程式によるパラメータ推定
    y_pred = X_poly @ theta # トレーニングデータに対する予測値
    
    train_error.append(mean_squared_error(y_pred, y_train)) # トレーニングデータの二乗誤差和
    valid_error.append(mean_squared_error(y_pred, y_valid)) # 検証データの二乗誤差和

    print(f'Polynomial degree: {n}') # 多項式の次数
    print(f'train data: S={mean_squared_error(y_pred, y_train)}') # yの予測値とトレーニングデータとの二乗誤差和
    print('validation data: S=',mean_squared_error(y_pred, y_valid)) # yの予測値と検証データとの二乗誤差和

# Plot Data and Regression line(グラフ作成)
x = range(1, 10) # 多項式の次数
plt.figure(figsize=(10, 5))
plt.scatter(x, train_error, color='blue', label='train error') # トレーニングデータの二乗誤差和をプロット(青)
plt.scatter(x, valid_error, color='orange', label='validation error') # 検証データの二乗誤差和をプロット（橙）

plt.legend()
plt.xlabel('degree'); plt.ylabel('train/validation error')
plt.show()