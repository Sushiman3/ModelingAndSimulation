import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures # 多項式回帰向けデータ変換用
from sklearn.metrics import mean_squared_error # 誤差和計算用ライブラリ

try:
    file=pd.read_csv('/data/data_for_ridge.csv') #データの読み込み
except FileNotFoundError:
    file=pd.read_csv('week6/data/data_for_ridge.csv')
X = np.array(file['x'], dtype=float).reshape(-1,1) #reshape(-1,1)はXを2次元配列の1列の縦ベクトルとする意味
y = np.array(file['y'], dtype=float)

poly = PolynomialFeatures(degree=3, include_bias=True) # 多項式の次数
X_poly = poly.fit_transform(X) # Xを多項式回帰用に変換
# Ridge Regression
alpha = [0,0.1,1]
I = np.eye(X_poly.shape[1]) # 単位行列
theta = []
for i in range(3):
    theta.append(np.linalg.inv(X_poly.T @ X_poly + alpha[i] * I) @ X_poly.T @ y) # 正規方程式によるパラメータ推定

# Plot Data and Regression line(グラフ作成)
x0 = np.linspace(1, 7).reshape(-1, 1) # 回帰曲線の描画用
color_list = ['red', 'orange', 'green'] # 色のリスト
plt.scatter(X, y, color='blue', label='data') # 散布図にデータをプロット(青)
for i in range(3):
    y_pred = poly.transform(x0) @ theta[i] # 回帰曲線の予測値
    plt.plot(x0, y_pred, label=f'Ridge Regression (alpha={alpha[i]})', color=color_list[i]) # 回帰曲線をプロット
plt.legend()
plt.xlabel('degree'); plt.ylabel('train/validation error')
plt.savefig('week6/images/ridge_regression.png') # グラフを保存
plt.show()