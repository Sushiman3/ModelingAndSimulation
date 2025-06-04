import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

# 課題1のプログラムを参考にオリジナルのパーセプトロン関数を作成
# (x,y,tn)(二次元データとそのラベル)を入力すると，データを分類する直線の定数w0,w1,w2を出力する関数
def my_perceptron(x, y, tn):
    alpha=0.005
    w0, w1, w2 = 1.0, 1.0, 1.0
    sum_e_pre = 0.0

    for i in range(10000):
        f = w0 + w1*x + w2*y # ***の部分を変更してください(課題1と同じ)
        p = 1/(1+np.exp(-f)) # ***の部分を変更してください(課題1と同じ)
        pn = (p**tn)*((1-p)**(1-tn)) # ***の部分を変更してください(課題1と同じ)
        sum_pn = np.prod(pn)
        sum_e = -np.log(sum_pn)

        de_dw0 = p - tn
        de_dw1 = (p - tn) * x
        de_dw2 = (p - tn) * y

        sum_de_dw0 = np.sum(de_dw0)
        sum_de_dw1 = np.sum(de_dw1)
        sum_de_dw2 = np.sum(de_dw2)

        grad_E = np.array([sum_de_dw0, sum_de_dw1, sum_de_dw2])
        W = np.array([w0, w1, w2])
        w0, w1, w2 = W + alpha * (-grad_E)

        if abs(sum_e_pre - sum_e) < 1e-5:
            break
        sum_e_pre = sum_e

    return w0, w1, w2

# Load data
try:
    file = pd.read_csv('./data/class_data2.csv')
except FileNotFoundError:
    file = pd.read_csv('week7/data/class_data2.csv')
x = np.array(file['x'], dtype=float)
y = np.array(file['y'], dtype=float)
tn_label = np.array(file['label'], dtype=float) #XOR(オリジナル)のデータラベル(0または1)

# Make labels
tn_or = np.logical_or(x, y).astype(int) #ORラベル(0または1)を作成
tn_and = np.logical_and(x, y).astype(int)
tn_nand = np.logical_not(tn_and).astype(int) #NANDラベル(0または1)を作成

# Train perceptrons
w0, w1, w2 = my_perceptron(x, y, tn_or) #ORラベルについての1段目のパーセプトロンを実行
w02, w12, w22 = my_perceptron(x, y, tn_nand) #NANDラベルについての1段目のパーセプトロンを実行

# Calculate outputs of the first two perceptrons
f1 = w0 + w1 * x + w2 * y
p1 = 1/(1+np.exp(-f1)) # ***の部分を変更してください(ORパーセプトロンの出力)
f2 = w02 + w12 * x + w22 * y
p2 = 1/(1+np.exp(-f2)) # ***の部分を変更してください(NANDパーセプトロンの出力)

# Train perceptron 3 using the outputs of perceptrons 1 and 2
# ORとNANDの1段目パーセプトロンの結果を入力として，2段目のパーセプトロンを実行
w03, w13, w23 = my_perceptron(p1, p2, tn_label)

#print weight matix for 1st layer (OR and NAND)
print("Weight matrix for 1st layer (OR and NAND):")
print(np.array([[w0, w1, w2], [w02, w12, w22]]))
# Print weight matrix for 2nd layer (XOR)
print("Weight matrix for 2nd layer (XOR):")
print(np.array([[w03, w13, w23]]))

# Plot Data
colors = ['blue' if i == 0 else 'orange' for i in tn_label]
X, Y = np.mgrid[-0.2:1.2:100j, -0.2:1.2:100j]
Z = w03+w13*(1/(1+np.exp(-(w0+w1*X+w2*Y))))+w23*(1/(1+np.exp(-(w02+w12*X+w22*Y))))
cont = plt.contourf(X,Y,Z, levels=100, alpha=0.5) # Zについての等高図をプロット
cont2 = plt.contour(X,Y,Z, levels=[0.0]) # Z=0 の線を表示
cont2.clabel(fmt='%1.3f', fontsize=16)
plt.scatter(x, y, color=colors) # ラベル0と1のデータをそれぞれ青・橙色でプロット
plt.xlabel('x'); plt.ylabel('y')
plt.colorbar(cont)
plt.show()

# ヒートマップデータをCSVに出力
with open('week7/heatmap_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['X', 'Y', 'Z'])
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            writer.writerow([X[i, j], Y[i, j], Z[i, j]])    