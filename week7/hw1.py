import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
  file = pd.read_csv('./data/class_data1.csv')
except FileNotFoundError:
  file = pd.read_csv('week7/data/class_data1.csv')


x = np.array(file['x'], dtype=float)
y = np.array(file['y'], dtype=float)
tn = np.array(file['label'], dtype=float)

# logstic regression by yourself
alpha = 0.005 # 変化量の調整パラメータ
w0 = 1.0 # w0の初期値
w1 = 1.0 # w1の初期値
w2 = 1.0 # w2の初期値

sum_e = 0.0
sum_e_pre = 0.0

for i in range(10000):
  f = w0 + w1*x + w2*y
  p = 1/(1+np.exp(-f)) # ***の部分を変更してください （ヒント：numpyでexp(x)はnp.exp(x)）
  pn = (p**tn)*((1-p)**(1-tn)) # ***の部分を変更してください
  sum_pn = np.prod(pn) # pnの総乗 P
  sum_e = -np.log(sum_pn) # 誤差関数 E=-lnP

  de_dw0 = (p - tn)
  de_dw1 = (p - tn)*x
  de_dw2 = (p - tn)*y

  sum_de_dw0 = np.sum(de_dw0) # dE/dw0
  sum_de_dw1 = np.sum(de_dw1) # dE/dw1
  sum_de_dw2 = np.sum(de_dw2) # dE/dw2

  grad_E = np.array([sum_de_dw0,sum_de_dw1,sum_de_dw2]) # gradient(勾配) E
  W = np.array([w0,w1,w2])
  w0_new,w1_new,w2_new = W + alpha*(-grad_E) # w0,w1,w2の更新
  w0,w1,w2=w0_new,w1_new,w2_new

  if(abs(sum_e_pre - sum_e) < 1e-5): # 更新による誤差関数の変化量が1e-5以下になったら終了
    break
  sum_e_pre = sum_e

print("final_step:"+str(i)+"  error:{:.3f}".format(sum_e_pre))

# Plot Data and Estimated Border
colors = ['blue' if i == 0 else 'orange' for i in tn]
plt.scatter(x, y, color=colors) # ラベル0と1のデータをそれぞれ青・橙色でプロット
plt.plot(x, ((-w1*x-w0)/w2), color='gray')
plt.xlabel('x'); plt.ylabel('y')
plt.text(1,6,"(w0,w1,w2)=({:.3f},{:.3f},{:.3f})".format(w0,w1,w2))
plt.show()
