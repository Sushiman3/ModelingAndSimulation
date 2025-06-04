# Backward Propagation in a simple neural network
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)

# Input dataset
try:
    df = pd.read_csv('./data/class_data2.csv')
except FileNotFoundError:
    df = pd.read_csv('week7/data/class_data2.csv')

# 入力Xの特徴量は2つ、インスタンスは4つ
# バイアス項を含めてXを拡張
X = df[['x', 'y']].values  # (4, 2)
y = df['label'].values.reshape(-1, 1)  # (4, 1)

# バイアス項を先頭に追加
X_ext = np.hstack([np.ones((X.shape[0], 1)), X])  # (4, 3)

# 重み初期化（バイアス含む）
np.random.seed(42)
#theta1 = np.array([[-2.21120958,  5.41357212,  5.41357212],
#                  [ 6.50817609, -4.21056015, -4.21056015]]).T  # (3, 2)
#theta2 = np.array([[-6.97713604,  4.79297965,  4.77275907]]).T  # (3, 1)
theta1 = np.random.randn(3, 2)*0.1  # (3, 2): 1層目（バイアス含む）
theta2 = np.random.randn(3, 1)*0.1  # (3, 1): 2層目（バイアス含む）

# 順伝播
z1 = X_ext @ theta1  # (4, 2)
a1 = sigmoid(z1)    # (4, 2)
a1_ext = np.hstack([np.ones((a1.shape[0], 1)), a1])  # (4, 3)
z2 = a1_ext @ theta2  # (4, 1)
y_pred = sigmoid(z2)  # (4, 1)

# ロス（Log Loss）
E = -np.mean(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))

def forwardpropagation(X_ext, theta1, theta2, sigmoid):
    z1 = X_ext @ theta1
    a1 = sigmoid(z1)
    a1_ext = np.hstack([np.ones((a1.shape[0], 1)), a1])
    z2 = a1_ext @ theta2
    y_pred = sigmoid(z2)
    return y_pred

def backpropagation(X_ext, y, theta1, theta2, sigmoid, sigmoid_derivative):
    # 順伝播
    z1 = X_ext @ theta1  # (N, 2)
    a1 = sigmoid(z1)    # (N, 2)
    a1_ext = np.hstack([np.ones((a1.shape[0], 1)), a1])  # (N, 3)
    z2 = a1_ext @ theta2  # (N, 1)
    y_pred = sigmoid(z2)  # (N, 1)

    # 逆伝播
    delta2 = (y_pred - y)  # (N, 1)
    dtheta2 = (a1_ext.T @ delta2) / X_ext.shape[0]  # (3, 1)
    delta1 = (delta2 @ theta2[1:].T) * sigmoid_derivative(a1)  # (N, 2)
    dtheta1 = (X_ext.T @ delta1) / X_ext.shape[0]  # (3, 2)
    return dtheta1, dtheta2

# 学習率とエポック数の設定
learning_rate = 0.5
n_epochs = 20000

# 学習経過のロスを記録
loss_history = []

for epoch in range(n_epochs):
    # 順伝播
    y_pred = forwardpropagation(X_ext, theta1, theta2, sigmoid)
    # ロス計算
    E = -np.mean(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))
    loss_history.append(E)
    # 逆伝播
    dtheta1, dtheta2 = backpropagation(X_ext, y, theta1, theta2, sigmoid, sigmoid_derivative)
    # パラメータ更新
    theta1 -= learning_rate * dtheta1
    theta2 -= learning_rate * dtheta2
    # 100エポックごとにロスを表示
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {E}")

# 最終結果の順伝播
z1 = X_ext @ theta1
a1 = sigmoid(z1)
a1_ext = np.hstack([np.ones((a1.shape[0], 1)), a1])
z2 = a1_ext @ theta2
y_pred = sigmoid(z2)
E = -np.mean(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))

print('最終順伝播出力 (y_pred):\n', y_pred)
print('最終ロス E:', E)
print('theta1 (1層目の勾配):\n', theta1)
print('theta2 (2層目の勾配):\n', theta2)

# ロスの推移をプロット
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()

# 連続的な出力を表示
x_new, y_new = np.mgrid[-0.2:1.2:100j, -0.2:1.2:100j]
X_new = np.column_stack([x_new.ravel(), y_new.ravel()])
X_new_ext = np.hstack([np.ones((X_new.shape[0], 1)), X_new])  # バイアス項を追加
y_new_pred = forwardpropagation(X_new_ext, theta1, theta2, sigmoid)
# 予測結果をプロット
plt.figure(figsize=(8, 6))
#plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='viridis', label='Data Points')
plt.scatter(X_new[:, 0], X_new[:, 1], c=y_new_pred.flatten(), cmap='coolwarm', alpha=0.5, label='Predicted Output')
plt.colorbar(label='Predicted Output')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Predicted Output Visualization')
plt.legend()
plt.show()
