# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    train_data = pd.read_csv('week6/data/train_data.csv')
except FileNotFoundError:
    train_data = pd.read_csv('./data/train_data.csv')
# %%

x = np.array(train_data['temp'],dtype=float).reshape(-1, 1)
y = np.array(train_data['ene'],dtype=float).reshape(-1, 1)
print(x)

# %%

# Preprocess the data
from sklearn.preprocessing import add_dummy_feature
x_original = x.copy()
x = add_dummy_feature(x, value=1)
print(x)

# %%
# Normal Equation
best_theta = np.linalg.inv(x.T @ x) @ x.T @ y
print(f"Best theta: {best_theta}")
y_pred = x @ best_theta

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.plot(x_original, y_pred, "r-", label="Predictions")
plt.plot(x_original, y, "b.")

plt.xlabel("$x$")
plt.ylabel("$y$", rotation=0)
plt.grid()
plt.legend(loc="upper left")

plt.show()
# %%