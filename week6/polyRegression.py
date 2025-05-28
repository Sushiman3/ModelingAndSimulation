# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    data = pd.read_csv('week6/data/temp_ene.csv')
except FileNotFoundError:
    data = pd.read_csv('./data/temp_ene.csv')
# %%

x = np.array(data['x'],dtype=float).reshape(-1, 1)
y = np.array(data['y'],dtype=float).reshape(-1, 1)
print(x)

# %%

# Preprocess the data
from sklearn.preprocessing import add_dummy_feature, PolynomialFeatures
x_original = x.copy()
x = PolynomialFeatures(degree=1, include_bias=False).fit_transform(x)
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