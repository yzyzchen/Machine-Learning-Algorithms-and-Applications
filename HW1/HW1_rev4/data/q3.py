import numpy as np
import math
import matplotlib.pyplot as plt

import linear_regression

import sys
import datetime

x_train = np.load('data/q3x.npy')
y_train = np.load('data/q3y.npy')
X_train = linear_regression.generate_polynomial_features(x_train, M=1)

K = 50
# Scatter plot of data
fig, ax = plt.subplots(figsize=(10, 7))

ax.set_title("Locally weighted linear regression")
ax.set(xlabel='x', ylabel='y')
ax.scatter(x_train, y_train, marker='.', c='black')

#
# The ordinary linear regression
#
w_linear = linear_regression.closed_form(X_train, y_train)

x_space = np.linspace(x_train.min(), x_train.max(), num=K)
ax.plot(x_space,
        x_space * w_linear[1] + w_linear[0],
        c='gray', linestyle='--', label='linear', linewidth=3)

#
# Locally-weighted linear regression
#
taus = [0.1, 0.3, 0.8, 2.0, 10.0]
colors = ['red', 'orange', 'green', 'blue', 'magenta']

for color, tau in zip(colors, taus):
    y_space = linear_regression.compute_y_space(X_train, x_train, y_train, x_space, tau)
    ax.plot(x_space, y_space,
            c=color, label=f'tau={tau}', linewidth=2)

ax.legend()
ax.grid()
plt.show()