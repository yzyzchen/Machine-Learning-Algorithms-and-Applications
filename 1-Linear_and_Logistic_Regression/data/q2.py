from IPython.display import display_html, HTML

display_html(HTML('''
<style type="text/css">
  .instruction { background-color: yellow; font-weight:bold; padding: 3px; }
</style>
'''));

###################################################################
# Enter your first and last name, e.g. "John Doe"                 #
# for example,                                                    #
# __NAME__ = "Honglak Lee"                                        #
# __UNIQID__ = "honglak"                                          #
###################################################################
__NAME__ = "Yuzhou Chen"
__UNIQID__ = "yzc"
# raise NotImplementedError("TODO: Add your implementation here.")
###################################################################
#                        END OF YOUR CODE                         #
###################################################################

print(f"Your name and email: {__NAME__} <{__UNIQID__}@umich.edu>")
assert __NAME__ and __UNIQID__

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# This is the python module you are going to implement. See linear_regression.py
import linear_regression

import sys
import datetime

# Dump environment information
print("Current time:", datetime.datetime.now())
print(f"python: {sys.version}")
print(f"numpy: {np.__version__}")
print(f"matplotlib: {matplotlib.__version__}")

# Use high-resolution images for inline matplotlib possible whenever possible
# config InlineBackend.figure_format = 'retina'

# Default params for plot
plt.rcParams['figure.figsize'] = 10, 5
plt.rcParams['font.size'] = 20

x_train, y_train, x_test, y_test = linear_regression.load_data()
X_train = linear_regression.generate_polynomial_features(x_train, M=1)
X_test = linear_regression.generate_polynomial_features(x_test, M=1)

print(f"{X_train.shape=}")
print(f"{y_train.shape=}")
print(f"{X_test.shape=}")
print(f"{y_test.shape=}")

import time

eta = 0.01

tic = time.time()
w_gd, info_gd = linear_regression.batch_gradient_descent(X_train, y_train, eta=eta)
toc = time.time()
gd_time = toc - tic
print(f'GD version took {gd_time:.2f} seconds')

gd_test = linear_regression.compute_objective(X_test, y_test, w=w_gd)
print(f"GD Test objective = {gd_test:.4f}")

w_sgd, info_sgd = linear_regression.stochastic_gradient_descent(X_train, y_train, eta=eta)
toc = time.time()
sgd_time = toc - tic
print(f'SGD version took {sgd_time:.2f} seconds')

sgd_test = linear_regression.compute_objective(X_test, y_test, w=w_sgd)
print(f"SGD Test objective = {sgd_test:.4f}")

fig, ax = plt.subplots()
ax.plot(info_gd["train_objectives"], linewidth=2.0, marker='o', markersize=4, label='Batch GD')
ax.plot(info_sgd["train_objectives"], linewidth=2.0, marker='x', markersize=4, label='Stochastic GD')

# NOTE: It is always a good practice to include label and title for matplotlib plots.
ax.set_title("Batch GD v.s. SGD")
ax.set_ylabel("Training Objective (loss)")
ax.set_xlabel("Epoch")
ax.legend()
plt.show()

# X_train = linear_regression.generate_polynomial_features(x_train, M=9)
# w_closed = linear_regression.closed_form(X_train, y_train)
# print(f"w_closed = {w_closed}")

fig, ax = plt.subplots()
M = 9

train_rms_errors = []
test_rms_errors = []
for M_candidate in range(1, M+1):
    train_rms_error, test_rms_error = linear_regression.compute_rms_for_m(
                                      x_train, y_train, x_test, y_test, M_candidate)
    
    train_rms_errors.append(train_rms_error)
    test_rms_errors.append(test_rms_error)

ax.plot(np.arange(1, M + 1), train_rms_errors,
        label='Train', color='b', marker='o', linewidth=2.0)
ax.plot(np.arange(1, M + 1), test_rms_errors,
        label='Test', color='r', marker='o', linewidth=2.0)

ax.grid()
ax.legend()
ax.set(xlabel="M", ylabel="RMS Error")

plt.show()

X_train = linear_regression.generate_polynomial_features(x_train, M=9)
X_test = linear_regression.generate_polynomial_features(x_test, M=9)
w_closed_1 = linear_regression.closed_form(X_train, y_train, reg=1.0)
print(f"w_closed (lambda = 1) = {w_closed_1}")

w_closed_10 = linear_regression.closed_form(X_train, y_train, reg=10.0)
print(f"w_closed (lambda = 10) = {w_closed_10}")

assert np.any(w_closed_1 != w_closed_10), "It should have different number once your function properly handle the lambda value."

fig, ax = plt.subplots()

M = 9
lambdas = np.array([0] + [pow(10, x) for x in [-5, -4, -3, -2, -1, 0]])

train_rms_error_lambda = []
test_rms_error_lambda = []
for reg in lambdas:
    train_rms_error, test_rms_error = linear_regression.compute_rms_for_m(
                                      x_train, y_train, x_test, y_test, M, reg)
    train_rms_error_lambda.append(train_rms_error)
    test_rms_error_lambda.append(test_rms_error)

assert train_rms_error_lambda[-1] != train_rms_error_lambda[-2], "It should have different number if your compute_rms_for_m function properly handle the lambda value."
assert test_rms_error_lambda[-1] != test_rms_error_lambda[-2], "It should have different number if your compute_rms_for_m function properly handle the lambda value."

log_lambdas = np.log10(lambdas + np.array([1e-6, *np.zeros(6)]))
log_lambda_labels = ["$-\infty$", *log_lambdas[1:]]

ax.plot(log_lambdas, train_rms_error_lambda, c='blue', marker='o', label='Training')
ax.plot(log_lambdas, test_rms_error_lambda, c='red', marker='o', label='Testing')


ax.grid()
ax.set(xlabel=r"$\log_{10} \lambda$", ylabel="RMS Error")

log_lambdas = np.log10(lambdas + np.array([1e-6, *np.zeros(6)]))
log_lambda_labels = ["$-\infty$", *log_lambdas[1:]]
ax.set_xticks(log_lambdas, labels=log_lambda_labels)

plt.show()
