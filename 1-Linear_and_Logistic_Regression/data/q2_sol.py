import numpy as np
import math
import matplotlib.pyplot as plt

def batch_grad(x_train, y_train, x_test, y_test, M, lam, eta=1, max_iters=10000):
    X_train = generate_polynomial_features(x_train, M)
    X_test = generate_polynomial_features(x_test, M)
    w = np.zeros(M+1)
    N = len(x_train)
    train_error_list = []
    convergence_iters = []
    for iter in range(max_iters):
        y_pred_train = np.matmul(X_train, w)
        error = y_train - y_pred_train
        tmp = np.matmul(np.transpose(X_train), error)
        tmp = -tmp + lam * w
        g = tmp/N
        w = w - eta*g
        y_pred_train = np.matmul(X_train, w)
        squared_error = (y_train-y_pred_train)**2
        mean_squared_error = np.mean(squared_error)
        if mean_squared_error < 0.2:
            #print('iter', iter, 'convergence iter', len(convergence_iters))
            if len(convergence_iters)==0 or (iter == convergence_iters[-1]+1 and len(convergence_iters)<100):
                convergence_iters.append(iter)
            elif len(convergence_iters)<100:
                convergence_iters = [iter]
        train_error_list.append(mean_squared_error)
    y_pred_train = np.matmul(X_train, w)
    y_pred_test = np.matmul(X_test, w)
    train_squared_error = (y_train-y_pred_train)**2
    test_squared_error = (y_test-y_pred_test)**2
    train_rms_error = np.sqrt(np.mean(train_squared_error))
    test_rms_error = np.sqrt(np.mean(test_squared_error))
    if len(convergence_iters)==0:
        convergence_iter = np.nan
    else:
        convergence_iter = convergence_iters[-1]
    return w, train_error_list, train_rms_error, test_rms_error, convergence_iter  

def stochastic_grad(x_train, y_train, x_test, y_test, M, lam, eta=4e-2, max_iters=10000):
    X_train = generate_polynomial_features(x_train, M)
    X_test = generate_polynomial_features(x_test, M) 
    w = np.zeros(M+1)
    N = len(x_train)
    train_error_list = []
    convergence_iters = []
    for iter in range(max_iters):
        for i in range(N):
            x = X_train[i]
            y_pred = np.dot(x, w)
            g = - (y_train[i]-y_pred)*x + lam/N*w
            w = w - eta*g
        y_pred_train = np.matmul(X_train, w)
        squared_error = (y_train-y_pred_train)**2
        mean_squared_error = np.mean(squared_error)
        if mean_squared_error < 0.2:
            #print('iter', iter, 'convergence iter', len(convergence_iters))
            if len(convergence_iters)==0 or (iter == convergence_iters[-1]+1 and len(convergence_iters)<100):
                convergence_iters.append(iter)
            elif len(convergence_iters)<100:
                convergence_iters = [iter]
        train_error_list.append(mean_squared_error)
    y_pred_train = np.matmul(X_train, w)
    y_pred_test = np.matmul(X_test, w)
    train_squared_error = (y_train-y_pred_train)**2
    test_squared_error = (y_test-y_pred_test)**2
    train_rms_error = np.sqrt(np.mean(train_squared_error))
    test_rms_error = np.sqrt(np.mean(test_squared_error))
    if len(convergence_iters)==0:
        convergence_iter = np.nan
    else:
        convergence_iter = convergence_iters[-1]
    return w, train_error_list, train_rms_error, test_rms_error, convergence_iter

def closed_form_solve(x_train, y_train, x_test, y_test, M, lam):
    X_train = generate_polynomial_features(x_train, M)
    X_test = generate_polynomial_features(x_test, M)
    tmp = np.matmul(np.transpose(X_train), X_train)+lam*np.eye(X_train.shape[1])
    tmp = np.linalg.inv(tmp)
    tmp = np.matmul(tmp, np.transpose(X_train))
    w = np.matmul(tmp, y_train)
    y_pred_train = np.matmul(X_train, w)
    y_pred_test = np.matmul(X_test, w)
    train_squared_error = (y_train-y_pred_train)**2
    test_squared_error = (y_test-y_pred_test)**2
    train_rms_error = np.sqrt(np.mean(train_squared_error))
    test_rms_error = np.sqrt(np.mean(test_squared_error))
    return w, train_rms_error, test_rms_error

def generate_polynomial_features(x, M):
    N = len(x)
    phi = np.zeros((N, M+1))
    for m in range(M+1):
        phi[:,m] = np.power(x, m)
    return phi
    

def main():
    x_train = np.load('q2xTrain.npy')
    y_train = np.load('q2yTrain.npy')
    x_test = np.load('q2xTest.npy')
    y_test = np.load('q2yTest.npy')

    w, _, _ = closed_form_solve(x_train, y_train, x_test, y_test, 1, 0)
    # (a)
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'cyan', 'violet']
    for i, eta in enumerate([1e-2, 2e-2, 4e-2, 8e-2, 1.6e-1, 3.2e-1, 6.4e-1]): 
        print('Learning rate', eta)
        w, train_error_list, train_rms_error, test_rms_error, convergence_iter = batch_grad(x_train, y_train, x_test, y_test, M=1, lam=0, eta=eta)
        print('Coefficient generated by batch gradient method', w)
        print('Convergence iter', convergence_iter)
        print('MSE', train_error_list[-1])
        plt.plot(train_error_list, c=colors[i], label=r'$\eta=$%g'%eta)
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('error')
    plt.savefig('q2a_batch_gradient.png')
    plt.close()

    for i, eta in enumerate([1e-2, 2e-2, 4e-2, 8e-2, 1.6e-1, 3.2e-1, 6.4e-1]):
        print('Learning rate', eta)
        w, train_error_list, train_rms_error, test_rms_error, convergence_iter = stochastic_grad(x_train, y_train, x_test, y_test, M=1, lam=0, eta=eta)    
        print('Coefficient generated by stochastic gradient method', w)    
        print('Convergence iter', convergence_iter)
        print('MSE', train_error_list[-1])
        plt.plot(train_error_list, c=colors[i], label=r'$\eta=$%g'%eta)
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('error')
    plt.savefig('q2a_stochastic_gradient.png')
    plt.close()
    
    # (b)
    train_rms_error_degree = []
    test_rms_error_degree = []
    for M in range(1, 10):
        w, train_rms_error, test_rms_error = closed_form_solve(x_train, y_train, x_test, y_test, M=M, lam=0)
        train_rms_error_degree.append(train_rms_error)
        test_rms_error_degree.append(test_rms_error)
    plt.plot(np.arange(1,10), train_rms_error_degree, c='blue', marker='o', label='Training')
    plt.plot(np.arange(1,10), test_rms_error_degree, c='red', marker='o', label='Testing')
    plt.xlabel('M')
    plt.ylabel('RMS error')
    plt.legend()
    plt.savefig('q2b.png')
    plt.close()
   
    # (c)
    
    train_rms_error_lambda = []
    test_rms_error_lambda = []
    lambda_list = [0] + [pow(10, x) for x in range(-8, 1)]
    for lam in lambda_list:
        w, train_rms_error, test_rms_error = closed_form_solve(x_train, y_train, x_test, y_test, M=9, lam=lam)
        train_rms_error_lambda.append(train_rms_error)
        test_rms_error_lambda.append(test_rms_error)
    plt.plot(train_rms_error_lambda, c='blue', marker='o', label='Training')
    plt.plot(test_rms_error_lambda, c='red', marker='o', label='Testing')
    plt.xlabel(r'$\lambda$ index')        
    plt.ylabel('RMS error')
    plt.legend()
    plt.savefig('q2c_lambda.png')                    
    plt.close()
   
    lambda_list[0] += 1e-10 
    log_lambda_list = np.log(lambda_list)
    plt.plot(log_lambda_list, train_rms_error_lambda, c='blue', marker='o', label='Training')
    plt.plot(log_lambda_list, test_rms_error_lambda, c='red', marker='o', label='Testing')
    plt.xlabel(r'$\ln(\lambda)$')
    plt.ylabel('RMS error')
    plt.legend()
    plt.savefig('q2c_lnlambda.png')
    plt.close()

if __name__ == "__main__":
    main()
        
