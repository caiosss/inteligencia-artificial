import numpy as np
import matplotlib.pyplot as plt
from linearRegressionIA import LinearRegressionIA

data = np.loadtxt('aerogerador.dat')
X_ = data[:, 0].reshape(-1, 1)  # independente --> velocidade do vento
y = data[:, 1].reshape(-1, 1)  # dependente --> potência do aerogerador

# aqui temos N (número de amostras) e p (número de variáveis independentes)
N, p = X_.shape
X = np.hstack((np.ones((N, 1)), X_))  # adicionando intercepto

lambdas_ridge = [0, 0.5, 1, 10, 100]
linear_regression = LinearRegressionIA(X, y, lambdas=lambdas_ridge, rounds=500)
linear_regression.run_cross_validation()
linear_regression.display_full_table()
linear_regression.plot_regression_lines()