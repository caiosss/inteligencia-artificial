import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv
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
linear_regression.display_results()
linear_regression.plot_regression_lines()

# def predicao(X, beta):
#     return X@beta


# def sse(desvio):
#     return np.sum(desvio**2)


# def mse(desvio):
#     return np.mean(desvio**2)


# def sst(y_teste):
#     y_mean = np.mean(y_teste)
#     return np.sum((y_teste-y_mean)**2)


# def r2(y_teste, desvio):
#     return 1 - (sse(desvio)/sst(y_teste))


# rodadas = 500
# lambdas = [0, 0.25, 0.5, 0.75, 1]  # hiperparâmetros do Ridge
# # <<OBSERVAÇÕES>> Para todos esses hiperparâmetros todas as curvas de ridge ficaram praticamente iguais, ou seja
# # mudar o hiperpaâmetro não muda o coeficiente, portanto só conseguimos enxegar a reta roxa no gráfico, as outras
# # ficaram embaixo

# resultados = {
#     "media": {"sse": [], "mse": [], "sst": [], "r2": []},
#     "mqo_sem_intercepto": {"sse": [], "mse": [], "sst": [], "r2": []},
#     "mqo_com_intercepto": {"sse": [], "mse": [], "sst": [], "r2": []},
#     "ridge": {l: {"sse": [], "mse": [], "sst": [], "r2": []} for l in lambdas}
# }

# for r in range(rodadas):
#     # embaralhar dados
#     idx = np.random.permutation(N)
#     Xr, yr = X[idx, :], y[idx, :]
#     Xr_,   = X_[idx, :],

#     # split 80/20
#     k = int(N*0.8)
#     X_treino, X_teste = Xr[:k, :], Xr[k:, :]
#     X_treino_, X_teste_ = Xr_[:k, :], Xr_[k:, :]
#     y_treino, y_teste = yr[:k, :], yr[k:, :]

#     # Modelo da média
#     beta_hat_media = np.array([
#         [np.mean(y_treino)],  # intercepto
#         [0]                  # coeficiente
#     ])
#     y_pred = predicao(X_teste, beta_hat_media)
#     desvios = y_teste - y_pred
#     resultados["media"]["sse"].append(sse(desvios))
#     resultados["media"]["mse"].append(mse(desvios))
#     resultados["media"]["sst"].append(sst(y_teste))
#     resultados["media"]["r2"].append(r2(y_teste, desvios))

#     # MQO sem intercepto
#     beta_hat_ = np.linalg.pinv(X_treino_.T@X_treino_)@X_treino_.T@y_treino
#     beta_hat_ = np.vstack((np.zeros((1, 1)), beta_hat_)
#                           )  # adiciona 0 no intercepto
#     y_pred = predicao(X_teste, beta_hat_)
#     desvios = y_teste - y_pred
#     resultados["mqo_sem_intercepto"]["sse"].append(sse(desvios))
#     resultados["mqo_sem_intercepto"]["mse"].append(mse(desvios))
#     resultados["mqo_sem_intercepto"]["sst"].append(sst(y_teste))
#     resultados["mqo_sem_intercepto"]["r2"].append(r2(y_teste, desvios))

#     # MQO com intercepto
#     beta_hat = pinv(X_treino.T@X_treino)@X_treino.T@y_treino
#     y_pred = predicao(X_teste, beta_hat)
#     desvios = y_teste - y_pred
#     resultados["mqo_com_intercepto"]["sse"].append(sse(desvios))
#     resultados["mqo_com_intercepto"]["mse"].append(mse(desvios))
#     resultados["mqo_com_intercepto"]["sst"].append(sst(y_teste))
#     resultados["mqo_com_intercepto"]["r2"].append(r2(y_teste, desvios))

#     # Ridge Regression (λ)
#     for l in lambdas:
#         I = np.eye(X_treino.shape[1])
#         I[0, 0] = 0  # não penaliza o intercepto
#         beta_hat_ridge = pinv(X_treino.T@X_treino + l *
#                               I) @ X_treino.T @ y_treino

#         y_pred = predicao(X_teste, beta_hat_ridge)
#         desvios = y_teste - y_pred
#         resultados["ridge"][l]["sse"].append(sse(desvios))
#         resultados["ridge"][l]["mse"].append(mse(desvios))
#         resultados["ridge"][l]["sst"].append(sst(y_teste))
#         resultados["ridge"][l]["r2"].append(r2(y_teste, desvios))

# print("\n===== Resultados Médios =====")
# print("Modelo da média -> MSE:",
#       np.mean(resultados["media"]["mse"]), "R²:", np.mean(resultados["media"]["r2"]))
# print("MQO sem intercepto -> MSE:",
#       np.mean(resultados["mqo_sem_intercepto"]["mse"]), "R²:", np.mean(resultados["mqo_sem_intercepto"]["r2"]))
# print("MQO com intercepto -> MSE:",
#       np.mean(resultados["mqo_com_intercepto"]["mse"]), "R²:", np.mean(resultados["mqo_com_intercepto"]["r2"]))

# for l in lambdas:
#     print(f"Ridge (λ={l}) -> MSE:", np.mean(
#         resultados["ridge"][l]["mse"]), "R²:", np.mean(resultados["ridge"][l]["r2"]))

# xx = np.linspace(X_[:, 0].min(), X_[:, 0].max(), 200).reshape(-1, 1)
# XX = np.hstack((np.ones((xx.shape[0], 1)), xx))

# plt.figure()
# plt.scatter(X_[:, 0], y[:, 0], s=15, alpha=0.5, label="Dados")
# for l in lambdas:
#     I = np.eye(X.shape[1])
#     I[0, 0] = 0
#     beta_hat_ridge = pinv(X.T@X + l*I) @ X.T @ y
#     plt.plot(xx[:, 0], predicao(XX, beta_hat_ridge), label=f"Ridge λ={l}")
# plt.xlabel("Velocidade do vento")
# plt.ylabel("Potência do aerogerador")
# plt.legend()
# plt.show()
