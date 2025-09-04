import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv

#1. Visualização dos dados por gráfico de espalhamento
#<<SE VER ESSE ERRO AO TENTAR RODAR>> "UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown plt.show", instala a biblioteca PyQt5
# Variável (DEPENDENTE) y é o que eu quero prever, ela depende de outras variáveis 
# e a variável x (INDEPENDENTE) é o que vai afetar na previsão de y. 
# No caso nós queremos prever qual vai ser a velocidade do vento quando eu tiver determinada potência.

data = np.loadtxt('aerogerador.dat')
X_ = data[:,0].reshape(-1,1)  # independente --> velocidade do vento (em formato coluna)
y  = data[:,1].reshape(-1,1)  # dependente --> potência gerada pelo aerogerador (coluna)

# aqui temos N (número de amostras) e p (número de variáveis independentes)
N, p = X_.shape 

# adicionando a coluna de 1s para o modelo com intercepto
X = np.hstack((np.ones((N,1)),X_))

# criação das listas para armazenar as medidas de desempenho
sse_list = []
mse_list = []
sst_list = []
r2_list  = []

def predicao(X,beta):
    return X@beta

def sse(desvio):
    return np.sum(desvio**2)

def mse(desvio):
    return np.mean(desvio**2)

def sst(y_teste):
    y_mean = np.mean(y_teste)
    return np.sum((y_teste-y_mean)**2)

def r2(y_teste, desvio):
    return 1 - (sse(desvio)/sst(y_teste))

rodadas = 500

# variáveis para guardar o último conjunto de treino/teste e os betas (gpt sugeriu)
X_treino_last = X_teste_last = None
y_treino_last = y_teste_last = None
beta_hat_last = beta_hat__last = beta_hat_media_last = None

for r in range(rodadas):
    # embaralhar o conjunto de dados
    idx = np.random.permutation(N)
    Xr_ = X_[idx,:]
    Xr  = X[idx,:]
    yr  = y[idx,:]

    # particionamento do conjunto de dados (80/20)
    k = int(N*0.8)
    X_treino, X_teste   = Xr[:k,:],  Xr[k:,:]
    X_treino_,X_teste_  = Xr_[:k,:], Xr_[k:,:]
    y_treino, y_teste   = yr[:k,:],  yr[k:,:]

    # treinamento dos modelos:
    # modelo baseado na média
    beta_hat_media = np.array([
        [np.mean(y_treino)], # intercepto = média do y_treino
        [0]                  # coeficiente da variável x
    ])

    # modelo baseado MQO (sem intercepto)
    beta_hat_ = np.linalg.pinv(X_treino_.T@X_treino_)@X_treino_.T@y_treino
    beta_hat_ = np.vstack((np.zeros((1,1)),beta_hat_)) # adiciona 0 no intercepto

    # modelo MQO tradicional (com intercepto)
    beta_hat = pinv(X_treino.T@X_treino)@X_treino.T@y_treino

    # teste de desempenho para cada modelo:
    # MQO (com intercepto)
    y_pred = predicao(X_teste,beta_hat)
    desvios = y_teste - y_pred
    sse_list.append(sse(desvios))
    mse_list.append(mse(desvios))
    sst_list.append(sst(y_teste))
    r2_list.append(r2(y_teste,desvios))

    # MQO (sem intercepto)
    y_pred = predicao(X_teste,beta_hat_)
    desvios = y_teste - y_pred
    sse_list.append(sse(desvios))
    mse_list.append(mse(desvios))
    sst_list.append(sst(y_teste))
    r2_list.append(r2(y_teste,desvios))

    # modelo baseado na média
    y_pred = predicao(X_teste,beta_hat_media)
    desvios = y_teste - y_pred
    sse_list.append(sse(desvios))
    mse_list.append(mse(desvios))
    sst_list.append(sst(y_teste))
    r2_list.append(r2(y_teste,desvios))

    # salvando a última rodada para plotar depois do laço (gpt sugeriu)
    X_treino_last, X_teste_last = X_treino, X_teste
    y_treino_last, y_teste_last = y_treino, y_teste
    beta_hat_last = beta_hat
    beta_hat__last = beta_hat_
    beta_hat_media_last = beta_hat_media


print(sse_list)
print(mse_list)
print(sst_list)
print(r2_list)


plt.scatter(X_,y,color='blue',label='dados')
plt.xlabel("Eixo x")
plt.ylabel("Eixo y")
plt.legend()
plt.title('Gráfico de espalhamento dos dados')
plt.show()

# Ao plotar o gráfico se observa que conforme os valores do eixo X(velocidade do vento) aumenta, 
# os valores do eixo y(potência do gerador) também aumentam. 
# A linearidade também é confirmada pelo motivo que ao observar o gráfico gerado nós conseguimos observar uma crescente de x e y 
# só há uma queda quando os valores chegam próximos de 12 para x e 500 para y.









## ==== A PLOTAGEM A SEGUIR FOI FEITA PELO GPT, QUISER TESTAR COMENTA A DE CIMA E RODA ESSA ABAIXO ==== ##

## eu não entendi muito bem essa plotagem e nem sei se ele pede esses 3 gráficos, mas tá aí


# # Gráfico de dispersão treino vs teste
# plt.figure(1)
# plt.scatter(X_treino_last[:,1], y_treino_last[:,0], label='Treino', alpha=0.7)
# plt.scatter(X_teste_last[:,1],  y_teste_last[:,0],  label='Teste',  alpha=0.7)
# plt.xlabel("Velocidade do vento")
# plt.ylabel("Potência do aerogerador")
# plt.title("Dispersão dos dados (Treino e Teste)")
# plt.legend()

# # Gráfico das curvas dos três modelos
# xx = np.linspace(X_[:,0].min(), X_[:,0].max(), 200).reshape(-1,1)
# XX = np.hstack((np.ones((xx.shape[0],1)), xx))

# plt.figure(2)
# plt.scatter(X_[:,0], y[:,0], s=15, alpha=0.5, label='Dados')
# plt.plot(xx[:,0], predicao(XX, beta_hat_last),     label='MQO (com intercepto)')
# plt.plot(xx[:,0], predicao(XX, beta_hat__last),    label='MQO (sem intercepto)')
# plt.plot(xx[:,0], predicao(XX, beta_hat_media_last), label='Modelo da média')
# plt.xlabel("Velocidade do vento")
# plt.ylabel("Potência do aerogerador")
# plt.title("Ajustes dos modelos")
# plt.legend()

# # Histograma dos resíduos (ε) para MQO com intercepto no conjunto de treino
# y_pred_treino = predicao(X_treino_last, beta_hat_last)
# eps = y_treino_last - y_pred_treino

# plt.figure(3)
# plt.hist(eps, bins=20, edgecolor='k')
# plt.title(r"Histograma de resíduos ($\varepsilon$) - Treino")

# plt.show()