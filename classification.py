import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv

# 1.

data = np.loadtxt("EMGsDataset.csv", delimiter=',')
data = data.T

classes = np.unique(data[:,-1])
nomes_classes = ["Neutro","Sorriso","Sobrancelhas Levantadas",
                 "Surpreso","Rabugento"]
cores = ["red","blue","green","yellow","cyan"]

N = data.shape[0]  # número de amostras
C = 5              # número de classes
p = 2              # número de características (sensores)

X = np.empty((0,p))
Y = np.empty((0,C))

for i,classe in enumerate(classes):
    X_classe = data[data[:,-1]==classe,0:-1]

    y_rotulo = np.zeros((X_classe.shape[0], C))
    y_rotulo[:, i] = 1

    X = np.vstack((X, X_classe))
    Y = np.vstack((Y, y_rotulo))

    plt.scatter(X_classe[:,0],X_classe[:,1],
                c=cores[i],label=nomes_classes[i],edgecolors='k')
    
X_mqo = np.hstack((np.ones((N,1)), X))   # N x (p+1)

# Versões para Bayesianos
X_bayes = X.T   
Y_bayes = Y.T  

plt.xlabel("Sensor 1 (Corrugador do Supercílio)")
plt.ylabel("Sensor 2 (Zigomático Maior)")
plt.legend()
plt.show()


# 2.
beta_mqo = pinv(X_mqo.T @ X_mqo) @ X_mqo.T @ Y
Y_pred_mqo = X_mqo @ beta_mqo
y_pred_classes = np.argmax(Y_pred_mqo, axis=1)
acuracia_mqo = np.mean(y_pred_classes == np.argmax(Y, axis=1))
print("Acurácia MQO:", acuracia_mqo)


# 3.
def calc_media(X, Y):
    """médias por classe"""
    medias = []
    for c in range(C):
        Xc = X[Y[:,c]==1]
        medias.append(np.mean(Xc, axis=0))
    return np.array(medias)

def calc_covariancia(X, Y):
    """covariâncias por classe"""
    covs = []
    for c in range(C):
        Xc = X[Y[:,c]==1]
        covs.append(np.cov(Xc.T))
    return covs

# médias e covariâncias
medias = calc_media(X, Y)
covs = calc_covariancia(X, Y)

# densidade gaussiana multivariada
def gaussiana(x, mean, cov, eps=1e-6):
    d = len(mean)
    cov_reg = cov + eps*np.eye(d)   # regularização
    inv = np.linalg.inv(cov_reg)
    det = np.linalg.det(cov_reg)
    norm = 1/np.sqrt((2*np.pi)**d * det)
    diff = x-mean
    return norm * np.exp(-0.5 * diff @ inv @ diff.T)


# Tradicional (covariância livre por classe)
preds = []
for x in X:
    probs = [gaussiana(x, medias[c], covs[c]) for c in range(C)]
    preds.append(np.argmax(probs))
acuracia_qda = np.mean(preds == np.argmax(Y, axis=1))
print("Acurácia Gaussiano Tradicional:", acuracia_qda)

# Covariâncias iguais (média das covs)
cov_media = sum(covs)/C
preds = []
for x in X:
    probs = [gaussiana(x, medias[c], cov_media) for c in range(C)]
    preds.append(np.argmax(probs))
acuracia_lda = np.mean(preds == np.argmax(Y, axis=1))
print("Acurácia Covariâncias Iguais:", acuracia_lda)

# Matriz agregada (cov ponderada pelo tamanho da classe)
cov_agregada = np.zeros((p,p))
for c in range(C):
    Xc = X[Y[:,c]==1]
    cov_agregada += (len(Xc)/N) * covs[c]
preds = []
for x in X:
    probs = [gaussiana(x, medias[c], cov_agregada) for c in range(C)]
    preds.append(np.argmax(probs))
acuracia_agregada = np.mean(preds == np.argmax(Y, axis=1))
print("Acurácia Matriz Agregada:", acuracia_agregada)

# Regularizado (lambda combina identidade com matriz agregada)
lambdas = [0, 0.25, 0.5, 0.75, 1]
for lam in lambdas:
    cov_reg = (1-lam)*cov_agregada + lam*np.eye(p)
    preds = []
    for x in X:
        probs = [gaussiana(x, medias[c], cov_reg) for c in range(C)]
        preds.append(np.argmax(probs))
    acc = np.mean(preds == np.argmax(Y, axis=1))
    print(f"Acurácia Gaussiano Regularizado (lambda={lam}):", acc)

# Bayes Ingênuo (cov diagonal)
# preds = []
# for x in X:
#     probs = []                                   <<<ERRO REFERENTE A UMA DIVISÃO POR ZERO!!>>>
#     for c in range(C):
#         variancias = np.var(X[Y[:,c]==1], axis=0)
#         mean = medias[c]
#         prob = np.prod(1/np.sqrt(2*np.pi*variancias) * 
#                        np.exp(-(x-mean)**2/(2*variancias)))
#         probs.append(prob)
#     preds.append(np.argmax(probs))
# acuracia_nb = np.mean(preds == np.argmax(Y, axis=1))
# print("Acurácia Bayes Ingênuo:", acuracia_nb)
