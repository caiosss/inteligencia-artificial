import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv

class ClassificationIA:
    def __init__(self, data, nomes_classes, cores):
        """
        Inicializa a classe com o conjunto de dados e metadados.

        Args:
            data (np.ndarray): O conjunto de dados, com a última coluna sendo o rótulo da classe.
            nomes_classes (list): Uma lista com os nomes de cada classe.
            cores (list): Uma lista de cores para a plotagem de cada classe.
        """
        self.data = data
        self.nomes_classes = nomes_classes
        self.cores = cores
        
        # Extrai metadados importantes do dataset
        self.classes = np.unique(self.data[:, -1])
        self.N = self.data.shape[0]  # número de amostras
        self.C = len(self.classes)   # número de classes
        self.p = self.data.shape[1] - 1  # número de características (sensores)

        # Prepara as matrizes de características (X) e rótulos (Y)
        self._prepare_data()

    def _prepare_data(self):
        """
        Prepara as matrizes X (características) e Y (rótulos one-hot)
        a partir do conjunto de dados bruto.
        """
        self.X = np.empty((0, self.p))
        self.Y = np.empty((0, self.C))

        for i, classe in enumerate(self.classes):
            X_classe = self.data[self.data[:, -1] == classe, 0:-1]
            
            # Criação de rótulos no formato one-hot
            y_rotulo = np.zeros((X_classe.shape[0], self.C))
            y_rotulo[:, i] = 1

            self.X = np.vstack((self.X, X_classe))
            self.Y = np.vstack((self.Y, y_rotulo))
        
        # Prepara matriz para o modelo MQO (adiciona coluna de 1s)
        self.X_mqo = np.hstack((np.ones((self.N, 1)), self.X))
        
        # Versões para modelos Bayesianos (formato transposto)
        self.X_bayes = self.X.T
        self.Y_bayes = self.Y.T
        
        # Rótulos de classe verdadeiros (não one-hot) para cálculo de acurácia
        self.y_true = np.argmax(self.Y, axis=1)

    def plot_data(self):
        """
        Gera um gráfico de dispersão dos dados, colorindo cada classe.
        """
        for i, classe in enumerate(self.classes):
            X_classe = self.data[self.data[:, -1] == classe, 0:-1]
            plt.scatter(X_classe[:, 0], X_classe[:, 1],
                        c=self.cores[i], label=self.nomes_classes[i], edgecolors='k')
        
        plt.xlabel("Sensor 1 (Corrugador do Supercílio)")
        plt.ylabel("Sensor 2 (Zigomático Maior)")
        plt.title("Visualização do Dataset de Expressões Faciais")
        plt.legend()
        plt.grid(True)
        plt.show()

    def run_mqo_classification(self):
        """
        Treina e avalia o classificador de Mínimos Quadrados Ordinários (MQO).
        """
        beta_mqo = pinv(self.X_mqo.T @ self.X_mqo) @ self.X_mqo.T @ self.Y
        Y_pred_mqo = self.X_mqo @ beta_mqo
        y_pred_classes = np.argmax(Y_pred_mqo, axis=1)
        
        acuracia_mqo = np.mean(y_pred_classes == self.y_true)
        print(f"Acurácia MQO: {acuracia_mqo:.4f}")
        return acuracia_mqo

    def _calc_media_por_classe(self):
        """Calcula as médias de características para cada classe."""
        medias = [np.mean(self.X[self.Y[:, c] == 1], axis=0) for c in range(self.C)]
        return np.array(medias)

    def _calc_covariancia_por_classe(self):
        """Calcula as matrizes de covariância para cada classe."""
        covs = [np.cov(self.X[self.Y[:, c] == 1].T) for c in range(self.C)]
        return covs

    def _gaussiana(self, x, mean, cov, eps=1e-6):
        """
        Calcula a densidade de probabilidade de uma Gaussiana multivariada.
        Adiciona regularização para evitar matrizes singulares.
        """
        d = len(mean)
        cov_reg = cov + eps * np.eye(d)
        inv_cov = np.linalg.inv(cov_reg)
        det_cov = np.linalg.det(cov_reg)
        norm = 1.0 / np.sqrt(((2 * np.pi) ** d) * det_cov)
        diff = x - mean
        return norm * np.exp(-0.5 * diff @ inv_cov @ diff.T)

    def _predict_gaussian(self, medias, cov_matrix, is_qda=False):
        """
        Realiza predições usando um modelo Gaussiano com uma dada matriz de covariância.
        """
        preds = []
        for x in self.X:
            if is_qda: # QDA usa uma matriz de covariância por classe
                probs = [self._gaussiana(x, medias[c], cov_matrix[c]) for c in range(self.C)]
            else: # Outros modelos usam uma matriz de covariância compartilhada
                probs = [self._gaussiana(x, medias[c], cov_matrix) for c in range(self.C)]
            preds.append(np.argmax(probs))
        return np.array(preds)

    def run_gaussian_classifiers(self):
        """
        Executa e avalia vários tipos de classificadores Bayesianos Gaussianos.
        """
        print("\n--- Classificadores Gaussianos ---")
        medias = self._calc_media_por_classe()
        covs = self._calc_covariancia_por_classe()

        # 1. Tradicional (QDA - Análise Discriminante Quadrática)
        preds_qda = self._predict_gaussian(medias, covs, is_qda=True)
        acuracia_qda = np.mean(preds_qda == self.y_true)
        print(f"Acurácia Gaussiano Tradicional (QDA): {acuracia_qda:.4f}")

        # 2. Covariâncias Iguais (LDA - Análise Discriminante Linear)
        cov_media = sum(covs) / self.C
        preds_lda = self._predict_gaussian(medias, cov_media)
        acuracia_lda = np.mean(preds_lda == self.y_true)
        print(f"Acurácia Covariâncias Iguais (LDA): {acuracia_lda:.4f}")

        # 3. Matriz Agregada (Covariância ponderada pelo tamanho da classe)
        cov_agregada = np.zeros((self.p, self.p))
        for c in range(self.C):
            Xc = self.X[self.Y[:, c] == 1]
            cov_agregada += (len(Xc) / self.N) * covs[c]
        preds_agregada = self._predict_gaussian(medias, cov_agregada)
        acuracia_agregada = np.mean(preds_agregada == self.y_true)
        print(f"Acurácia Matriz Agregada: {acuracia_agregada:.4f}")
        
        # 4. Regularizado (combina identidade com matriz agregada)
        lambdas = [0, 0.25, 0.5, 0.75, 1]
        print("\n--- Regularização da Matriz Agregada ---")
        for lam in lambdas:
            cov_reg = (1 - lam) * cov_agregada + lam * np.eye(self.p)
            preds_reg = self._predict_gaussian(medias, cov_reg)
            acc = np.mean(preds_reg == self.y_true)
            print(f"Acurácia Gaussiano Regularizado (lambda={lam}): {acc:.4f}")

        # 5. Bayes Ingênuo (Naive Bayes - covariância diagonal)
        eps = 1e-6
        preds_nb = []
        for x in self.X:
            probs = []
            for c in range(self.C):
                Xc = self.X[self.Y[:, c] == 1]
                variancias = np.var(Xc, axis=0) + eps
                mean = medias[c]
                # Calcula a probabilidade produto das probabilidades de cada característica
                prob_features = 1 / np.sqrt(2 * np.pi * variancias) * np.exp(-(x - mean) ** 2 / (2 * variancias))
                prob = np.prod(prob_features)
                probs.append(prob)
            preds_nb.append(np.argmax(probs))
        
        acuracia_nb = np.mean(preds_nb == self.y_true)
        print(f"\nAcurácia Bayes Ingênuo (Naive Bayes): {acuracia_nb:.4f}")