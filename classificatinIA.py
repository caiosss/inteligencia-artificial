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
        self.classes = np.unique(self.data[:, -1])
        self.N = self.data.shape[0]
        self.C = len(self.classes)
        self.p = self.data.shape[1] - 1
        
        # Mapeamento dos rótulos de 1-5 para 0-4
        self.mapeamento_rotulos = {cls: i for i, cls in enumerate(self.classes)}

    def plot_data(self):
        """
        Gera um gráfico de dispersão dos dados, colorindo cada classe.
        """
        X = self.data[:, 0:self.p]
        y = self.data[:, -1]
        
        plt.figure(figsize=(8, 6))
        for i, classe in enumerate(self.classes):
            X_classe = X[y == classe]
            plt.scatter(X_classe[:, 0], X_classe[:, 1],
                        c=self.cores[i], label=self.nomes_classes[i], edgecolors='k', alpha=0.7)
        
        plt.xlabel("Sensor 1 (Corrugador do Supercílio)")
        plt.ylabel("Sensor 2 (Zigomático Maior)")
        plt.title("Visualização do Dataset de Expressões Faciais")
        plt.legend()
        plt.grid(True)
        plt.show()

    def _calc_media(self, X_data, Y_one_hot):
        """Calcula as médias por classe para um subconjunto de dados."""
        medias = []
        for c in range(self.C):
            Xc = X_data[Y_one_hot[:, c] == 1]
            if Xc.size > 0:
                medias.append(np.mean(Xc, axis=0))
            else:
                medias.append(np.zeros(X_data.shape[1]))
        return np.array(medias)

    def _calc_covariancia(self, X_data, Y_one_hot):
        """Calcula as covariâncias por classe para um subconjunto de dados."""
        covs = []
        for c in range(self.C):
            Xc = X_data[Y_one_hot[:, c] == 1]
            if Xc.shape[0] > 1:
                covs.append(np.cov(Xc.T))
            else:
                covs.append(np.eye(self.p) * 1e-6)
        return covs

    def _calc_variancia(self, X_data, Y_one_hot):
        """Calcula as variâncias por classe para Naive Bayes."""
        variancias = []
        for c in range(self.C):
            Xc = X_data[Y_one_hot[:, c] == 1]
            if Xc.size > 0:
                variancias.append(np.var(Xc, axis=0))
            else:
                variancias.append(np.ones(self.p) * 1e-6)
        return np.array(variancias)

    def _gaussiana_pdf(self, x, mean, cov, eps=1e-6):
        """Calcula a densidade de probabilidade Gaussiana multivariada."""
        d = len(mean)
        cov_reg = cov + eps * np.eye(d)
        try:
            inv = np.linalg.inv(cov_reg)
            det = np.linalg.det(cov_reg)
            if det <= 0: return 0
            norm = 1 / np.sqrt((2 * np.pi)**d * det)
            diff = x - mean
            return norm * np.exp(-0.5 * diff @ inv @ diff.T)
        except np.linalg.LinAlgError:
            return 0
    
    def _naive_bayes_pdf(self, x, mean, var, eps=1e-6):
        """Calcula a densidade de probabilidade para Naive Bayes."""
        probs = 1.0
        for j in range(len(x)):
            sigma_sq = var[j] + eps
            probs *= (1 / np.sqrt(2 * np.pi * sigma_sq)) * np.exp(-0.5 * ((x[j] - mean[j])**2) / sigma_sq)
        return probs

    def run_monte_carlo(self, R, test_size=0.2):
        """
        Executa a validação por Monte Carlo para todos os modelos de classificação.
        """
        print(f"Iniciando simulação de Monte Carlo com R = {R} rodadas...")
        
        acuracias = {
            "MQO tradicional": [],
            "Classificador Gaussiano Tradicional": [],
            "Classificador Gaussiano (Cov. de todo cj. treino)": [],
            "Classificador Gaussiano (Cov. Agregada)": [],
            "Classificador de Bayes Ingênuo (Naive Bayes Classifier)": [],
        }
        
        lambdas_reg = [0, 0.25, 0.5, 0.75, 1]
        
        # Adiciona as chaves dos modelos regularizados ao dicionário
        for lam in lambdas_reg:
             # Troca 'λ' por 'lambda'
             acuracias[f"Classificador Gaussiano Regularizado (Friedman lambda={lam})"] = []
        
        rng = np.random.default_rng(12345)
        
        for r in range(R):
            # Adicionado para exibir o progresso
            print(f"Executando rodada {r+1}/{R}...")

            # Particionamento dos dados em 80/20
            idx = np.arange(self.N)
            rng.shuffle(idx)
            n_treino = int((1 - test_size) * self.N)
            idx_treino = idx[:n_treino]
            idx_teste = idx[n_treino:]

            X_treino = self.data[idx_treino, :-1]
            y_treino = self.data[idx_treino, -1]
            X_teste = self.data[idx_teste, :-1]
            y_teste = self.data[idx_teste, -1]

            # Mapeia os rótulos de 1-5 para 0-4
            y_treino_map = np.array([self.mapeamento_rotulos[label] for label in y_treino])
            y_teste_map = np.array([self.mapeamento_rotulos[label] for label in y_teste])

            # Converte rótulos para o formato one-hot-encoding
            Y_treino_one_hot = np.zeros((X_treino.shape[0], self.C))
            for i, label in enumerate(y_treino_map):
                Y_treino_one_hot[i, int(label)] = 1
            
            # --- MODELOS DE CLASSIFICAÇÃO ---

            # MQO Tradicional
            X_mqo_treino = np.hstack((np.ones((X_treino.shape[0], 1)), X_treino))
            X_mqo_teste = np.hstack((np.ones((X_teste.shape[0], 1)), X_teste))
            
            beta_mqo = pinv(X_mqo_treino.T @ X_mqo_treino) @ X_mqo_treino.T @ Y_treino_one_hot
            Y_pred_mqo = X_mqo_teste @ beta_mqo
            y_pred_mqo = np.argmax(Y_pred_mqo, axis=1)
            acuracias["MQO tradicional"].append(np.mean(y_pred_mqo == y_teste_map))
            
            # Classificadores Bayesianos
            medias_treino = self._calc_media(X_treino, Y_treino_one_hot)
            covs_treino = self._calc_covariancia(X_treino, Y_treino_one_hot)
            
            # Gaussiano Tradicional (QDA)
            preds_qda = [np.argmax([self._gaussiana_pdf(x, medias_treino[c], covs_treino[c]) for c in range(self.C)]) for x in X_teste]
            acuracias["Classificador Gaussiano Tradicional"].append(np.mean(np.array(preds_qda) == y_teste_map))
            
            # Naive Bayes
            variancias_treino = self._calc_variancia(X_treino, Y_treino_one_hot)
            preds_nb = [np.argmax([self._naive_bayes_pdf(x, medias_treino[c], variancias_treino[c]) for c in range(self.C)]) for x in X_teste]
            acuracias["Classificador de Bayes Ingênuo (Naive Bayes Classifier)"].append(np.mean(np.array(preds_nb) == y_teste_map))

            # Gaussiano com Covariâncias Iguais (LDA)
            cov_media = sum(covs_treino) / self.C
            preds_lda = [np.argmax([self._gaussiana_pdf(x, medias_treino[c], cov_media) for c in range(self.C)]) for x in X_teste]
            acuracias["Classificador Gaussiano (Cov. de todo cj. treino)"].append(np.mean(np.array(preds_lda) == y_teste_map))

            # Gaussiano com Matriz Agregada
            cov_agregada = np.zeros((self.p, self.p))
            for c in range(self.C):
                Xc = X_treino[Y_treino_one_hot[:, c] == 1]
                if Xc.shape[0] > 1:
                    cov_agregada += (Xc.shape[0] / X_treino.shape[0]) * np.cov(Xc.T)
                else:
                    cov_agregada += (Xc.shape[0] / X_treino.shape[0]) * np.eye(self.p) * 1e-6
            
            preds_agregada = [np.argmax([self._gaussiana_pdf(x, medias_treino[c], cov_agregada) for c in range(self.C)]) for x in X_teste]
            acuracias["Classificador Gaussiano (Cov. Agregada)"].append(np.mean(np.array(preds_agregada) == y_teste_map))
            
            # Gaussiano Regularizado (Friedman)
            for lam in lambdas_reg:
                cov_reg = (1 - lam) * cov_agregada + lam * np.eye(self.p)
                preds_reg = [np.argmax([self._gaussiana_pdf(x, medias_treino[c], cov_reg) for c in range(self.C)]) for x in X_teste]
                # Troca 'λ' por 'lambda'
                acuracias[f"Classificador Gaussiano Regularizado (Friedman lambda={lam})"].append(np.mean(np.array(preds_reg) == y_teste_map))

        # 6. Cálculo e exibição da tabela final
        print("\n" + "="*90)
        print(f"Resultados dos Classificadores (Validação Monte Carlo com {R} rodadas):")
        print("="*90)
        print(f"{'Modelo':60s} {'Média':>10s} {'Desv.':>10s} {'Maior':>10s} {'Menor':>10s}")
        print("-" * 90)

        for modelo, accs in acuracias.items():
            if accs:
                media = np.mean(accs)
                desvio = np.std(accs)
                maior_valor = np.max(accs)
                menor_valor = np.min(accs)
                print(f"{modelo:60s} {media:10.4f} {desvio:10.4f} {maior_valor:10.4f} {menor_valor:10.4f}")
            else:
                print(f"{modelo:60s} {'-':>10s} {'-':>10s} {'-':>10s} {'-':>10s}")

        print("-" * 90)