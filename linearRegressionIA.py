import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv

class LinearRegressionIA:
    """
    Uma classe para realizar, avaliar e visualizar modelos de regressão linear,
    incluindo Média, Mínimos Quadrados Ordinários (MQO) e Ridge Regression.
    """
    def __init__(self, X, y, lambdas=None, rounds=500):
        """
        Inicializa a classe de análise de regressão.

        Args:
            X (np.ndarray): Matriz de características com a coluna de intercepto.
            y (np.ndarray): Vetor de variável dependente.
            lambdas (list, optional): Lista de hiperparâmetros para Ridge. Default é [0, 0.25, 0.5, 0.75, 1].
            rounds (int, optional): Número de rodadas para validação cruzada. Default é 500.
        """
        if lambdas is None:
            self.lambdas = [0, 0.25, 0.5, 0.75, 1]
        else:
            self.lambdas = lambdas
            
        self.X = X
        self.y = y
        self.X_ = X[:, 1:] # Matriz sem o intercepto
        self.N, self.p = self.X_.shape
        self.rounds = rounds
        
        # Estrutura para armazenar os resultados da validação cruzada
        self.results = {
            "media": {"sse": [], "mse": [], "r2": []},
            "mqo_sem_intercepto": {"sse": [], "mse": [], "r2": []},
            "mqo_com_intercepto": {"sse": [], "mse": [], "r2": []},
            "ridge": {l: {"sse": [], "mse": [], "r2": []} for l in self.lambdas}
        }
    
    # --- Métodos de Métricas ---
    def _predicao(self, X, beta):
        return X @ beta

    def _sse(self, deviation):
        return np.sum(deviation**2)

    def _mse(self, deviation):
        return np.mean(deviation**2)

    def _sst(self, y_test):
        y_mean = np.mean(y_test)
        return np.sum((y_test - y_mean)**2)

    def _r2(self, y_test, deviation):
        sse = self._sse(deviation)
        sst = self._sst(y_test)
        # Previne divisão por zero se a variância de y_test for nula
        return 1 - (sse / sst) if sst > 0 else 0

    def run_cross_validation(self):
        """
        Executa a validação cruzada para todos os modelos por um número definido de rodadas.
        """
        print(f"Iniciando validação cruzada com {self.rounds} rodadas...")
        for _ in range(self.rounds):
            # Embaralhar dados
            idx = np.random.permutation(self.N)
            Xr, yr = self.X[idx, :], self.y[idx, :]
            Xr_ = self.X_[idx, :]

            # Divisão 80/20 para treino e teste
            k = int(self.N * 0.8)
            X_train, X_test = Xr[:k, :], Xr[k:, :]
            X_train_, X_test_ = Xr_[:k, :], Xr_[k:, :]
            y_train, y_test = yr[:k, :], yr[k:, :]

            # Avaliar cada modelo
            self._evaluate_model("media", X_test, y_test, y_train=y_train)
            self._evaluate_model("mqo_sem_intercepto", X_test, y_test, X_train_=X_train_, y_train=y_train)
            self._evaluate_model("mqo_com_intercepto", X_test, y_test, X_train=X_train, y_train=y_train)
            
            for l in self.lambdas:
                self._evaluate_model("ridge", X_test, y_test, X_train=X_train, y_train=y_train, l=l)
        print("Validação cruzada concluída.")

    def _evaluate_model(self, model_name, X_test, y_test, **kwargs):
        """
        Calcula as predições e métricas para um modelo específico.
        """
        beta_hat = None
        # --- Treinamento ---
        if model_name == "media":
            beta_hat = np.array([[np.mean(kwargs['y_train'])], [0]])
        elif model_name == "mqo_sem_intercepto":
            beta_ = pinv(kwargs['X_train_'].T @ kwargs['X_train_']) @ kwargs['X_train_'].T @ kwargs['y_train']
            beta_hat = np.vstack(([0], beta_)) # adiciona 0 no intercepto
        elif model_name == "mqo_com_intercepto":
            beta_hat = pinv(kwargs['X_train'].T @ kwargs['X_train']) @ kwargs['X_train'].T @ kwargs['y_train']
        elif model_name == "ridge":
            l = kwargs['l']
            X_train = kwargs['X_train']
            I = np.eye(X_train.shape[1])
            I[0, 0] = 0 # não penaliza o intercepto
            beta_hat = pinv(X_train.T @ X_train + l * I) @ X_train.T @ kwargs['y_train']

        # --- Predição e Métricas ---
        y_pred = self._predicao(X_test, beta_hat)
        deviation = y_test - y_pred
        
        # Armazenar resultados
        results_dict = self.results[model_name] if model_name != "ridge" else self.results[model_name][kwargs['l']]
        results_dict["sse"].append(self._sse(deviation))
        results_dict["mse"].append(self._mse(deviation))
        results_dict["r2"].append(self._r2(y_test, deviation))

    def display_results(self):
        """
        Imprime os resultados médios (MSE e R²) da validação cruzada para cada modelo.
        """
        print("\n===== Resultados Médios =====")
        print(f"Modelo da média      -> MSE: {np.mean(self.results['media']['mse']):.4f}, R²: {np.mean(self.results['media']['r2']):.4f}")
        print(f"MQO sem intercepto -> MSE: {np.mean(self.results['mqo_sem_intercepto']['mse']):.4f}, R²: {np.mean(self.results['mqo_sem_intercepto']['r2']):.4f}")
        print(f"MQO com intercepto -> MSE: {np.mean(self.results['mqo_com_intercepto']['mse']):.4f}, R²: {np.mean(self.results['mqo_com_intercepto']['r2']):.4f}")
        
        print("\n--- Ridge Regression ---")
        for l in self.lambdas:
            print(f"Ridge (λ={l:<4}) -> MSE: {np.mean(self.results['ridge'][l]['mse']):.4f}, R²: {np.mean(self.results['ridge'][l]['r2']):.4f}")

    def plot_regression_lines(self):
        """
        Plota os dados originais e as retas de regressão treinadas com o dataset completo.
        O gráfico é ajustado para melhor visualização do intercepto.
        """
        xx = np.linspace(0, self.X_[:, 0].max(), 200).reshape(-1, 1) # Garante que o eixo X comece em 0
        XX = np.hstack((np.ones((xx.shape[0], 1)), xx))

        plt.figure(figsize=(10, 6))
        plt.scatter(self.X_[:, 0], self.y[:, 0], s=20, alpha=0.6, label="Dados Originais")

        # Treina os modelos no dataset completo para plotagem
        for l in self.lambdas:
            I = np.eye(self.X.shape[1])
            I[0, 0] = 0
            beta_hat_ridge = pinv(self.X.T @ self.X + l * I) @ self.X.T @ self.y
            plt.plot(xx[:, 0], self._predicao(XX, beta_hat_ridge), label=f"Ridge λ={l}")
        
        # Melhorias no plot
        plt.title("Regressão Linear: Potência do Aerogerador vs. Velocidade do Vento")
        plt.xlabel("Velocidade do vento (m/s)")
        plt.ylabel("Potência gerada (kW)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Ajusta os limites para garantir que o intercepto (x=0) seja visível
        plt.xlim(left=0) 
        plt.ylim(bottom=0)
        
        # Adiciona linhas de eixo para destacar o intercepto
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        
        plt.show()