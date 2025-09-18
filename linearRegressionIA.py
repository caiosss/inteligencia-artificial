import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv

class LinearRegressionIA:
    def __init__(self, X, y, lambdas=None, rounds=500):
        if lambdas is None:
            self.lambdas = [0, 0.25, 0.5, 0.75, 1]
        else:
            self.lambdas = lambdas
            
        self.X = X
        self.y = y
        self.X_ = X[:, 1:] 
        self.N, self.p = self.X_.shape
        self.rounds = rounds
        
        self.results = {
            "media": {"sse": [], "mse": [], "r2": []},
            "mqo_sem_intercepto": {"sse": [], "mse": [], "r2": []},
            "mqo_com_intercepto": {"sse": [], "mse": [], "r2": []},
            "ridge": {l: {"sse": [], "mse": [], "r2": []} for l in self.lambdas}
        }
    
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
        return 1 - (sse / sst) if sst > 0 else 0

    def run_cross_validation(self):
        print(f"Iniciando validação cruzada com {self.rounds} rodadas...")
        for _ in range(self.rounds):
            idx = np.random.permutation(self.N)
            Xr, yr = self.X[idx, :], self.y[idx, :]
            Xr_ = self.X_[idx, :]

            k = int(self.N * 0.8)
            X_train, X_test = Xr[:k, :], Xr[k:, :]
            X_train_, X_test_ = Xr_[:k, :], Xr_[k:, :]
            y_train, y_test = yr[:k, :], yr[k:, :]

            self._evaluate_model("media", X_test, y_test, y_train=y_train)
            self._evaluate_model("mqo_sem_intercepto", X_test, y_test, X_train_=X_train_, y_train=y_train)
            self._evaluate_model("mqo_com_intercepto", X_test, y_test, X_train=X_train, y_train=y_train)
            
            for l in self.lambdas:
                self._evaluate_model("ridge", X_test, y_test, X_train=X_train, y_train=y_train, l=l)
        print("Validação cruzada concluída.")

    def _evaluate_model(self, model_name, X_test, y_test, **kwargs):
        beta_hat = None

        if model_name == "media":
            beta_hat = np.array([[np.mean(kwargs['y_train'])], [0]])
        elif model_name == "mqo_sem_intercepto":
            beta_ = pinv(kwargs['X_train_'].T @ kwargs['X_train_']) @ kwargs['X_train_'].T @ kwargs['y_train']
            beta_hat = np.vstack(([0], beta_)) 
        elif model_name == "mqo_com_intercepto":
            beta_hat = pinv(kwargs['X_train'].T @ kwargs['X_train']) @ kwargs['X_train'].T @ kwargs['y_train']
        elif model_name == "ridge":
            l = kwargs['l']
            X_train = kwargs['X_train']
            I = np.eye(X_train.shape[1])
            I[0, 0] = 0
            beta_hat = pinv(X_train.T @ X_train + l * I) @ X_train.T @ kwargs['y_train']

        y_pred = self._predicao(X_test, beta_hat)
        deviation = y_test - y_pred
        
        results_dict = self.results[model_name] if model_name != "ridge" else self.results["ridge"][kwargs['l']]
        results_dict["sse"].append(self._sse(deviation))
        results_dict["mse"].append(self._mse(deviation))
        results_dict["r2"].append(self._r2(y_test, deviation))

    def display_results(self):
        print("\n===== Resultados Médios =====")
        print(f"Modelo da média          -> MSE: {np.mean(self.results['media']['mse']):.4f}, R²: {np.mean(self.results['media']['r2']):.4f}")
        print(f"MQO sem intercepto       -> MSE: {np.mean(self.results['mqo_sem_intercepto']['mse']):.4f}, R²: {np.mean(self.results['mqo_sem_intercepto']['r2']):.4f}")
        print(f"MQO com intercepto       -> MSE: {np.mean(self.results['mqo_com_intercepto']['mse']):.4f}, R²: {np.mean(self.results['mqo_com_intercepto']['r2']):.4f}")
        print(f"SSE médio (modelo da média): {np.mean(self.results['media']['sse']):.4f}")

        print("\n--- Ridge Regression ---")
        for l in self.lambdas:
            print(f"Ridge (lambda={l:<4}) -> MSE: {np.mean(self.results['ridge'][l]['mse']):.4f}, R²: {np.mean(self.results['ridge'][l]['r2']):.4f}")

    def display_full_table(self):
        """
        Exibe uma tabela completa de resultados com média, desvio padrão, maior e menor valor.
        """
        print("\n" + "="*110)
        print("Tabela Completa de Métricas de Regressão (Validação Cruzada)")
        print("="*110)
        print(f"{'Modelo':30s} {'Métrica':>10s} {'Média':>15s} {'Desv. Padrão':>15s} {'Maior Valor':>15s} {'Menor Valor':>15s}")
        print("-" * 110)

        modelos = {
            "Modelo da Média": self.results["media"],
            "MQO sem Intercepto": self.results["mqo_sem_intercepto"],
            "MQO com Intercepto": self.results["mqo_com_intercepto"]
        }

        for nome, res in modelos.items():
            for metric in ["sse", "mse", "r2"]:
                if res[metric]:
                    media = np.mean(res[metric])
                    desvio = np.std(res[metric])
                    maior_val = np.max(res[metric])
                    menor_val = np.min(res[metric])
                    print(f"{nome:30s} {metric.upper():>10s} {media:15.4f} {desvio:15.4f} {maior_val:15.4f} {menor_val:15.4f}")
            print("-" * 110)

        print("--- Ridge Regression ---")
        for l in self.lambdas:
            for metric in ["sse", "mse", "r2"]:
                res = self.results["ridge"][l]
                if res[metric]:
                    media = np.mean(res[metric])
                    desvio = np.std(res[metric])
                    maior_val = np.max(res[metric])
                    menor_val = np.min(res[metric])
                    print(f"{f'Ridge (lambda={l})':30s} {metric.upper():>10s} {media:15.4f} {desvio:15.4f} {maior_val:15.4f} {menor_val:15.4f}")
            print("-" * 110)


    def plot_regression_lines(self):
        xx = np.linspace(0, self.X_[:, 0].max(), 200).reshape(-1, 1) 
        XX = np.hstack((np.ones((xx.shape[0], 1)), xx))

        plt.figure(figsize=(10, 6))
        plt.scatter(self.X_[:, 0], self.y[:, 0], s=20, alpha=0.6, label="Dados Originais")

        y_mean = np.mean(self.y)
        plt.plot(xx[:, 0], np.full_like(xx[:, 0], y_mean), color="red", linewidth=2, 
                 label="Reta da Média")

        for l in self.lambdas:
            I = np.eye(self.X.shape[1])
            I[0, 0] = 0
            beta_hat_ridge = pinv(self.X.T @ self.X + l * I) @ self.X.T @ self.y
            plt.plot(xx[:, 0], self._predicao(XX, beta_hat_ridge), label=f"Ridge lambda={l}")
    
        plt.title("Regressão Linear: Potência do Aerogerador vs. Velocidade do Vento")
        plt.xlabel("Velocidade do vento (m/s)")
        plt.ylabel("Potência gerada (kW)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
    
        plt.xlim(left=0) 
        plt.ylim(bottom=0)

        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
    
        plt.show()