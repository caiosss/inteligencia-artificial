import numpy as np
import matplotlib.pyplot as plt

#1. Visualização dos dados por gráfico por espalhamento
#<<SE VER ESSE ERRO AO TENTAR RODAR>> "UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown plt.show", instala essa biblioteca PyQt5
#Variável (DEPENDENTE) y é o que eu quero prever, ela depedende de outras variáveis 
# e a variável x (INDEPENDENTE) é o que vai afetar na previsão de y. No caso nós queremos prever quais vai ser a velocidade do vento quando eu tiver determinada potência.
data = np.loadtxt('aerogerador.dat')
X = data[:,0] # independente --> medidade de velocidade do vento
y = data[:,1] # dependente --> potência gerada pelo aerogerador
plt.scatter(X,y,color='blue',label='dados')
plt.xlabel("Eixo x")
plt.ylabel("Eixo y")
plt.legend()
plt.title('Gráfico de espalhamento dos dados')
plt.show()
# Ao plotar o gráfico se observa que conform os valores do eixo X(velocidade do vento) aumenta, os valores do eixo y(potência do gerador.
# A linearidade também é confirmada pelo motivo que ao observar o gráfico gerado nós conseguimos observar uma crescente de x e y só há uma queda
# quando os valores chegam próximos de 12 para x e 500 para y


#2. Separação do vetor para variáveis regressoras e um vetor para a variável dependente