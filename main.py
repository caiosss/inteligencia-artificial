import numpy as np

array_teste = np.array([[1,2,3],
                       [4,5,6],
                       [7,8,9]])

print(array_teste.shape)
print(array_teste.size)
print(array_teste[:,1]) # INDEXAÇÃO estou dizendo que não quero pegar nenhuma linha e somente a coluna 1
print(array_teste[1,:]) # INDEXAÇÃo pego só a linha 1 e nenhuma coluna