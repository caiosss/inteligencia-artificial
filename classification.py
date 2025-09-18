import numpy as np
from classificatinIA import ClassificationIA

data_completo = np.loadtxt("EMGsDataset.csv", delimiter=',')
data_completo = data_completo.T 

nomes_classes = ["Neutro", "Sorriso", "Sobrancelhas Levantadas", "Surpreso", "Rabugento"]
cores = ["red", "blue", "green", "yellow", "cyan"]

classifier = ClassificationIA(data_completo, nomes_classes, cores)
classifier.plot_data()
classifier.run_monte_carlo(500) #<<SETAR A QUANTIDADE DE TESTES DESEJADOS, 500 DEMORAM MESMO!!>>