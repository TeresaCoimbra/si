import numpy as np
from .model import Model
from si.util.util import l2_distance
from si.util.metrics import accuracy_score

class KNN():
    def __init__(self, num_n, classification = True):
        super(KNN).__init__()    # instanciar a flag para saber se foi feito o fit ao modelo ou não
        self.k_n = num_n
        self.classification = classification

    def fit(self, dataset):
        self.dataset = dataset
        self.is_fitted = True
    
    def get_neighbors(self, x):
        #idxs = np.apply_along_axis(self.dataset.X, axis=0, arr=self.dataset.X)
        distances = l2_distance(x, self.dataset.X) # calcular as distâncias de X a todos os outros pontos do dataset usando l2 - euclidian distance
        sorted_index = np.argsort(distances)      # ordenar para ter os índices que correspondem às melhores distâncias
        # ordenar os índices por ordem crescente de distância
        return sorted_index[:self.k_n]   # selecionar os knn

    def predict(self, x):
        assert self.is_fitted, "Model must be fitted before predict"
        neighbors = self.get_neighbors(x)
        values = self.dataset.Y[neighbors].tolist() # y_values= dataset.Y[máscara]
        # os valores vão servir para fazer a votação (retornar aquele que aparece mais vezes) - isto para problema de classificação
        if self.classification:
            prediction = max(set(values), key = values.count)
        else:                                       # problema de regressão
            prediction = sum(values)/len(values)
        return prediction

    def cost(self):
        # np.ma with mask
        y_pred = np.ma.apply_along_axis(self.predict,axis=0,arr=self.dataset.X.T)
        return accuracy_score(self.dataset.Y, y_pred)