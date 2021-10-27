import numpy as np

class KMeans:

    def __init__(self, k: iter, n_iter):
        self.k = k
        self.max_iter = n_iter
        self.centroids = None
        #self.distance = l2_distance

    def fit(self, dataset):
        '''Escolher k centroides de forma aleatoria'''
        x = dataset.X
        self._min = np.min(x, axis=0)
        self._max = np.max(x, axis=0)

    def init_centroids(self, dataset):
        '''Inicialização dos centroides'''
        x = dataset.X
        self.centroids = np.array(
            [np.random.uniform(
                low=self._min[1], high= self._max[i], size=(self.k,)
            ) for i in range (x.shape[1])]).T

    def get_closest_centroid(self, x):
        '''retorna o id do centroide mais proximo'''
        dist = self.distance(x, self.centroids)
        closest_centroids_index = np.argmin(dist, axis = 0)
        return closest_centroids_index

    def transform(self,dataset):
        self.init_centroids(dataset)
        print(self.centroids)
        X = dataset.X
        changed = True
        count = 0
        old_idxs= np.zeros(x.shape[0])
        while changed or count < self.max_iter:
            idxs = np.apply_along_axis(self.get_closest_centroid(), axis=0, arr=X.T)
            cent = []
            for i in range(self.k):
                cent.append(np.mean(X[idxs == i], axis = 0))
                self.centroids = np.array(cent)
            count += 1
            changed = np.all(old_idxs==idxs)
            old_idxs = idxs
        return self.centroids, idxs


    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)