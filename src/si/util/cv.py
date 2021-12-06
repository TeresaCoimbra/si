from .util import train_test_split
import numpy as np
import itertools


class CrossValidationScore:

    def __init__(self, model, dataset, **kwargs):
        self.model = model
        self.dataset = dataset
        self.cv = kwargs.get('cv', 3)
        self.split = kwargs.get('split', 0.8)
        self.train_scores = None
        self.test_scores = None
        self.ds = None

    def run(self):
        train_scores = []
        test_scores = []
        ds = []
        for _ in range(self.cv):
            train, test = train_test_split(self.dataset, self.split)
            ds.append((train,test))
            self.model.fit(train)
            if not self.score:
                train_scores.append(self.model.cost())
                test_scores.append(self.model.cost(test.X, test.Y))
            else:
                y_train = np.ma.apply_along_axis(self.model.predict, axis = 0, arr= train.X.T)
                train_scores.append(self.score(train.Y, y_train))
                y_test = np.ma.apply_along_axis(self.model.predict, axis = 0, arr= train.X.T)
                test_scores.append(self.score(test.Y, y_test))
        self.train_scores = train_scores 
        self.test_scores = test_scores
        self.ds = ds
        return train_scores, test_scores