import importlib
import numpy as np
import als as als
import pandas as pd
import numpy.ma as ma
import timeit
from abc import ABCMeta, abstractmethod

importlib.reload(als)

class RecSysALS():
    
    __metaclass__ = ABCMeta
    
    def __init__(self, rank, lambda_=1e-6, ratings=None):
        self.rank = rank
        self.lambda_ = lambda_
        if ratings is not None:
            self.ratings = ratings
            self.num_of_known_ratings_per_user = (~self.ratings.isnull()).sum(axis=1)
            self.num_of_known_ratings_per_movie = (~self.ratings.isnull()).sum(axis=0)
    
    def set_ratings(self, ratings):
        self.ratings = ratings
        self.num_of_known_ratings_per_user = (~self.ratings.isnull()).sum(axis=1)
        self.num_of_known_ratings_per_movie = (~self.ratings.isnull()).sum(axis=0)
    
    def get_U(self):
        return pd.DataFrame(self.U, index = self.ratings.index)
    
    def get_V(self):
        return pd.DataFrame(self.V, columns = self.ratings.columns)
    
    @abstractmethod
    #def fit_model(self):
    def fit_model(self):
        pass
    
        
class als_RecSysALS(RecSysALS):
    
    def fit_model(self, ratings=None, max_iter=50, threshold=1e-5):
        X = self.ratings if ratings is None else ratings
        self.ratings = X # ratings é a matriz X com as avaliações
        self.U, self.V = als.als(X, self.rank, self.lambda_, max_iter, threshold)
        self.pred = pd.DataFrame(self.U.dot(self.V), index = X.index, columns = X.columns)
        self.error = ma.power(ma.masked_invalid(X-self.pred),2).sum()
        return self.pred, self.error

####################################################################################################################
'''
#Read Movielens Dataset Movie10Items
n_users=  8
n_movies= 10
top_users= False
X, genres, user_info = read_movielens_1M(n_movies, n_users, top_users)

known = X.count().sum() / (1.0*X.size)
print ("known: ", )

n_known_item = (~X.isnull()).sum(axis=0).sort_values()
n_known_user = (~X.isnull()).sum(axis=1).sort_values()

#Factorization parameters
rank = 1 # before 20
lambda_ = 1 # before 20 #Ridge regularizer parameter

#Initiate a recommender system of type ALS
RS = als_RecSysAntidoteData(rank,lambda_)

pred,error = RS.fit_model(X)
print(pred)
print ('RMSE:',np.sqrt(error/X.count().sum()))

V = RS.get_V()
U = RS.get_U()
'''
