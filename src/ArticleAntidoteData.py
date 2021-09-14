import numpy as np
import pandas as pd
#from scipy.spatial.distance import pdist
#from scipy.special import comb
import RecSysALS
import RecSysKNN
import RecSysNMF
from RecSysExampleData20Items import RecSysExampleData20Items

class ArticleAntidoteData():
        
    def __init__(self, n_users, n_movies, top_users, top_movies, l, theta, k):
        self.n_users = n_users
        self.n_movies = n_movies
        self.top_users = top_users
        self.top_movies = top_movies
        self.l = l
        self.theta = theta
        self.k = k

    ###################################################################################################################
    # function to read the data
    def read_movieitems(self, n_users, n_movies, top_users, top_movies, data_dir):
        # get ratings
        df = pd.read_table('{}/ratings.dat'.format(data_dir),names=['UserID','MovieID','Rating','Timestamp'], sep='::', engine='python')

        # create a dataframe with movie IDs on the rows and user IDs on the columns
        ratings = df.pivot(index='MovieID', columns='UserID', values='Rating')
        
        movies = pd.read_table('{}/movies.dat'.format(data_dir), names=['MovieID', 'Title', 'Genres'], sep='::', engine='python')
                            
        user_info = pd.read_table('{}/users.dat'.format(data_dir), names=['UserID','Gender','Age','Occupation','Zip-code'], sep='::', engine='python')
        user_info = user_info.rename(index=user_info['UserID'])[['Gender','Age','Occupation','Zip-code']]
        
        # put movie titles as index on rows
        movieSeries = pd.Series(list(movies['Title']), index=movies['MovieID'])
        ratings = ratings.rename(index=movieSeries)
        
        # read movie genres
        movie_genres = pd.Series(list(movies['Genres']),index=movies['Title'])
        movie_genres = movie_genres.apply(lambda s:s.split('|'))

        if top_movies:
            # select the top n_movies with the highest number of ratings
            num_ratings = (~ratings.isnull()).sum(axis=1) # quantitative ratings for each movie: Movie 1: 4, Movie 2: 5, Movie 3: 2 ...
            rows = num_ratings.nlargest(n_movies) # quantitative ratings for each movie (n_movies) sorted: Movie 7: 6, Movie 2: 5, Movie 1: 4 ...
            ratings = ratings.loc[rows.index] # matrix[n_movies rows , original columns]; before [original rows x original columns]
        
        if top_users:
            # select the top n_users with the highest number of ratings
            num_ratings = (~ratings.isnull()).sum(axis=0) # quantitative ratings made by each user: User 1: 5, User 2: 5, User 3: 5, ...
            cols = num_ratings.nlargest(n_users) # quantitative evaluations by each user (n_users) sorted: User 1: 5, User 2: 5, User 3: 5, ...
            ratings = ratings[cols.index] # matrix [n_movies rows , original columns]; before [n_movies rows , original columns] (just updated the index)
        else:
            # select the first n_users from the matrix
            cols = ratings.columns[0:n_users]
            ratings = ratings[cols] # matrix [n_movies rows , n_users columns]; before [n_movies rows , original columns]

        ratings = ratings.T # transposed: matrix [n_users rows x n_movies columns];

        return ratings, movie_genres, user_info

    ###################################################################################################################
    # compute_X_est: 
    def  compute_X_est(self, X, algorithm='RecSysALS', data_dir="Data/Movie20Items"):
        if(algorithm == 'RecSysALS'):
            
            # factorization parameters
            rank = 1 # before 20
            lambda_ = 1 # before 20 - ridge regularizer parameter

            # initiate a recommender system of type ALS (Alternating Least Squares)
            RS = RecSysALS.als_RecSysALS(rank,lambda_)

            X_est,error = RS.fit_model(X)

        elif(algorithm == 'RecSysKNN'):
            RecSysKNN
        elif(algorithm == 'RecSysNMF'):
            RecSysNMF
        elif(algorithm == 'RecSysExampleAntidoteData20Items'):
            RS = RecSysExampleData20Items()
            X_est, movie_genres, user_info = RecSysExampleData20Items.read_movieitems(40, 20, False, False, data_dir)
            #X_est, movie_genres, user_info = RS.read_movieitems(40, 20, False, False, "recsys-antidote/data/Movie20Items")
        else:
            RecSysNMF
        return X_est  
        

#######################################################################################################################
class Polarization():
    
    def evaluate(self, X_est):
        #print("def evaluate(self, X_est):")
        #print("X_est")
        #print(X_est)
        return X_est.var(axis=0,ddof=0).mean()

    def gradient(self, X_est):
        """
        Returns the gradient of the divergence utility defined on the
        estimated ratings of the original users.
        The output is an n by d matrix which is flatten.
        """
        D = X_est - X_est.mean()
        G = D.values
        return  G

#######################################################################################################################
class IndividualLossVariance():
    
    def __init__(self, X, omega, axis):
        self.axis = axis
        self.omega = omega
        self.X = X.mask(~omega)
        self.omega_user = omega.sum(axis=axis)
        
    def get_losses(self, X_est):
        X = self.X
        X_est = X_est.mask(~self.omega)
        E = (X_est - X).pow(2)
        losses = E.mean(axis=self.axis)
        return losses
        
    def evaluate(self, X_est):
        losses = self.get_losses(X_est)
        var =  losses.values.var()
        return var

    def gradient(self, X_est):
        """
        Returns the gradient of the utility.
        The output is an n by d matrix which is flatten.
        """
        X = self.X
        X_est = X_est.mask(~self.omega)
        diff = X_est - X
        if self.axis == 0:
            diff = diff.T
            
        losses = self.get_losses(X_est)
        B = losses - losses.mean()
        C = B.divide(self.omega_user)
        D = diff.multiply(C,axis=0)
        G = D.fillna(0).values
        if self.axis == 0:
            G = G.T
        return  G

#######################################################################################################################
class GroupLossVariance():
    
    def __init__(self, X, omega, G, axis):
        self.X = X 
        self.omega = omega
        self.G = G
        self.axis = axis
        
        if self.axis == 0:
            self.X = self.X.T
            self.omega = self.omega.T
            
        self.group_id ={}
        for group in self.G: #G [user1, user2, user3, user4]
            for user in G[group]:
                self.group_id[user] = group
        
        self.omega_group = {}
        for group in self.G:
            self.omega_group[group] = (~self.X.mask(~self.omega).loc[self.G[group]].isnull()).sum().sum()
        
        omega_user = {}
        for user in self.X.index:
            omega_user[user] = self.omega_group[self.group_id[user]]
        self.omega_user = pd.Series(omega_user)
        
    def get_losses(self, X_est):
        if self.axis == 0:
            X_est = X_est.T
            
        X = self.X.mask(~self.omega)
        X_est = X_est.mask(~self.omega)
        E = (X_est - X).pow(2)
        if not E.shape == X.shape:
            print ('dimension error')
            return
        losses = {}
        for group in self.G:
            losses[group] = np.nanmean(E.loc[self.G[group]].values)
        losses = pd.Series(losses)
        return losses
        
    def evaluate(self, X_est):
        losses = self.get_losses(X_est)
        var =  losses.values.var()
        return var

    def gradient(self, X_est):
        """
        Returns the gradient of the utility.
        The output is an n by d matrix which is flatten.
        """
        group_losses = self.get_losses(X_est)
        #n_group = len(self.G)
        
        X = self.X.mask(~self.omega)
        if self.axis == 0:
            X_est = X_est.T
        
        X_est = X_est.mask(~self.omega)
        diff = X_est - X
        if not diff.shape == X.shape:
            print ('dimension error')
            return
        
        user_group_losses ={}
        for user in X.index:
            user_group_losses[user] = group_losses[self.group_id[user]]
        losses = pd.Series(user_group_losses)
        
        B = losses - group_losses.mean()
        C = B.divide(self.omega_user)
        #C = (4.0/n_group) * C
        D = diff.multiply(C,axis=0)
        G = D.fillna(0).values
        if self.axis == 0:
            G = G.T
        return  G