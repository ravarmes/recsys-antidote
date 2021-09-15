from ArticleAntidoteData import ArticleAntidoteData
from ArticleAntidoteData import Polarization
from ArticleAntidoteData import IndividualLossVariance
from ArticleAntidoteData import GroupLossVariance


# reading data from a base with 20 movies and 40 users
Data_path = 'Data/Movie20Items'
n_users=  40
n_movies= 20
top_users = False # True: to use users with more ratings; False: otherwise
top_movies = False # True: to use movies with more ratings; False: otherwise

# reading data from 3883 movies and 6040 users 
#Data_path = 'Data/MovieLens-1M'
#n_users=  300
#n_movies= 1000
#top_users = True # True: to use users with more ratings; False: otherwise
#top_movies = False # True: to use movies with more ratings; False: otherwise

# recommendation algorithm
#algorithm = 'RecSysALS'
algorithm = 'RecSysExampleAntidoteData20Items' # this algorithm should only be used for a database with 40 users and 20 movies 'Data/Movie20Items'

# parameters for calculating fairness measures
l = 5
theta = 3
k = 3

article = ArticleAntidoteData(n_users, n_movies, top_users, top_movies, l, theta, k)

X, genres, user_info = article.read_movieitems(n_users, n_movies, top_users, top_movies, data_dir = Data_path) # returns matrix of ratings with n_users rows and n_moveis columns
omega = ~X.isnull() # matrix X with True in cells with evaluations and False in cells not rated

X_est = article.compute_X_est(X, algorithm) # RecSysALS or RecSysKNN or RecSysNMF or RecSysExampleAntidoteData20Items

#print("X")
#print(X)
#print("X_est")
#print(X_est)
#print("Omega")
#print(omega)

print("\n\n------------ SOCIAL OBJECTIVE FUNCTIONS ------------")

# To capture polarization, we seek to measure the extent to which the user ratings disagree
polarization = Polarization()
Rpol = polarization.evaluate(X_est)
print("Polarization (Rpol):", Rpol)

# Individual fairness. For each user i, the loss of user i, is  the mean squared estimation error over known ratings of user i
ilv = IndividualLossVariance(X, omega, 1) #axis = 1 (0 rows e 1 columns)
Rindv = ilv.evaluate(X_est)
print("Individual Loss Variance (Rindv):", Rindv)

# Group fairness. Let I be the set of all users/items and G = {G1 . . . ,Gg} be a partition of users/items into g groups
# G group: identifying the groups (NA: users grouped by number of ratings for available items)
# advantaged group: 5% users with the highest number of item ratings
# disadvantaged group: 95% users with the lowest number of item ratings
G = {1: [1,2], 2: [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]}
glv = GroupLossVariance(X, omega, G, 1) #axis = 1 (0 rows e 1 columns)
RgrpNA = glv.evaluate(X_est)
print("Group Loss Variance (Rgrp NA):", RgrpNA)