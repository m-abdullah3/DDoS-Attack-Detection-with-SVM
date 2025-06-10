import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV

#Loading the dataset as a pandas frame
dataFrame=pd.read_csv("preprocessed_ddos.csv")

#Printing 1st five rows
print(dataFrame.head().to_string())

#seperating the features and the labels
X=dataFrame.drop("Label",axis=1)
Y=dataFrame["Label"]
#spliting the dataset into 10% validation set
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.10)

#Defining the Grid Search with the needed parameters for sigmoid
param_grid_sigmoid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['sigmoid']
}
#Starting the grid serach based on the defined parameters for sigmoid
grid_search_sigmoid = GridSearchCV(SVC(), param_grid_sigmoid, refit=False, verbose=2)

#Fitting the grid search to find the best result through cross validation
#k=5 folds by default
grid_search_sigmoid.fit(X_train, y_train)

#printing the best paramters
print("Best Parameters for Sigmoid Kernel: ", grid_search_sigmoid.best_params_)

#Defining the Grid Search with the needed parameters for poly
param_grid_poly = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'degree': [2, 3, 4],
    'kernel': ['poly']
}

#Starting the grid serach based on the defined parameters for poly
grid_search_poly = GridSearchCV(SVC(), param_grid_poly, refit=False, verbose=2)

#Fitting the grid search to find the best result through cross validation
#k=5 folds by default
grid_search_poly.fit(X_train, y_train)

#printing the best paramters
print("Best Parameters for Polynomial Kernel: ", grid_search_poly.best_params_)

#Defining the Grid Search with the needed parameters for linear
param_grid_linear = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear']
}

#Starting the grid serach based on the defined parameters
grid_search_linear = GridSearchCV(SVC(), param_grid_linear, refit=False, verbose=2)

#Fitting the grid search to find the best result through cross validation
#k=5 folds by default
grid_search_linear.fit(X_train, y_train)

#printing the best results
print("Best Parameters for Linear Kernel: ", grid_search_linear.best_params_)

