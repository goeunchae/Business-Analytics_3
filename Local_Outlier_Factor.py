# Import libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from math import sqrt, inf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# LOF from scratch 
from sklearn.neighbors import NearestNeighbors

def LOF(X, k):

    knn = NearestNeighbors(n_neighbors=k)

    knn.fit(X)
    
    # Gather the kth nearest neighbor distance
    neighbors_and_distances = knn.kneighbors(X)
    knn_distances = neighbors_and_distances[0]
    neighbors = neighbors_and_distances[1]
    kth_distance = [x[-1] for x in knn_distances]
    
    local_reach_density = []
    for i in range(X.shape[0]):
        pt = X[i]
        sum_reachability = 0
        neighbor_distances = knn_distances[i]
        pt_neighbors = neighbors[i]
        for neighbor_distance, neighbor_index in zip(neighbor_distances, pt_neighbors):
            neighbors_kth_distance = kth_distance[neighbor_index]
            sum_reachability = sum_reachability + max([neighbor_distance, neighbors_kth_distance])
            
        avg_reachability = sum_reachability / k
        local_reach_density.append(1/avg_reachability)

    local_reach_density = np.array(local_reach_density)
    lofs = []
    for i in range(X.shape[0]):
        pt = X[i]
        avg_lrd = np.mean(local_reach_density[neighbors[i]])
        lofs.append(avg_lrd/local_reach_density[i])
        
    return lofs
        
        
def run_LOF_scratch():
    names = ['data_1', 'data_2', 'data_3']
    acc_lst = []
    p_lst = []
    r_lst = []
    f_lst = []
    
    for name in names:
        data = pd.read_csv(f'./data/{name}.csv'.format(name))
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1]
        
        lof = LOF(X, 10)
        lof = np.array(lof)
        an = np.where(lof>=1.5,1,0)
        
        acc = accuracy_score(y, an)
        p = precision_score(y, an)
        r = recall_score(y, an)
        f = f1_score(y, an)

        acc_lst.append(acc)
        p_lst.append(p)
        r_lst.append(r)
        f_lst.append(f)

        cf = confusion_matrix(y, an)
        print(cf)

    result = pd.DataFrame({'data': names, 'accuracy': acc_lst,
                        'precision_score': p_lst, 'recall_score': r_lst, 'f1_score': f_lst})

    result.to_csv('./results/scratch_LOF.csv', index=False)

        
#run_LOF_scratch()

# LOF with Sklearn and Gridsearch 
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

def LOF_Search():
    names = ['data_1', 'data_2', 'data_3']
    acc_lst = []
    p_lst = []
    r_lst = []
    f_lst = []
    for name in names:
        data = pd.read_csv(f'./data/{name}.csv'.format(name))
        X = data.iloc[:, :-1]
        X = X.sample(frac=0.2)
        y = data.iloc[:, -1]
        y = y.sample(frac=0.2)
            
        model = LocalOutlierFactor()

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

        param_grid = {'n_neighbors': [5, 10, 20, 30, 50], 
                    'p': [1, 2], 
                    'contamination': ['auto', 0.001, 0.005, 0.01, 0.02, 0.03], 
                    'n_jobs': [-1]}

        grid_search = GridSearchCV(model,param_grid,scoring="accuracy", refit=True,cv=10, return_train_score=True)

        grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        print(f"Best params: {best_params}")
        
        if_clf = LocalOutlierFactor(**best_params)
        if_clf.fit(X_train, y_train)

        y_pred = if_clf.fit_predict(X)
        an = np.where(y_pred<=-1,1,0)
        
        acc = accuracy_score(y, an)
        p = precision_score(y, an)
        r = recall_score(y, an)
        f = f1_score(y, an)

        acc_lst.append(acc)
        p_lst.append(p)
        r_lst.append(r)
        f_lst.append(f)

        cf = confusion_matrix(y, an)
        print(cf)

    result = pd.DataFrame({'data': names, 'accuracy': acc_lst,
                        'precision_score': p_lst, 'recall_score': r_lst, 'f1_score': f_lst})

    result.to_csv('./results/grid_search_LOF.csv', index=False)


#LOF_Search()

# Final Result with Best Hyperparameters
def LOF_run():
    names = ['data_1', 'data_2', 'data_3']
    acc_lst = []
    p_lst = []
    r_lst = []
    f_lst = []
    for name in names:
        data = pd.read_csv(f'./data/{name}.csv'.format(name))
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
            
        if_clf = LocalOutlierFactor(n_neighbors=5, p=1, contamination='auto', n_jobs=-1)
        y_pred = if_clf.fit_predict(X)
        an = np.where(y_pred<=-1,1,0)
        
        acc = accuracy_score(y, an)
        p = precision_score(y, an)
        r = recall_score(y, an)
        f = f1_score(y, an)

        acc_lst.append(acc)
        p_lst.append(p)
        r_lst.append(r)
        f_lst.append(f)

        cf = confusion_matrix(y, an)
        print(cf)

    result = pd.DataFrame({'data': names, 'accuracy': acc_lst,
                        'precision_score': p_lst, 'recall_score': r_lst, 'f1_score': f_lst})

    result.to_csv('./results/LOF.csv', index=False)
    
#LOF_run()