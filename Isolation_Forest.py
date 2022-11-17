from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Isolation Forest From Scratch

# Selection a feature of the data


def select_feature(data):
    return random.choice(data.columns)

# Select a random value within the range


def select_value(data, feat):
    mini = data[feat].min()
    maxi = data[feat].max()
    return (maxi-mini)*np.random.random()+mini

# Split Data


def split_data(data, split_column, split_value):
    data_below = data[data[split_column] <= split_value]
    data_above = data[data[split_column] > split_value]

    return data_below, data_above

# Classification


def classify_data(data):

    label_column = data.values[:,]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

    index = counts_unique_classes.argmax()
    classification = unique_classes[index]

    return classification


# The Isolation Tree
def isolation_tree(data, counter=0, max_depth=50):
    if data.empty ==False: 
        # End Loop if max depth or isolated
        if (counter == max_depth) or data.shape[0] == 1:
            classification = classify_data(data)
            return classification
        else:
            # Counter
            counter += 1

            # Select feature
            split_column = select_feature(data)

            # Select value
            split_value = select_value(data, split_column)
            # Split data
            data_below, data_above = split_data(data, split_column, split_value)

            # instantiate sub-tree
            question = "{} <= {}".format(split_column, split_value)
            sub_tree = {question: []}

            # Recursive part
            below_answer = isolation_tree(data_below, counter, max_depth=max_depth)
            above_answer = isolation_tree(data_above, counter, max_depth=max_depth)

            if below_answer == above_answer:
                sub_tree = below_answer
            else:
                sub_tree[question].append(below_answer)
                sub_tree[question].append(above_answer)

        return sub_tree

# Isolation Forest


def isolation_forest(df, n_trees, max_depth, subspace):
    forest = []
    for i in range(n_trees):
        # Sample the subspace
        if subspace <= 1:
            df = df.sample(frac=subspace)
        else:
            df = df.sample(subspace)
        # Fit tree
        tree = isolation_tree(df, max_depth=max_depth)

        # Save tree to forest
        forest.append(tree)

    return forest

# Path Length


def pathLength(example, iTree, path=0, trace=False):
    # Initialize question and counter
    path = path+1
    question = list(iTree.keys())[0]
    feature_name, comparison_operator, value = question.split()

    # ask question
    if example[feature_name].values <= float(value):
        answer = iTree[question][0]
    else:
        answer = iTree[question][1]

    # base case
    if not isinstance(answer, dict):
        return path

    # recursive part
    else:
        residual_tree = answer
        return pathLength(example, residual_tree, path=path)

    return path

# Evaluate Distance


def evaluate_instance(instance, forest):
    paths = []
    for tree in forest:
        paths.append(pathLength(instance, tree))
    return paths

# C_factor


def c_factor(n):
    return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))

# Anomaly Score


def anomaly_score(data_point, forest, n):
    '''
    Anomaly Score

    Returns
    -------
    0.5 -- sample does not have any distinct anomaly
    0 -- Normal Instance
    1 -- An anomaly
    '''
    # Mean depth for an instance
    E = np.mean(evaluate_instance(data_point, forest))

    c = c_factor(n)

    return 2**-(E/c)

def IF_Scratch():
    names = ['data_1', 'data_2', 'data_3']
    acc_lst = []
    p_lst = []
    r_lst = []
    f_lst = []

    for name in names:
        data = pd.read_csv(f'./data/{name}.csv'.format(name))
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        iForest = isolation_forest(X, n_trees=20, max_depth=100, subspace=256)
        an = []
        for i in range(X.shape[0]):
            an.append(anomaly_score(X.iloc[[i]], iForest, 256))

        an = np.array(an)
        an[an >= 0.5] = 1
        an[an < 0.5] = 0

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

    result.to_csv('./results/scratch.csv', index=False)

#IF_Scratch()

# Isolation Forest with Sklearn and Gridsearch 

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

def IF_Search():
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
            
        model = IsolationForest(random_state=42)

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

        param_grid = {'n_estimators': [1000, 1500], 
                    'max_samples': [10], 
                    'contamination': ['auto', 0.001, 0.005, 0.01, 0.02, 0.03], 
                    'max_features': [10, 15], 
                    'bootstrap': [True], 
                    'n_jobs': [-1]}

        grid_search = GridSearchCV(model,param_grid,scoring="accuracy", refit=True,cv=10, return_train_score=True)

        grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        print(f"Best params: {best_params}")
        
        if_clf = IsolationForest(**best_params)
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

    result.to_csv('./results/grid_search.csv', index=False)


#IF_Search()


# Final Result with Best Hyperparameters
def IF_run():
    names = ['data_1', 'data_2', 'data_3']
    acc_lst = []
    p_lst = []
    r_lst = []
    f_lst = []
    for name in names:
        data = pd.read_csv(f'./data/{name}.csv'.format(name))
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
            
        if_clf =  IsolationForest(bootstrap=True, contamination=0.001, max_samples=10, n_estimators=1000, n_jobs=-1)
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

    result.to_csv('./results/IF.csv', index=False)
    
#IF_run()