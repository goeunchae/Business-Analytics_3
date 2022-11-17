# Import libraries
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

# LOF With Principal Components
def LOF_dim():
    names = ['data_1', 'data_2', 'data_3']
    dims = ['2D', '3D']
    acc_lst = []
    p_lst = []
    r_lst = []
    f_lst = []
    name_lst = []
    dim_lst = []
    for name in names: 
        for dim in dims:
            data = pd.read_csv(f'./results/{name}_{dim}.csv'.format(name,dim))

            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]

            if_clf = LocalOutlierFactor(n_neighbors=5, p=1, contamination='auto', n_jobs=-1)

            y_pred = if_clf.fit_predict(X)
            an = np.where(y_pred <= -1, 1, 0)

            acc = accuracy_score(y, an)
            p = precision_score(y, an)
            r = recall_score(y, an)
            f = f1_score(y, an)

            acc_lst.append(acc)
            p_lst.append(p)
            r_lst.append(r)
            f_lst.append(f)
            name_lst.append(name)
            dim_lst.append(dim)
            cf = confusion_matrix(y, an)
            print(cf)

        result = pd.DataFrame({'data':name_lst, 'dim':dim_lst, 'accuracy': acc_lst,
                                'precision_score': p_lst, 'recall_score': r_lst, 'f1_score': f_lst})

        result.to_csv('./results/LOF_result.csv', index=False)

LOF_dim()