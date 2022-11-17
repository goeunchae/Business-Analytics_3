# Import libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

def PCA_plotting_3d():
    names = ['data_1', 'data_2', 'data_3']

    for name in names:
        data = pd.read_csv(f'./data/{name}.csv'.format(name))
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        pca = PCA(n_components=3)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        result = pca.fit_transform(X)

        df_pca = pd.DataFrame(
            result, columns=['component 0', 'component 1', 'component 2'])
        df_pca['class'] = y

        fig2 = plt.figure(figsize=(15, 15))
        ax2 = fig2.add_subplot(111, projection='3d')

        ax2.set_xlabel('Principal Component 1', fontsize=15)
        ax2.set_ylabel('Principal Component 2', fontsize=15)
        ax2.set_zlabel('Principal Component 3', fontsize=15)
        ax2.set_title('3 Component PCA', fontsize=20)

        colors = ["#7fc97f", "#beaed4"]
        for label, color in zip(y.unique(), colors):
            indicesToKeep = df_pca['class'] == label
            ax2.scatter(df_pca.loc[indicesToKeep, 'component 0'], df_pca.loc[indicesToKeep,
                        'component 1'], df_pca.loc[indicesToKeep, 'component 2'], c=color, s=30)

        ax2.legend(y.unique())
        ax2.grid()

        plt.savefig(f'./pics/{name}_3D.png'.format(name))
        df_pca.to_csv(f'./results/{name}_3D.csv'.format(name), index=False)

# PCA_plotting_3d()


def PCA_plotting_2d():
    names = ['data_1', 'data_2', 'data_3']

    for name in names:
        data = pd.read_csv(f'./data/{name}.csv'.format(name))
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        pca = PCA(n_components=2)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        result = pca.fit_transform(X)

        df_pca = pd.DataFrame(result, columns=['component 0', 'component 1'])
        df_pca['class'] = y

        fig2 = plt.figure(figsize=(15, 15))
        ax2 = fig2.add_subplot(1, 1, 1)

        ax2.set_xlabel('Principal Component 1', fontsize=15)
        ax2.set_ylabel('Principal Component 2', fontsize=15)
        ax2.set_title('2 Component PCA', fontsize=20)

        colors = ["#7fc97f", "#beaed4"]
        for label, color in zip(y.unique(), colors):
            indicesToKeep = df_pca['class'] == label
            ax2.scatter(df_pca.loc[indicesToKeep, 'component 0'],
                        df_pca.loc[indicesToKeep, 'component 1'], c=color, s=50)

        ax2.legend(y.unique())
        ax2.grid()

        plt.savefig(f'./pics/{name}_2D.png'.format(name))
        df_pca.to_csv(f'./results/{name}_2D.csv'.format(name), index=False)


# PCA_plotting_2d()

# Isolation Forest with Principal Components

def IF_dim():
    names = ['data_1', 'data_2', 'data_3']
    dims = ['2D', '3D']
    name_lst = []
    dim_lst = []
    acc_lst = []
    p_lst = []
    r_lst = []
    f_lst = []
    
    for name in names: 
        for dim in dims:
            data = pd.read_csv(f'./results/{name}_{dim}.csv'.format(name,dim))

            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]

            if_clf = IsolationForest(bootstrap=True, contamination=0.001, max_samples=10, n_estimators=1000, n_jobs=-1)

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

        result.to_csv('./results/IF_result.csv', index=False)

#IF_dim()