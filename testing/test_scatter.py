# from pandas.tools.plotting import scatter_matrix
from autoregression.galgraphs import plot_scatter_matrix
import pandas as pd
from sklearn import datasets
# import seaborn as sns
# sns.set(style='ticks', palette='Set2')
# sns.despine()


def main():
    iris = datasets.load_iris()
    iris_data = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
    iris_data["target"] = iris['target']
    iris_data["target"].iloc[0:20] = 3
    iris_data["target"].iloc[60:80] = 4
    iris_data
    plot_scatter_matrix(iris_data,
                        y_continuous=False,
                        y_var_name='target',
                        colors=True)
    plot_scatter_matrix(iris_data,
                        y_continuous=True,
                        y_var_name='petal length (cm)',
                        colors=True)

if __name__ == "__main__":
    main()
