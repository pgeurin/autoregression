import autoregression
import pandas as pd
import matplotlib 
# matplotlib.interactive(True)
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(seed=99)
# plt.show(block=False)
iris_df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
# autoregression.compare_predictions(iris_df,'sepal_length')
iris_df['foods'] = np.random.choice(['hot dogs', 'bacon', 'sweets', np.NaN, np.inf], iris_df.shape[0], )
autoregression.compare_predictions(iris_df,'sepal_length', percent_data=1,
                        # corr_matrix=True,
                        scatter_matrix=True, 
                        bootstrap_coefs=True,
                        partial_dep=True, 
                        plot_predicted_vs_actuals_flag=True,
                        plot_coefs_flag=True,
                        feature_importances=True,
                        actual_vs_predicted=True,
                        plot_predicteds_vs_actuals=True,
                        residuals=True, 
                        univariates=True, 
                        compare_models=True,
                        ROC=True
                        )
# autoregression.compare_predictions(iris_df,'sepal_length',
#                         feature_importances=False
# )


