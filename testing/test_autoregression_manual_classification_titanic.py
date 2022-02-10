import autoregression
import pandas as pd
import matplotlib 
# matplotlib.interactive(True)
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(seed=99)
# plt.show(block=False)
df_titanic = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
df_titanic = df_titanic[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
# df_titanic['Pclass'] =  df_titanic['Pclass'].map(lambda x: str(x)+' class')
df_titanic['Pclass'] =  np.array(['zero', 'class one', 'class two', 'class three'])[df_titanic['Pclass']]
df_titanic['Survived'] = np.array([True, False])[df_titanic['Survived']]
autoregression.compare_predictions(df_titanic,'survived', 
                        percent_data=1,
                        corr_matrix=True,
                        # scatter_matrix=True, #doesn't work IF categorical values of 4 groups in X.
                        bootstrap_coefs=True,
                        partial_dep=True, 
                        plot_alphas=True,
                        plot_predicted_vs_actuals_flag=True,
                        plot_coefs_flag=True,
                        feature_importances=True,
                        actual_vs_predicted=True,
                        plot_predicteds_vs_actuals=True,
                        residuals=True, 
                        univariates=True, 
                        compare_models=True,
                        ROC=True, 
                        )
# autoregression.compare_predictions(iris_df,'sepal_length',
#                         feature_importances=False
# )


