import autoregression
import pandas as pd
import matplotlib 
import numpy as np
np.random.seed(seed=99)
# matplotlib.interactive(True)
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

wine = load_wine()
df = pd.DataFrame(wine.data, columns = wine.feature_names)
df['foods'] = np.random.choice(['hot dogs', 'bacon', 'sweets', np.NaN, np.inf], df.shape[0])
# df['wine_class'] = wine['target_names'][wine.target] # This has 3 classifications!!!
df['wine_class'] = (wine['target_names'][wine.target] == 'class_1')

autoregression.compare_predictions(df, 'wine_class',
                        corr_matrix=True,
                        scatter_matrix=True, 
                        bootstrap_coefs=True,
                        partial_dep=True, 
                        plot_predicted_vs_actuals=True,
                        plot_coefs_flag=True,
                        feature_importances=True,
                        actual_vs_predicted=True,
                        plot_predicteds_vs_actuals=True,
                        residuals=True, 
                        univariates=True, 
                        compare_models=True,
                        ROC=True
)