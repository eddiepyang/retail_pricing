# algebra and dataframes
import numpy as np
import pandas as pd

# hypothesis testing
from scipy.stats import gamma, kstest, lognorm, mannwhitneyu, ks_2samp

import xgboost as xgb

# charts
import matplotlib.pyplot as plt
import seaborn as sns

# charting paramemters
from pylab import rcParams

rcParams.update({'font.size': 14, 'legend.fontsize': "small",
                 "xtick.labelsize": 14, "ytick.labelsize": 14,
                 "figure.figsize": (9, 6), "axes.titlesize": 20,
                 "axes.labelsize": 14, "lines.linewidth": 3,
                 "lines.markersize": 10
                 })

# data prep
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# measurement metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
