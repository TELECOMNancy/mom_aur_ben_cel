import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
#from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
import sklearn.metrics as metrics

df: pd.DataFrame = pd.read_csv('resources/training_dat.csv')
features: list = ['GR', 'ILD_log10',
                  'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
