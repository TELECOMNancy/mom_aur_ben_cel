import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
#from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
from sklearn import neighbors
random_state = 42  # On gare toujours la même graine pou les tests.
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier


features: list = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
test_well_name: str = "SHANKLE"
data: pd.DataFrame = pd.read_csv('resources/training_dat.csv')
features = features

# Decoupage des donnees en un dataframe d'aprentissage et un autre de test
learn_data = data[data['Well Name'] != test_well_name]
# Well = puit -> nom du puit, pour tester le machine learning
test_well_data = data[data['Well Name'] == test_well_name]
learn_features_vector = learn_data[features]    
learn_facies_labels = learn_data["Facies"]
test_well_features_vector = test_well_data[features]

# Scaler les vecteurs
scaler = StandardScaler().fit(learn_features_vector)
learn_scaled_features = scaler.transform(learn_features_vector)
test_well_scaled_features = scaler.transform(test_well_features_vector)
x_test_well = test_well_scaled_features
y_test_well = test_well_data["Facies"]
    
x_train = learn_scaled_features
y_train = learn_facies_labels

""" PCA """

"""
pca = PCA(.95) #95% de la variance est retenu pour la pca
pca.fit(x_train)
x_train = pca.transform(x_train)
x_test_well = pca.transform(x_test_well)
"""

# Paramétrage de la machine
"""
clf = svm.SVC(C= 10,
           kernel= 'rbf',
           degree= 3,
           gamma = 1,
           coef0= 0.0,
           shrinking=True,
           probability=False,
           tol =0.001,
           cache_size=200,
           class_weight=None,
           verbose=False,
           max_iter=-1,
           decision_function_shape='ovr',
           random_state=42)
#from sklearn.linear_model import LogisticRegression
"""
#clf = LogisticRegression(solver='lbfgs')

"""
vp = [] #vecteur poid
for i in range (1,9): 
    vp.append(float(len(data[data["Facies"]==i]))/float(len(data)))
"""
# Réduit la

# One versus maa
#clf =  svm.LinearSVC(C = 10,max_iter=1000, random_state=42, class_weight="balanced")
clf = AdaBoostClassifier(n_estimators=200)
#clf = neighbors.KNeighborsClassifier(n_neighbors=42,weights='distance')

# Apprentissage
clf.fit(x_train, y_train)

# Prediction
y_pred = clf.predict(x_test_well)

test_well_data_with_pred = test_well_data
test_well_data_with_pred.loc[:, "Prediction"] = y_pred

#target_names = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']

#report = classification_report(y_test_well, y_pred, target_names=target_names)
accuracy_score = metrics.accuracy_score(y_test_well, y_pred)
print("Accuracy_score :", accuracy_score)

### FIN CLASSE ###

#