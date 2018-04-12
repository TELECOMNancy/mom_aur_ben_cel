import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn import svm

data = pd.read_csv('resources/training_dat.csv')

test_well = data[data['Well Name'] == 'SHANKLE'] #Well = puit -> nom du puit, pour tester le machine learning
data = data[data['Well Name'] != 'SHANKLE'] #Pour le machine learning 
#print(data)

# Extraction du vecteur features composés des colonnes 
features = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
features_vectors = data[features]
facies_labels = data['Facies']
#print(facies_labels)


# Affichage d'un crossplots
"""
sns.pairplot(features_vectors[['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']])
plt.show()
"""

# Faire devenir les echantillons gaussiens (moyenne nulle, variance 1)
scaler = StandardScaler().fit(features_vectors)
scaled_feature = scaler.transform(features_vectors)

# Préparer la cross-validation
x_train, x_cv, y_train, y_cv =  train_test_split(scaled_feature, facies_labels, test_size=0.05, random_state=10)

# Paramétrage de la machine
clf = svm.SVC(C=10, gamma=1)

# Apprentissage
clf.fit(x_train, y_train)

#TODO voir ce qu'on fait avec x_cv et y_cv

# On normalise les données de test
y_test = test_well['Facies']
well_features = test_well.drop(['Facies', 'Formation', 'Well Name', 'Depth'], axis=1) # On retire les colonnes en param, on aurait pu utiliser l15-l16
x_test = scaler.transform(well_features)

# On prédit y
y_pred = clf.predict(x_test)

test_well['Prediction'] = y_pred

from sklearn.metrics import classification_report

target_names = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']
print(classification_report(y_test, y_pred, target_names=target_names))









