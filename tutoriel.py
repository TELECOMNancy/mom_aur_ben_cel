import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('resources/training_dat.csv')

test_well = data[data['Well Name'] == 'SHANKLE']
data = data[data['Well Name'] != 'SHANKLE']
#print(data)

# Extraction du vecteur features composés des colonnes 
features = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
features_vectors = data[features]
facies_labels = data['Facies']

# Affichage d'un crossplots
sns.pairplot(features_vectors[['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']])
plt.show()



