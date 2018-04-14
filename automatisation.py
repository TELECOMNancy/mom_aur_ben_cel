import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report

class MachineLearn():
  """[summary]
    Wrapper du proccessus de macine learning
  """

  def __init__(self, test_well_name: str = "SHANKLE", features: list = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']):
    """[summary]
    
    Keyword Arguments:
      test_well_name {str} -- nom du puit utilisé pour le test (default: {"SHANKLE"})
      features {list} -- paramètres utilisés pour le processus d'apprentissage (default: {['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']})
    """

    # Attributs
    self.well_name: str = test_well_name
    self.data: pd.DataFrame = pd.read_csv('resources/training_dat.csv')
    self.features = features

    # Decoupage des donnees en un dataframe d'aprentissage et un autre de test  
    self.learn_data = self.data[self.data['Well Name'] != test_well_name]
    self.test_well_data = self.data[self.data['Well Name'] == test_well_name] #Well = puit -> nom du puit, pour tester le machine learning

    # Preparation des vecteurs d'apprentissage/de test
    self.learn_features_vector = self.learn_data[features]
    self.learn_facies_labels = self.learn_data["Facies"]
    self.test_well_features_vector = self.test_well_data[features]

    # Scaler les vecteurs
    scaler = StandardScaler().fit(self.learn_features_vector)
    self.learn_scaled_features = scaler.transform(self.learn_features_vector) 
    self.test_well_scaled_features = scaler.transform(self.test_well_features_vector)

    # 
    self.y_test_well = self.test_well_data["Facies"]

  def show_plot(self):
    """Affiche le plot des paramètres
    """

    sns.pairplot(self.learn_features_vector[['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']])
    plt.show()

  def test(self,
      C:int = 10,
      kernel: str = 'rbf',
      degree: int = 3,
      gamma: int = 1,
      coef0: float = 0.0,
      shrinking: bool=True,
      probability: bool=False,
      tol: float =0.001,
      cache_size=200,
      class_weight=None,
      verbose=False,
      max_iter=-1,
      decision_function_shape='ovr',
      random_state=None,
      verbose_report=True):
    """Pour tester les paramètres sans tout réécrire à chaque fois
    
    Keyword Arguments:
      C {int} -- Penalty parameter C of the error term. (default: {10})
      kernel {str} -- ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’ or ‘precomputed’ (default: {'rbf'})
      degree {int} -- Degrés du kernel 'poly' si choisit (default: {3})
      gamma {int} -- Coefficent des kernel 'rbf', 'poly' et 'sigmoid'. Si gamme est'auto' 1/n_features sera utilisé.  (default: {1})
      coef0 {float} -- Coef0 pour 'poly' et 'sigmoid' (default: {0.0})
      shrinking {bool} -- Utilisation ou non de l'heuristique (default: {True})
      probability {bool} -- Méthode des probabilités (default: {False})
      tol {float} -- Tolerance pour arreter l'apprentissage (default: {0.001})
      cache_size {int} -- Taille du kernel en mémoire (MB) (default: {200})
      class_weight {[type]} -- Dictionnaires qui prmet d'associer des poids aux classes (default: {None})
      verbose {bool} -- Permet d'afficher des informations (default: {False})
      max_iter {int} -- Limite du nombre d'itératins, -1 pour aucunne (default: {-1})
      decision_function_shape {str} -- 'ovo' ou 'ovr' (default: {'ovr'})
      random_state {[type]} -- Graine aléatoire (default: {None})
      verbose_report {bool} -- Affichage ou non du report (default: True)
    """


    # Préparer la cross-validation
    x_train, x_cv, y_train, y_cv = train_test_split(
      self.learn_scaled_features,
       self.learn_facies_labels,
        test_size=0.05,
        random_state=10)
    
    #TODO Se renseigner sur la cross validation

    # Paramétrage de la machine
    clf = svm.SVC(
      C=C, 
      kernel=kernel,
      gamma=gamma,
      shrinking=shrinking,
      probability=probability,
      tol=tol,
      cache_size=cache_size,
      class_weight=class_weight,
      verbose=verbose,
      max_iter=max_iter,
      decision_function_shape=decision_function_shape,
      random_state=random_state
      )
    # Apprentissage
    clf.fit(x_train, y_train)

    # Prediction
    y_pred = clf.predict(self.test_well_scaled_features)

    test_well_data_with_pred = self.test_well_data
    test_well_data_with_pred["Prediction"] = y_pred

    target_names = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']
    
    report = classification_report(self.y_test_well, y_pred, target_names=target_names)
    print(report)
    
    #TODO Faire retourner une liste possédant les trois avg/total


mach = MachineLearn()
#mach.test(gamma="auto")

#TODO Faire des tests itératifs sur l'influence d'un seul paramètre 
for c in range(5, 30, 5):
  print("===== C =",c,"======")
  mach.test(C=c)
  print("====================")


  



    
