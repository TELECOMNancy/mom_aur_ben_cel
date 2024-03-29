import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
#from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
import copy
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier

random_state = 42  # On gare toujours la même graine pou les tests.

class MachineLearn():
  """[summary]
    Wrapper du proccessus de macine learning
  """

  def __init__(self,
     test_well_name: str = "SHANKLE",
      features: list = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS'],
      facies_group: list = [1,2,3,4,5,6,7,8]
      ):
    """[summary]

    Keyword Arguments:
      test_well_name {str} -- nom du puit utilisé pour le test (default: {"SHANKLE"})
      features {list} -- paramètres utilisés pour le processus d'apprentissage (default: {['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']})
    """

    # Attributs
    self.well_name: str = test_well_name
    self.data: pd.DataFrame = pd.read_csv('resources/training_dat.csv')
    self.features = features

    #print(self.data['Facies'])

    # Regouprer les facies
    self.data.Facies.replace([1, 2, 3, 4, 5, 6, 7, 8],
     [int(facies_group[0]),
      int(facies_group[1]),
      int(facies_group[2]),
      int(facies_group[3]),
      int(facies_group[4]),
      int(facies_group[5]),
      int(facies_group[6]),
      int(facies_group[7])], inplace=True)

    # Decoupage des donnees en un dataframe d'aprentissage et un autre de test
    self.learn_data = self.data[self.data['Well Name'] != test_well_name]
    # Well = puit -> nom du puit, pour tester le machine learning
    self.test_well_data = self.data[self.data['Well Name'] == test_well_name]

    # Preparation des vecteurs d'apprentissage/de test
    ## Discrimination
    """
    print(len(self.learn_data))
    discrim = "GR"
    self.learn_data = self.learn_data[
        (self.learn_data[discrim] > self.learn_data[discrim].mean() - self.learn_data[discrim].std()) &
        (self.learn_data[discrim] < self.learn_data[discrim].mean() + self.learn_data[discrim].std())
        ]
    print(len(self.learn_data))
    """
    # -> ne marche pas : baisse de performances

    self.learn_features_vector = self.learn_data[features]
    ## Logarithme
    """
    print(self.learn_features_vector)
    self.learn_features_vector["GR"] = self.learn_features_vector["GR"]
    print(self.learn_features_vector)
    """
    # -> ne marche pas, résultats trop faibles

    self.learn_facies_labels = self.learn_data["Facies"]
    self.test_well_features_vector = self.test_well_data[features]

    # Scaler les vecteurs
    scaler = StandardScaler().fit(self.learn_features_vector)
    self.learn_scaled_features = scaler.transform(self.learn_features_vector)
    self.test_well_scaled_features = scaler.transform(
        self.test_well_features_vector)

    #
    self.y_test_well = self.test_well_data["Facies"]

  def show_plot(self):
    """Affiche le plot des paramètres
    """

    sns.pairplot(self.learn_features_vector[[
                 'GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']])
    plt.show()

  def test(self,
           C: int = 10,
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
           verbose_report=False):
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
    """
    x_train, x_cv, y_train, y_cv = train_test_split(
      self.learn_scaled_features,
      self.learn_facies_labels,
        test_size=0.05,
        random_state=random_state)
    """
    x_train = self.learn_scaled_features
    y_train = self.learn_facies_labels

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
    """
    clf = svm.LinearSVC(C=10, max_iter=1000, random_state=42, class_weight="balanced")
    """

    # Apprentissage
    clf.fit(x_train, y_train)

    # Prediction
    y_pred = clf.predict(self.test_well_scaled_features)

    test_well_data_with_pred = self.test_well_data
    test_well_data_with_pred.loc[:, "Prediction"] = y_pred

    #target_names = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']

    #report = classification_report(self.y_test_well, y_pred, target_names=target_names)
    accuracy_score = metrics.accuracy_score(self.y_test_well, y_pred)
    if (verbose_report):
      print("Accuracy_score :", accuracy_score)
    return accuracy_score

### FIN CLASSE ###


if __name__ == '__main__':

  def combinations(target,data):
      for i in range(len(data)):
        new_target = copy.copy(target)
        new_data = copy.copy(data)
        new_target.append(data[i])
        new_data = data[i+1:]
        #print(new_target)
        combft.append(new_target)
        combinations(new_target, new_data)

  #def listfacies():


  target = []
  facies_group = [1,2,3,4,5,6,7,8] # Si'lon veut regrouper les facies
  combfacies = []
  ft = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
  combft = [] # Liste contenant la liste des combinaisons des éléments de ft
  results = list()


  combinations(target,ft)
  for i in combft:
      mach = MachineLearn(
          features=i,
          facies_group = facies_group
      )
      results.append([i, mach.test()])

  print("RESULTS")
  print("For facies =", facies_group)
  print("--------------------")
  for i in results:
    if i[1] - results[6][1] >= 0:
        print(i[0], "{", "accuracy :", "{0:.3f}".format(i[1]),
            "| delta_accuracy :", "{0:.3f}".format(i[1] - results[6][1]), "}")
  print("--------------------")


  #We have 6 results improving the result by at least 5% :
  #['GR', 'PHIND']
  #['GR', 'PHIND', 'PE']
  #['GR', 'PHIND', 'PE', 'NM_M']
  #['GR', 'PHIND', 'PE', 'NM_M', 'RELPOS']
  #['GR', 'PE', 'NM_M']
  #['ILD_log10', 'PHIND', 'PE', 'NM_M']
