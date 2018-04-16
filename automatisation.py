import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
#from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
import sklearn.metrics as metrics

random_state = 42 # On gare toujours la même graine pou les tests.

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
    # Apprentissage
    clf.fit(x_train, y_train)

    # Prediction
    y_pred = clf.predict(self.test_well_scaled_features)

    test_well_data_with_pred = self.test_well_data
    test_well_data_with_pred.loc[:,"Prediction"] = y_pred

    #target_names = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']
    
    #report = classification_report(self.y_test_well, y_pred, target_names=target_names)
    accuracy_score = metrics.accuracy_score(self.y_test_well, y_pred)
    if (verbose_report):
      print("Accuracy_score :", accuracy_score)
    return accuracy_score

### FIN CLASSE ###

if __name__ == '__main__':
  mach = MachineLearn(
      features = ['GR', 'ILD_log10', 'DeltaPHI','PHIND', 'PE', 'NM_M', 'RELPOS']
  )
  mach2 = MachineLearn(
      features=['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'RELPOS'])
  mach3 = MachineLearn(
      features=['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M'])
  mach4 = MachineLearn(
      features=['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE'])
  mach_sans_GR = MachineLearn(
      features=['ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'RELPOS', 'NM_M']
  )
  mach_sans_ILD_log10 = MachineLearn(
    features=['GR', 'DeltaPHI', 'PHIND', 'PE', 'RELPOS', 'NM_M']
  )
  mach_sans_DeltaPHI = MachineLearn(
      features=['GR', 'ILD_log10', 'PHIND', 'PE', 'NM_M', 'RELPOS']
  )
  mach_sans_PHIND = MachineLearn(
      features=['GR', 'ILD_log10', 'DeltaPHI', 'PE', 'NM_M', 'RELPOS']
  )
  mach_sans_PE = MachineLearn(
      features=['GR', 'ILD_log10', 'DeltaPHI', 'NM_M', 'RELPOS']
  )

  #mach.test(gamma="auto")

  #TODO Faire des tests itératifs sur l'influence d'un seul paramètre
  
  results = list()

  #print("== Test originel ==")
  results.append(["Originel", mach.test()])
  #print("\n== Test sans NM_M ==")
  results.append(["Sans NM_M", mach2.test()])
  #print("\n== Test sans RELPOS ==") #RELPOS par rapport à une position de référence
  results.append(["Sans RELPOS", mach3.test()])
  #print("\n== Test sans NM_M et RELPOS ==")
  results.append(["Sans GR", mach4.test()])
  #print("\n== Test sans GR ==")
  results.append(["Sans GR", mach_sans_GR.test()])
  #print("\n== Test sans ILD_log10 ==")
  results.append(["Sans ILD_log10", mach_sans_ILD_log10.test()])
  #print("\n== Test sans DeltaPHI ==")
  results.append(["Sans DeltaPHI", mach_sans_DeltaPHI.test()])
  #print("\n== Test sans PHIND ==")
  results.append(["Sans PHIND", mach_sans_PHIND.test()])
  #print("\n== Test sans PE ==")
  results.append(["Sans PE", mach_sans_PE.test()])

  print("RESULTS")
  for i in results :
    print(i[0], "{", "accuracy :", i[1], "| delta_accuracy :", i[1] - results[0][1], "}")


  # Amélioration lorsque l'on retire la resistivité, peut être expliquée par la différence des fluides, par la resistivité 
  # variable au sein d'un même facies (la résitance est très différente en fonction de la position de la mesure dans la formation géologique)




  



    
