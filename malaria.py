#importar blibliotecas que irao permitir trabalhar
import numpy as np
import pandas as pd

from pickle import dump

from matplotlib import pyplot
from collections import Counter

from sklearn.model_selection import KFold,train_test_split,cross_val_score

#alqoritmos de classificacao
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

#importar dados do dataset (base de conhecimento)
dataset = np.loadtxt('dataset_detencao_malaria.data',delimiter=',')
#A Primeiro coluna se esta ou nao com febre(37.2 - 37.8 > 37.8)
#A Segunda coluna se esta ou nao com dor de cabeca, 0 ou  1
#A Terceira coluna se esta ou nao com calafrios, 0 ou  1
#A Quarta coluna se esta ou nao com suor excessivo, 0 ou  1
#A Quinta coluna se esta ou nao com dores musculares, 0 ou  1
#A Sexta coluna se esta ou nao com nauseas, 0 ou  1
#A Setima coluna se esta ou nao com dor nas costas, 0 ou  1
#A Oitava coluna se esta ou nao com vomitos, 0 ou  1
#A Nona coluna se esta ou nao com tosse, 0 ou  1
#A Decima coluna se esta ou nao com diarreia, 0 ou  1
#A Decima Primeira coluna se esta ou nao com malaria(resultado), 0 ou  1
names=['febre', 'dorDeCabeca', 'calafrios', 'suorExcessivo', 'doresMusculares', 'nauseas','dorNasCostas', 'vomitos','tosse','diarreia','malaria']

#criamos um data frame para trabalhar de uma forma melhor
dataset=pd.DataFrame(dataset,columns=names)
print(dataset.head())

#distribuicao da variavel target(alvo)
print(Counter(dataset['malaria']))

#dataset.plot

#Extrai a variavel target(alvo) para uma variavel especifica e a elimina da base
target = dataset['malaria']
datasetFiltered = dataset.drop('malaria',1)

print(datasetFiltered.describe())

#configura alguns parametros
seed=7
scoring='recall'
test_size=0.25
n_splits=10
kFold = KFold(n_splits=n_splits,random_state=seed)

#separa a base entre treino e teste de forma aleatoria
X = datasetFiltered.values
Y = target
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=test_size)

#alogritmos de classificacao
models = []
models.append(('ADA',AdaBoostClassifier()))
models.append(('GB',GradientBoostingClassifier()))
models.append(('RF',RandomForestClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('SVM',SVC()))

#Avalia os algoritmos
results = []
names =[]

for name, model in models:
    cv_results = cross_val_score(model,X_train,Y_train, cv=kFold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg="%s: %f (%f)" % (name, cv_results.mean(),cv_results.std())
    print(msg)

#compara os algoritmos
fig = pyplot.figure()
fig.suptitle('Comparacao de Algoritmos')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

#cria o modelo base com o melhor algoritmo
#no caso pode ser qualquer um dos 4 primeiros
baseline = RandomForestClassifier(random_state=seed)
baseline.fit(X_train,Y_train)

#verfica a importancia de cada variavel no modelo
featureName = pd.DataFrame(datasetFiltered.columns,columns=['Feature'])
featureImp = pd.DataFrame(baseline.feature_importances_, columns=['Imp'])
print(pd.concat([featureName, featureImp], axis=1))

#testando o treinamento com base de teste
predictions = baseline.predict(X_test)

print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test,predictions))

#guarda o treinamento em arquivo do tipo sav
filename = 'binary_class_model.sav'
dump(baseline,open(filename,'wb'))