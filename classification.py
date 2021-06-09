# Classification Project: Sonar rocks or mines

from matplotlib import pyplot
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn import tree

# Load dataset
dataset = read_csv('sonar.csv', header=None)

dataFrame = dataset.values
X = dataFrame[:, 0:60].astype(float)
Y = dataFrame[:, 60]
# Split into training and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

models = []
models.append(('NaiveBayes', GaussianNB()))
models.append(('LogisticRegression', LogisticRegression()))
models.append(('DecisionTree', DecisionTreeClassifier()))
models.append(('NearestNeighbors', KNeighborsClassifier()))
results = []
names = []
for name, model in models:
    knoll = KFold(n_splits=10, shuffle=False, random_state=None)  # Indices to split
    cv_results = cross_val_score(model, X_train, Y_train, cv=knoll,
                                 scoring='accuracy')  # Evaluate estimator performance with cross-validation
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    results.append(cv_results)
    names.append(name)
    '''
    # Decision tree graph
    if name == 'DecisionTree':
        clf = model.fit(X_train, Y_train)
        tree.plot_tree(model, filled=True)
        plt.show()
    '''

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# Standardize dataset
pipelines = []
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()), ('NB', GaussianNB())])))
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()), ('LR', LogisticRegression())])))
pipelines.append(('ScaledDT', Pipeline([('Scaler', StandardScaler()), ('DT', DecisionTreeClassifier())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsClassifier())])))

for name, model in pipelines:
    # Model training
    model.fit(X_train, Y_train)

    predictions = model.predict(X_test)
    print('Precision del modelo %s:' % name, model.score(X_test, Y_test))

# Finally evaluate the most accurate model (KNN)
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = KNeighborsClassifier()
model.fit(rescaledX, Y_train)

# Estimate accuracy on validation dataset
rescaledValidationX = scaler.transform(X_test)
predictions = model.predict(rescaledValidationX)
print('\nVecino mas Cercano\nPrecision de entrenamiento:', round(model.score(rescaledX, Y_train) * 100, 2), '%')
print('Precision de prueba:', round(accuracy_score(Y_test, predictions) * 100, 2), '%')
print('Matriz de Confusion:\n', confusion_matrix(Y_test, predictions))
print('Reporte de Clasificacion\n', classification_report(Y_test, predictions))

# Prediction new case
ejemplo = [[0.0185, 0.0346, 0.0168, 0.0177, 0.0393, 0.1630, 0.2028, 0.1694, 0.2328, 0.2684, 0.3108, 0.2933,
            0.2275, 0.0994, 0.1801, 0.2200, 0.2732, 0.2862, 0.2034, 0.1740, 0.4130, 0.6879, 0.8120, 0.8453,
            0.8919, 0.9300, 0.9987, 1.0000, 0.8104, 0.6199, 0.6041, 0.5547, 0.4160, 0.1472, 0.0849, 0.0608,
            0.0969, 0.1411, 0.1676, 0.1200, 0.1201, 0.1036, 0.1977, 0.1339, 0.0902, 0.1085, 0.1521, 0.1363,
            0.0858, 0.0290, 0.0203, 0.0116, 0.0098, 0.0199, 0.0033, 0.0101, 0.0065, 0.0115, 0.0193, 0.0157]]
y_pred = model.predict(ejemplo)
print("Prediccion: " + str(y_pred))
