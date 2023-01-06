import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from statistics import mean
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier

features_df = pd.read_csv('features.csv')
# mag_person1 = features_df.loc[features_df['label'] == 4]['magnitude']
# mag_person2 = features_df.loc[features_df['label'] == 6]['magnitude']

# plt.plot(range(450), mag_person1, color = 'r', label = 'person 3')
# plt.plot(range(450), mag_person2, color = 'b', label = 'person 2')
# plt.xlabel('samples')
# plt.ylabel('magnitude')
# plt.legend()
# plt.show()

labels = features_df['label']
data = features_df.drop(columns = ['label'])
data = data.set_index('id')

data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size = 2000, random_state = 42)
#hyperparam tuning
n_estimators = [int(x) for x in np.linspace(64, 2048, num = 32)]
max_features = ['auto', 'sqrt', 'log2']
max_depth = [int(x) for x in np.linspace(10, 200, num = 20)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap
}

rfc = RandomForestClassifier(
    n_estimators = 1920, min_samples_split = 2, 
    min_samples_leaf = 1,
    max_features = 'sqrt', 
    max_depth = 180,
    bootstrap = False
)
dt = DecisionTreeClassifier()
rf_randomsearch = RandomizedSearchCV(
    estimator = rfc, param_distributions = random_grid,
    n_iter = 100, cv = 10, verbose = 2, random_state = 35, n_jobs = -1
)
print(rf_randomsearch.best_params_)
# rfc.fit(data_train, labels_train)
# labels_pred_test = rfc.predict(data_test)
# print(accuracy_score(labels_test, labels_pred_test))
# cm = confusion_matrix(labels_test, labels_pred_test, labels = rfc.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = rfc.classes_)
# disp.plot()
# plt.show()

# best params {'n_estimators': 1920, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 180, 'bootstrap': False}