import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from statistics import mean
import pickle

rfc = pickle.load(open('rfc_try_3.sav', 'rb'))
# submission
testdata_df = pd.read_csv('features_test.csv')
testdata_df = testdata_df.set_index('id')
labels_pred_test = rfc.predict(testdata_df)

submission_df = pd.DataFrame()
rows = []
for id, label in zip(testdata_df.index, labels_pred_test):
    rows.append({
        'id': id,
        'class': label
    })
submission_df = pd.concat([submission_df, pd.DataFrame(rows)], axis = 0)
submission_df.to_csv('submission.csv', index = False)