import os
from preprocess import preprocess_data
import numpy as np
import pandas as pd
from keras.saving import load_model
from metrics import f1_m
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

train_set, val_set, test_set = preprocess_data('dataset_folds', (64, 64))

cnn = load_model('/kaggle/input/cnn/tensorflow2/cnn/1/cnn.keras', custom_objects={'f1_m': f1_m})
vgg16 = load_model('vgg16.keras', custom_objects={'f1_m': f1_m})
alexnet = load_model('alexnet.keras', custom_objects={'f1_m': f1_m})

print(cnn.evaluate(test_set))
print(vgg16.evaluate(test_set))
print(alexnet.evaluate(test_set))

cnn_preds = cnn.predict(test_set)
cnn_preds = np.argmax(cnn_preds, axis=1)
vgg16_preds = vgg16.predict(test_set)
vgg16_preds = np.argmax(vgg16_preds, axis=1)
alexnet_preds = alexnet.predict(test_set)
alexnet_preds = np.argmax(alexnet_preds, axis=1)

conf_matrix_cnn = confusion_matrix(cnn_preds, test_set.labels)
conf_matrix_vgg16 = confusion_matrix(vgg16_preds, test_set.labels)
conf_matrix_alexnet = confusion_matrix(alexnet_preds, test_set.labels)
  
sns.heatmap(conf_matrix_cnn, fmt='d', cmap='Blues', xticklabels=[], yticklabels=[])
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.savefig('conf_matrices/cnn_conf_matrix.png')
plt.clf()

sns.heatmap(conf_matrix_vgg16, fmt='d', cmap='Blues', xticklabels=[], yticklabels=[])
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.savefig('conf_matrices/vgg16_conf_matrix.png')
plt.clf()

sns.heatmap(conf_matrix_alexnet, fmt='d', cmap='Blues', xticklabels=[], yticklabels=[])
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.savefig('conf_matrices/alexnet_conf_matrix.png')
plt.clf()