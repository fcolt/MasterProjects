from custom_cnn import CNNClassifier
from custom_effinet import EffiNetClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def map_preds_to_truth(preds_csv_path, truth_csv_path):
    return pd.merge(
        pd.read_csv(truth_csv_path), 
        pd.read_csv(preds_csv_path), 
        on='Image'
    )


train_df = pd.read_csv("train.csv", dtype=str)
val_df = pd.read_csv("val.csv", dtype=str)

custom_cnn_model = CNNClassifier(
    train_df, 
    val_df,
    "train_images/",
    "val_images/"
)

custom_effinet_model = EffiNetClassifier(
    train_df, 
    val_df,
    "train_images/",
    "val_images/"
)

custom_cnn_model.load_model(model_path='submission_models/custom_cnn.keras')
custom_cnn_model.generate_predictions('val_images', 'val_custom_cnn.csv')

custom_effinet_model.load_model(model_path='submission_models/effinet-norm.keras')
custom_effinet_model.generate_predictions('val_images', 'val_effinet.csv')

val_df_custom_cnn = map_preds_to_truth('val_custom_cnn.csv', 'val.csv') 
val_df_effinet = map_preds_to_truth('val_effinet.csv', 'val.csv')

conf_matrix_custom_cnn = confusion_matrix(y_pred=val_df_custom_cnn['Class_x'], y_true=val_df_custom_cnn['Class_y'])
conf_matrix_effinet = confusion_matrix(y_pred=val_df_effinet['Class_x'], y_true=val_df_effinet['Class_y'])

sns.heatmap(conf_matrix_custom_cnn, fmt='d', cmap='Blues', xticklabels=[], yticklabels=[])
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.savefig('submission_models/custom_cnn_conf_matrix.png')
plt.clf()

sns.heatmap(conf_matrix_effinet, fmt='d', cmap='Blues', xticklabels=[], yticklabels=[])
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.savefig('submission_models/effinet_conf_matrix.png')

_, val = custom_cnn_model.train_test()
custom_cnn_model.model.evaluate(val)
custom_effinet_model.model.evaluate(val)