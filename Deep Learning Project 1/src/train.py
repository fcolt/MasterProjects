import pandas as pd
from custom_cnn import CNNClassifier
from custom_effinet import EffiNetClassifier

train_df = pd.read_csv("train.csv", dtype=str)
val_df = pd.read_csv("val.csv", dtype=str)

classifier1 = CNNClassifier(
    train_df, 
    val_df,
    "train_images/",
    "val_images/"
)

classifier2 = EffiNetClassifier(
    train_df, 
    val_df,
    "train_images/",
    "val_images/"
)

train, val = classifier1.train_test()

history = classifier1.train_model(train, val, 'custom_cnn.keras', 'custom_cnn.json', epochs=50)
history2 = classifier2.train_model(train, val, 'effinet-norm.keras', 'effinet-norm.json', epochs=50)
