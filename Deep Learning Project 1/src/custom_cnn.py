from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Conv2D, Input, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
import json
import pandas as pd
from metrics import f1_m
from keras.callbacks import EarlyStopping

class CNNClassifier:
    def __init__(self, train_df = None, val_df = None, train_path = None, val_path = None):
        self.train_path = train_path
        self.train_df = train_df
        self.val_path = val_path
        self.val_df = val_df

    def train_test(self):
        datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split = 0.25
        )

        train_set = datagen.flow_from_dataframe(
            dataframe = self.train_df,
            directory = self.train_path,
            x_col = "Image",
            y_col = "Class",
            target_size = (64, 64), 
            batch_size = 32,
            class_mode = 'categorical',
            shuffle = True,
            subset = 'training',
            seed = 123
        )

        test_set = datagen.flow_from_dataframe(
            dataframe = self.val_df,
            directory = self.val_path,
            x_col = "Image",
            y_col = "Class",
            target_size = (64, 64), 
            batch_size = 32,
            class_mode = 'categorical',
            shuffle = False,
            subset = 'validation'
        )

        return (train_set,test_set)
    
    def create_cnn(self):
        input_layer = Input(shape=(64, 64, 3))
        conv1 = Conv2D(25, (5, 5), activation='relu', strides=(1, 1), padding='same')(input_layer)
        pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1)
        conv2 = Conv2D(50, (5, 5), activation='relu', strides=(2, 2), padding='same')(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)
        batch_norm1 = BatchNormalization()(pool2)
        conv3 = Conv2D(70, (3, 3), activation='relu', strides=(2, 2), padding='same')(batch_norm1)
        pool3 = MaxPooling2D(pool_size=(2, 2), padding='valid')(conv3)
        batch_norm2 = BatchNormalization()(pool3)
        flatten = Flatten()(batch_norm2)
        dense1 = Dense(units=256, activation='relu')(flatten)
        dense2 = Dense(units=128, activation='relu')(dense1)
        dropout = Dropout(0.25)(dense2)
        output_layer = Dense(units=100, activation='softmax')(dropout)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', f1_m])
        
        return model
    
    def train_model(self, train_set, test_set, model_path = 'model.keras', history_path = 'history.json', epochs=30):
        model = self.create_cnn()
        
        model.summary()
        
        earlystopper = EarlyStopping(monitor='val_f1_m', patience=10, verbose=1)
        history = model.fit(train_set, validation_data = test_set, epochs = epochs, batch_size = 32, callbacks=[earlystopper])
        
        model.save(model_path)
        json.dump(history.history, open(history_path, 'w'))
        
        self.model = model
        
        return history
        
    def load_model(self, model_path='model.keras'):
        self.model = tf.keras.models.load_model(model_path, custom_objects={
            'swish_activation': lambda x: x * tf.keras.activations.sigmoid(x),
            'f1_m': f1_m
        })
        
    def print_loss_and_accuracy(self, test_set):
        val_loss, val_accuracy = self.model.evaluate(test_set)
        print(val_loss,val_accuracy)

    def generate_predictions(self, img_path, output_filepath, batch_size=32):
        if not img_path.endswith("/"):
            img_path = img_path + "/"

        train_set = ImageDataGenerator(rescale=1./255, validation_split = 0.25).flow_from_dataframe(
            dataframe = self.train_df,
            directory = self.train_path,
            x_col = "Image",
            y_col = "Class",
            target_size = (64, 64), 
            batch_size = 32,
            class_mode = 'categorical',
            shuffle = True,
            subset = 'training',
            seed = 123
        )

        datagen = ImageDataGenerator(rescale=1./255).flow_from_directory(
            directory=img_path,
            color_mode='rgb',
            batch_size=batch_size,
            shuffle=False,
            class_mode='binary',
            classes=['.'],
            target_size=(64,64)
        )
            
        labels = (train_set.class_indices)
        labels = dict((v,k) for k,v in labels.items())
        predicted_class_indices = np.argmax(self.model.predict(datagen), axis=1)
        preds = [labels[k] for k in predicted_class_indices]

        filenames = [filename[2:] for filename in datagen.filenames]
        filenames_to_cls = list(zip(filenames, preds))

        df = pd.DataFrame(filenames_to_cls, columns=["Image", "Class"])

        df.to_csv(output_filepath, index=False)