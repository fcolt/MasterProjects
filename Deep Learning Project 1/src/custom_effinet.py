from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import keras.layers as layers
import json
import pandas as pd
from metrics import f1_m
from keras.callbacks import EarlyStopping

class EffiNetClassifier:
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
    
    def create_custom_effnet_l2(input_shape=(64, 64, 3), num_classes=100, lr=0.001):
        def swish_activation(x):
            return x * tf.keras.activations.sigmoid(x)

        def se_block(x, ratio=16):
            channels = x.shape[-1]
            reduced_channels = max(1, channels // ratio)

            se_branch = layers.GlobalAveragePooling2D()(x)
            se_branch = layers.Reshape((1, 1, channels))(se_branch)
            se_branch = layers.Conv2D(reduced_channels, (1, 1), activation='relu', padding='same')(se_branch)
            se_branch = layers.Conv2D(channels, (1, 1), activation='sigmoid', padding='same')(se_branch)

            return layers.Multiply()([x, se_branch])

        def block(x, filters, kernel_size, stride, expand_ratio, se_ratio=0.25):
            channels = x.shape[-1]
            expand_channels = int(channels * expand_ratio)

            # Expansion phase
            y = layers.Conv2D(expand_channels, (1, 1), padding='same')(x)
            y = layers.BatchNormalization()(y)
            y = layers.Activation(swish_activation)(y)

            # Depthwise Convolution
            y = layers.DepthwiseConv2D(kernel_size, strides=(stride, stride), padding='same')(y)
            y = layers.BatchNormalization()(y)
            y = layers.Activation(swish_activation)(y)

            # Squeeze and Excitation
            if 0 < se_ratio <= 1:
                y = se_block(y, se_ratio)

            # Projection phase
            y = layers.Conv2D(filters, (1, 1), padding='same')(y)
            y = layers.BatchNormalization()(y)

            # Shortcut connection
            if stride == 1 and channels == filters:
                y = layers.Add()([x, y])

            return y

        input_tensor = layers.Input(shape=(64,64,3))

        x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(swish_activation)(x)

        x = block(x, filters=128, kernel_size=3, stride=1, expand_ratio=1)
        x = block(x, filters=256, kernel_size=3, stride=2, expand_ratio=6)
        x = block(x, filters=512, kernel_size=5, stride=2, expand_ratio=6)
        x = block(x, filters=1024, kernel_size=3, stride=2, expand_ratio=6)
        x = block(x, filters=2048, kernel_size=3, stride=1, expand_ratio=6)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        output_tensor = layers.Dense(num_classes, activation='softmax')(x)

        model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        model.compile(optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', f1_m]
        )

        return model 
    
    def train_model(self, train_set, test_set, model_path = 'model.keras', history_path = 'history.json', epochs=30):
        model = self.create_custom_effnet_l2()
        
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