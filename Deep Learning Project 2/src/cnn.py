import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, Input, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation
from keras.optimizers import Adam
import json
from metrics import f1_m
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from preprocess import preprocess_data, generate_train_val_test_folds
import os

class CNN():
    def __init__(self, train_set = None, val_set = None, test_set = None):
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
    
    def compile_model(self):
        input_layer = Input(shape=(64,64,3))
        conv_1 = Conv2D(32, (3, 3), padding='same')(input_layer)
        conv_1 = BatchNormalization()(conv_1)
        conv_1 = Activation('relu')(conv_1)
        conv_1 = MaxPooling2D(pool_size=(2,2))(conv_1)

        conv_2 = Conv2D(64, (3, 3), padding='same')(conv_1)
        conv_2 = BatchNormalization()(conv_2)
        conv_2 = Activation('relu')(conv_2)
        conv_2 = MaxPooling2D(pool_size=(2,2))(conv_2)

        conv_3 = Conv2D(64, (3, 3), padding='same')(conv_2)
        conv_3 = BatchNormalization()(conv_3)
        conv_3 = Activation('relu')(conv_3)
        conv_3 = MaxPooling2D(pool_size=(2,2))(conv_3)

        conv_4 = Conv2D(64, (3, 3), padding='same')(conv_3)
        conv_4 = BatchNormalization()(conv_4)
        conv_4 = Activation('relu')(conv_4)
        conv_4 = MaxPooling2D(pool_size=(2,2))(conv_4)

        fully_connected = Flatten()(conv_4)
        fully_connected = Dense(512)(fully_connected)
        fully_connected = Activation('relu')(fully_connected)
        output_layer = Dense(150)(fully_connected)
        output_layer = Activation('softmax')(output_layer)

        model = Model(input_layer, output_layer)

        # Compile the model
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy', f1_m])

        return model

    def train_model(self, train_set, val_set, model_path = 'model.keras', history_path = 'history.json', epochs=30, batch_size = 32):
        model = self.compile_model()
        
        model.summary()
        
        earlystopper = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, mode="max")
        
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001, verbose=1)
        history = model.fit(train_set, validation_data = val_set, epochs = epochs, batch_size = batch_size, callbacks=[earlystopper, reduce_lr])
        
        model.save(model_path)
        
        self.model = model

        json.dump(str(history.history), open(history_path, 'w'))
        return history
        
    def load_model(self, model_path='model.keras'):
        self.model = tf.keras.models.load_model(model_path, custom_objects={
            'f1_m': f1_m
        })
        
    def print_loss_and_accuracy(self, test_set):
        val_loss, val_accuracy = self.model.evaluate(test_set)
        print(val_loss,val_accuracy)

DATASET_PATH = 'dataset'
if not os.path.exists("dataset_folds"):
    generate_train_val_test_folds(DATASET_PATH)

train_set, val_set, test_set = preprocess_data('dataset_folds', (64, 64))
cnn = CNN(train_set, val_set, test_set)

history = cnn.train_model(train_set, val_set, 'cnn.keras', 'cnn.json', epochs=50, batch_size=32)
