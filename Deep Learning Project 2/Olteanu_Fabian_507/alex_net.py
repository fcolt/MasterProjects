import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, Input, MaxPooling2D, Flatten, Dense, BatchNormalization
from keras.optimizers import SGD
import json
from metrics import f1_m
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from preprocess import preprocess_data, generate_train_val_test_folds
import os

class AlexNet():
    def __init__(self, train_set, val_set, test_set):
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
    
    def compile_model(self):
        input_layer = Input(shape=(64,64,3))

        conv_1 = Conv2D(96, (11,11),strides=(4,4), activation='relu')(input_layer)
        conv_1 = BatchNormalization()(conv_1)
        conv_1 = MaxPooling2D(2, strides=(2,2))(conv_1)

        conv_2 = Conv2D(256, (11,11),strides=(1,1), activation='relu',padding="same")(conv_1)
        conv_2 = BatchNormalization()(conv_2)

        conv_3 = Conv2D(384, (3,3),strides=(1,1), activation='relu',padding="same")(conv_2)
        conv_3 = BatchNormalization()(conv_3)

        conv_4 = Conv2D(384, (3,3),strides=(1,1), activation='relu',padding="same")(conv_3)
        conv_4 = BatchNormalization()(conv_4)

        conv_5 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu',padding="same")(conv_4)
        conv_5 = BatchNormalization()(conv_5)
        conv_5 = MaxPooling2D(2, strides=(2,3))(conv_5)

        fully_connected_1 = Flatten()(conv_5)
        fully_connected_1 = Dense(4096, activation='relu')(fully_connected_1)

        fully_connected_2 = Dense(4096, activation='relu')(fully_connected_1)
        
        output_layer = Dense(150, activation="softmax")(fully_connected_2)

        model = Model(input_layer, output_layer)

        opt = SGD(learning_rate=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc', f1_m])
        
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
alexnet = AlexNet(train_set, val_set, test_set)

history = alexnet.train_model(train_set, val_set, 'vgg16.keras', 'vgg16.json', epochs=50, batch_size=32)