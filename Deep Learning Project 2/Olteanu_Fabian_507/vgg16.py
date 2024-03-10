import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, Input, MaxPooling2D, Flatten, Dense, BatchNormalization
from keras.optimizers import SGD
import json
from metrics import f1_m
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from preprocess import preprocess_data, generate_train_val_test_folds
import os

class VGG16():
    def __init__(self, train_set = None, val_set = None, test_set = None):
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
    
    def compile_model(self):
        input_layer = Input(shape=(64, 64, 3))
        first_conv_block = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(input_layer)
        first_conv_block = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(first_conv_block)
        first_conv_block = BatchNormalization()(first_conv_block)
        first_conv_block = MaxPooling2D(pool_size=(2,2), strides=(2,2))(first_conv_block)

        second_conv_block = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(first_conv_block)
        second_conv_block = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(second_conv_block)
        second_conv_block = BatchNormalization()(second_conv_block)
        second_conv_block = MaxPooling2D(pool_size=(2,2), strides=(2,2))(second_conv_block)

        third_conv_block = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(second_conv_block)
        third_conv_block = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(third_conv_block)
        third_conv_block = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(third_conv_block)
        third_conv_block = BatchNormalization()(third_conv_block)
        third_conv_block = MaxPooling2D(pool_size=(2,2), strides=(2,2))(third_conv_block)

        fourth_conv_block = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(third_conv_block)
        fourth_conv_block = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(fourth_conv_block)
        fourth_conv_block = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(fourth_conv_block)
        fourth_conv_block = BatchNormalization()(fourth_conv_block)
        fourth_conv_block = MaxPooling2D(pool_size=(2,2), strides=(2,2))(fourth_conv_block)

        fifth_conv_block = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(fourth_conv_block)
        fifth_conv_block = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(fifth_conv_block)
        fifth_conv_block = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(fifth_conv_block)
        fifth_conv_block = BatchNormalization()(fourth_conv_block)
        fifth_conv_block = MaxPooling2D(pool_size=(2,2), strides=(2,2))(fifth_conv_block)

        flatten = Flatten()(fifth_conv_block)
        fully_connected_1 = Dense(units=4096, activation='relu')(flatten)
        fully_connected_2 = Dense(units=4096, activation='relu')(fully_connected_1)
        output_layer = Dense(units=150, activation='softmax')(fully_connected_2)
        model = Model(inputs=input_layer, outputs=output_layer)

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
vgg16 = VGG16(train_set, val_set, test_set)

history = vgg16.train_model(train_set, val_set, 'vgg16.keras', 'vgg16.json', epochs=50, batch_size=32)
