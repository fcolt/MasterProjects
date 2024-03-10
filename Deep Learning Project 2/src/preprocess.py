import os
import matplotlib.pyplot as plt
import splitfolders
from shutil import rmtree
from keras.preprocessing.image import ImageDataGenerator

def generate_histogram(dataset_path):
    classes = os.listdir(dataset_path)
    class_counts = []
    
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            num_images = len(os.listdir(class_path))
            class_counts.append(num_images)

    plt.bar(classes, class_counts, align='center')
    plt.xticks(range(len(classes)), classes)
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.title('Histogram of Image Classes')
    plt.show()

def class_to_label(dataset_path):
    classes = os.listdir(dataset_path)
    id_by_label = {} 
    label_by_id = {}
    
    for i, label in enumerate(classes):
        id_by_label[label] = i
        label_by_id[i] = label
    
    return id_by_label, label_by_id

def generate_train_val_test_folds(dataset_path, output_path="dataset_folds"):
    if (os.path.isdir(output_path)):
        rmtree(output_path)
    splitfolders.ratio(dataset_path, output=output_path, seed=1337, ratio=(0.75, 0.15, 0.1))

def preprocess_data(folds_folder="dataset_folds", target_size=(64, 64)):
    train_datagen = ImageDataGenerator(
        rescale=1 / 255.0,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest",
        validation_split=0)
    
    test_datagen = ImageDataGenerator(rescale=1 / 255.0)
    
    train_generator = train_datagen.flow_from_directory(
        directory=folds_folder + '/train',
        target_size=target_size,
        color_mode="rgb",
        class_mode="categorical",
    )
    
    valid_generator = train_datagen.flow_from_directory(
        directory=folds_folder + '/val',
        target_size=target_size,
        color_mode="rgb",
        class_mode="categorical",
    )
    test_generator = test_datagen.flow_from_directory(
        directory=folds_folder + '/test',
        target_size=target_size,
        color_mode="rgb",
        batch_size=1,
        class_mode="categorical",
        shuffle=False
    )
    
    return train_generator, valid_generator, test_generator