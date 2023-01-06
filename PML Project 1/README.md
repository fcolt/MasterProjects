# Dataset Description

The task is to discriminate between 20 mobile device users. For each user, there are 450 accelerometer signal recordings (examples) available for training. The signals are recorded for 1.5 seconds, while the user taps on the smartphone's screen. The accelerometer signal is recorded at 100 Hz, thus containing roughly 150 values. The values are recorded on three different axes: x, y, z.

Each example is assigned to 1 of 20 classes. The training set consists of 9,000 labeled examples. The test set consists of another 5,000 examples. The test labels are not provided with the data.

# File descriptions

    - train.zip - the training samples (one sample per .csv file)
    - train_labels.csv - the training labels (one label per row)
    - test.zip - the test samples (one sample per .csv file)
    - sample_submission.csv - a sample submission file in the correct format

# Metadata file format

The labels associtated to the training samples are provided in the train_labels_.csv file with the following format:

id,class
10003,7

...
23999,18

For example, the first row indicates that the data sample file named '10003.csv' belongs to class 7.

# Data sample file format

Each sample (accelerometer signal) is provided in a .csv file with the following format:

0.486023,5.588067,7.275979
0.469264,5.566519,7.316082
0.420781,5.550358,7.382521
...
-0.527323,6.221333,6.872556

Each row in the .csv file provides the x, y and z accelerometer values at a given moment in time.