import pandas as pd
import os
import numpy as np

#data preprocessing
directory = 'train'
rows = []
features_df = pd.DataFrame()
labels_df = pd.read_csv('train_labels.csv')
labels_dict = dict(zip(labels_df['id'], labels_df['class']))

for filename in os.listdir(directory):
    df = pd.read_csv(directory + '/' + filename, header = None)
    df_describe = df.describe()
    rows.append({
        'id': filename.split('.')[0],
        'label': labels_dict[int(filename.split('.')[0])],
        'axisx_mean': df_describe[0]['mean'],
        'axisy_mean': df_describe[1]['mean'],
        'axisz_mean': df_describe[2]['mean'],
        'axisx_median': df_describe[0]['50%'],
        'axisy_median': df_describe[1]['50%'],
        'axixz_median': df_describe[2]['50%'],
        'crosscor_xz': df_describe[0]['mean'] / df_describe[2]['mean'],
        'crosscor_yz': df_describe[1]['mean'] / df_describe[2]['mean'],
        'magnitude': np.sum(np.sqrt(df[0]**2 + df[1]**2 + df[2]**2)) / 150,
        'avgdist_x': np.sum(np.abs(df[0] - df_describe[0]['mean'])) / 150,   #avg distance from mean (x axis)
        'avgdist_y': np.sum(np.abs(df[1] - df_describe[1]['mean'])) / 150,   #avg distance from mean (y axis)
        'avgdist_z': np.sum(np.abs(df[2] - df_describe[2]['mean'])) / 150    #avg distance from mean (z axis)
    })
    print(f'{filename} done')

features_df = pd.concat([features_df, pd.DataFrame(rows)], axis = 0, ignore_index = True)
features_df.to_csv('features.csv', index = False)
