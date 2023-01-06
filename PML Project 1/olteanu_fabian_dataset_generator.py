import pandas as pd
import os
import numpy as np
from scipy.signal import find_peaks

#data preprocessing
directory = 'test'

rows = []
features_df = pd.DataFrame()
labels_df = pd.read_csv('train_labels.csv')
labels_dict = dict(zip(labels_df['id'], labels_df['class']))

for filename in os.listdir(directory):
    df = pd.read_csv(directory + '/' + filename, header = None)
    df_describe = df.describe()
    axisx_fft = pd.Series(np.abs(np.fft.fft(df[0]))) #instantenous values in the frequency domain
    axisy_fft = pd.Series(np.abs(np.fft.fft(df[1])))
    axisz_fft = pd.Series(np.abs(np.fft.fft(df[2])))

    axisx_fft_describe = axisx_fft.describe()
    axisy_fft_describe = axisy_fft.describe()
    axisz_fft_describe = axisz_fft.describe()

    axisx_peaks = find_peaks(df[0])[0]
    axisy_peaks = find_peaks(df[1])[0]
    axisz_peaks = find_peaks(df[2])[0]

    avg_dist_peaksx = np.mean([np.abs(axisx_peaks[idx] - axisx_peaks[idx + 1]) for idx in range(len(axisx_peaks) - 1)]) #average time between successive peaks in a sample
    avg_dist_peaksy = np.mean([np.abs(axisy_peaks[idx] - axisy_peaks[idx + 1]) for idx in range(len(axisy_peaks) - 1)])
    avg_dist_peaksz = np.mean([np.abs(axisz_peaks[idx] - axisz_peaks[idx + 1]) for idx in range(len(axisz_peaks) - 1)])

    row = {
        'id': filename.split('.')[0],
        'axisx_mean': df_describe[0]['mean'],
        'axisy_mean': df_describe[1]['mean'],
        'axisz_mean': df_describe[2]['mean'],
        'axisx_fft_mean': axisx_fft_describe['mean'],
        'axisy_fft_mean': axisy_fft_describe['mean'],
        'axisz_fft_mean': axisz_fft_describe['mean'],
        'axisx_median': df_describe[0]['50%'],
        'axisy_median': df_describe[1]['50%'],
        'axisz_median': df_describe[2]['50%'],
        'axisx_fft_median': axisx_fft_describe['50%'],
        'axisy_fft_median': axisy_fft_describe['50%'],
        'axisz_fft_median': axisz_fft_describe['50%'],
        'crosscor_xz': df_describe[0]['mean'] / df_describe[2]['mean'],
        'crosscor_yz': df_describe[1]['mean'] / df_describe[2]['mean'],
        'crosscor_fftxz': axisx_fft_describe['mean'] / axisz_fft_describe['mean'],
        'crosscor_fftyz': axisy_fft_describe['mean'] / axisz_fft_describe['mean'],
        'avg_peak_count': np.mean(len(axisx_peaks) + len(axisy_peaks) + len(axisz_peaks)),
        'avg_dist_peaksx' : avg_dist_peaksx,
        'avg_dist_peaksy' : avg_dist_peaksy,
        'avg_dist_peaksz' : avg_dist_peaksz,
        'magnitude_time_dmn': np.sum(np.sqrt(df[0]**2 + df[1]**2 + df[2]**2)) / 150,
        'magnitude_freq_dmn': np.sum(np.sqrt(axisx_fft**2 + axisy_fft**2 + axisz_fft**2)) / 150,
        'centroid_x': np.dot(df[0], axisx_fft) / 150,
        'centroid_y': np.dot(df[1], axisy_fft) / 150,
        'centroid_z': np.dot(df[2], axisz_fft) / 150,
        'avgdist_x': np.sum(np.abs(df[0] - df_describe[0]['mean'])) / 150,   #avg distance from mean (x axis time dmn)
        'avgdist_y': np.sum(np.abs(df[1] - df_describe[1]['mean'])) / 150,   #avg distance from mean (y axis time dmn)
        'avgdist_z': np.sum(np.abs(df[2] - df_describe[2]['mean'])) / 150,   #avg distance from mean (z axis time dmn)
        'avgdist_fftx': np.sum(np.abs(axisx_fft - axisx_fft_describe['mean'])) / 150,
        'avgdist_ffty': np.sum(np.abs(axisy_fft - axisx_fft_describe['mean'])) / 150,
        'avgdist_fftz': np.sum(np.abs(axisz_fft - axisx_fft_describe['mean'])) / 150
    }
    if directory == 'train':
        row['label'] = labels_dict[int(filename.split('.')[0])]
    rows.append(row)
    print(f'{filename} done')

features_df = pd.concat([features_df, pd.DataFrame(rows)], axis = 0, ignore_index = True)
if directory == 'train':
    features_df.to_csv('features.csv', index = False)
elif directory == 'test':
    features_df.to_csv('features_test.csv', index = False)

# data visualization

# data_df = pd.read_csv('train/10003.csv', header = None)

# #FFT test
# axisx_fft = pd.Series(np.abs(np.fft.fft(data_df[0])))
# data_df[0].plot()
# plt.show()
# print(find_peaks(axisx_fft)[0])

# axisy_fft = pd.Series(np.abs(np.fft.fft(data_df[0])))
# axisz_fft = pd.Series(abs(np.fft.fft(data_df[0])))
# # pd.Series(np.fft.fft(pd.Series(data_df[0]))).plot()
# # plt.show()

# mag_person1 = features_df.loc[features_df['label'] == 4]['magnitude_freq_dmn']
# mag_person2 = features_df.loc[features_df['label'] == 6]['magnitude_freq_dmn']

# plt.plot(range(450), mag_person1, color = 'r', label = 'person 3')
# plt.plot(range(450), mag_person2, color = 'b', label = 'person 2')
# plt.xlabel('samples')
# plt.ylabel('magnitude')
# plt.legend()
# plt.show()
