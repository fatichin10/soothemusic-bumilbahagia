import os
import librosa
import numpy as np
import csv


def extract_feature(src):
    dataset = []
    genres = [x for x in os.listdir(src)]
    for genre in genres:
        for file in os.listdir(f'{src}/{genre}'):
            if file.endswith('.wav'):
                filename = os.path.splitext(file)[0]
                feature_vector = get_feature(f'{src}/{genre}/{file}')
                record = [filename, genre] + feature_vector
                dataset.append(record)

    dataset.insert(0, ['filename', 'genre', 'zcr_mean', 'zcr_std', 'chroma_stft_mean', 'chroma_stft_std',
                       'spectral_centroid_mean', 'spectral_centroid_std',
                       'spectral_rolloff_mean', 'spectral_rolloff_std',
                       'spectral_bandwidth_mean', 'spectral_bandwidth_std',
                       'mfcc_0_mean', 'mfcc_0_std', 'mfcc_1_mean', 'mfcc_1_std', 'mfcc_2_mean', 'mfcc_2_std',
                       'mfcc_3_mean', 'mfcc_3_std', 'mfcc_4_mean', 'mfcc_4_std', 'mfcc_5_mean', 'mfcc_5_std',
                       'mfcc_6_mean', 'mfcc_6_std', 'mfcc_7_mean', 'mfcc_7_std', 'mfcc_8_mean', 'mfcc_8_std',
                       'mfcc_9_mean', 'mfcc_9_std', 'mfcc_10_mean', 'mfcc_10_std', 'mfcc_11_mean', 'mfcc_11_std'])

    with open("audio_feature.csv", "w", newline='') as feature_file:
        writer = csv.writer(feature_file)
        writer.writerows(dataset)
        feature_file.close()


def get_feature(file):
    y, sr = librosa.load(file)

    zcr = librosa.feature.zero_crossing_rate(y=y)
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stft_mean = np.mean(chroma_stft)
    chroma_stft_std = np.std(chroma_stft)

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = np.mean(spectral_centroid)
    spectral_centroid_std = np.std(spectral_centroid)

    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_rolloff_mean = np.mean(spectral_rolloff)
    spectral_rolloff_std = np.std(spectral_rolloff)

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)
    spectral_bandwidth_std = np.std(spectral_bandwidth)

    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfcc_mean = np.array([np.mean(mfcc[i]) for i in range(len(mfcc))])
    mfcc_std = np.array([np.std(mfcc[i]) for i in range(len(mfcc))])

    feature_vector = [
        zcr_mean,
        zcr_std,
        chroma_stft_mean,
        chroma_stft_std,
        spectral_centroid_mean,
        spectral_centroid_std,
        spectral_rolloff_mean,
        spectral_rolloff_std,
        spectral_bandwidth_mean,
        spectral_bandwidth_std,
    ]

    for i in range(0, 12):
        feature_vector.append(mfcc_mean[i])
        feature_vector.append(mfcc_std[i])

    return feature_vector
