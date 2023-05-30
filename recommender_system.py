import pickle
import pandas as pd
from feature_extraction import get_feature
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


def get_song_genre(src):
    model_file = open('model.pkl', 'rb')
    model = pickle.load(model_file, encoding='bytes')

    feature_vector = get_feature(src)
    feature_vector_ = feature_vector.copy()
    for i in [33, 31, 30, 26, 18, 16, 14, 5, 4, 0]:
        del feature_vector_[i]
    
    prediction = model.predict([feature_vector_])

    decode = {1: 'Ambient', 2: 'Classical', 3: 'Country - Folk', 4: 'Jazz - Blues', 5: 'RnB - Soul'}
    genre = decode[prediction[0]]

    return genre, feature_vector


def find_similar_songs(src, filename):
    data = pd.read_csv('audio_feature.csv', index_col='filename')

    genre, feature_vector = get_song_genre(src)
    
    filtered_data = data[data.genre == genre]
    if filename in filtered_data.index:
        filtered_data = filtered_data.drop(filename)
    labels = filtered_data[['genre']]
    filtered_data = filtered_data.drop(columns='genre')

    scaler = StandardScaler()
    scaler.fit(filtered_data.values)
    filtered_data = scaler.transform(filtered_data.values)
    feature_vector = scaler.transform([feature_vector])

    similarity = cosine_similarity(filtered_data, feature_vector)

    sim_df = pd.DataFrame(similarity).set_index(labels.index)
    sim_df.columns = ['similarity']

    series = sim_df['similarity'].sort_values(ascending=False)
    series = series.head(5)

    return genre, series
