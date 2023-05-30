from flask import Flask, request, render_template
from recommender_system import find_similar_songs
import pandas as pd
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'

data = pd.read_csv('audio_feature.csv')


@app.route('/')
def index():
    return render_template('index.html', music_list=data.filename)


@app.route('/play/<filename>')
def play(filename=''):
    filename = filename
    genre = data.loc[data['filename'] == filename].values[0][1]
    src = f'static/Soothing Music/{genre}/{filename}.wav'

    genre, series = find_similar_songs(src, filename)

    recommendation_list = [x for x in series.index]

    return render_template('music-player.html', recommendation_list=recommendation_list, genre=genre, src=src, filename=filename)


@app.route('/play-music', methods=['GET', 'POST'])
def play_music():
    if request.method == 'POST':
        if request.files:
            audio = request.files['audio']

            filename = os.path.splitext(audio.filename)[0]
            src = f'static/uploads/{filename}.wav'

            audio.save(src)

            genre, series = find_similar_songs(src, filename)

            recommendation_list = [x for x in series.index]

            return render_template('music-player.html', recommendation_list=recommendation_list, src=src, filename=filename, genre=genre)

    return render_template('music-player.html', recommendation_list=data.filename, src='', filename='', genre='')


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)