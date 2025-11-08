import pandas as pd
from flask import Flask, jsonify, request
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# --- 1. Load Data ---
try:
    # Use .pkl for faster and more memory-efficient loading than CSV
    df_mood = pd.read_pickle("data/clustered_df.pkl")
    CLUSTER_COLUMN = 'cluster'
    
except FileNotFoundError:
    print("Error: filtered_by_mood.pkl not found. Check your data/ directory.")
    df_mood = pd.DataFrame() # Create an empty DataFrame to prevent app crash

# --- 2. Recommendation Logic ---
def get_songs(mood_id: int):
    if df_mood.empty:
        return ["Data not loaded. Check server logs."]

    songs_in_mood  = df_mood[df_mood[CLUSTER_COLUMN] == mood_id]

    if songs_in_mood.empty:
        return [f"No songs found for cluster: {mood_id}"]

    # Select 5 random sample of songs
    recommended_songs = songs_in_mood.sample(5)[['song_name', 'uri']]

    return recommended_songs

def get_similar_songs(song):
    if song['song_name'] in df_mood['song_name'].values:
        song_features = df_mood.loc[df_mood['song_name'] == song['song_name'], ['energy', 'valence']].values.flatten()
        all_song_features = df_mood[['energy', 'valence']].values

        similarities = cosine_similarity([song_features], all_song_features)
        sim_songs = pd.DataFrame({'song_name': df_mood['song_name'], 'similarity': similarities.flatten(), 'uri': df_mood['uri']})
        sim_songs = sim_songs.sort_values(by = 'similarity', ascending=False).reset_index(drop=True)
        sim_songs = sim_songs[sim_songs['song_name'] != song['song_name']]
        top_sim_song = sim_songs.head(3)
    
    return top_sim_song
# --- 3. Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    songs = pd.Series([])
    if request.method == 'POST':
        selected_mood_json = request.get_json()
        selected_mood: int = selected_mood_json['mood']

        songs = get_songs(selected_mood)

    return jsonify(songs.to_dict('records'))

@app.route('/song', methods=['POST'])
def songs():
    song = pd.Series()

    selected_song = request.get_json()
    sim_songs = get_similar_songs(selected_song)

    return jsonify(sim_songs.to_dict('records'))
if __name__ == '__main__':
    app.run(debug=True) 
