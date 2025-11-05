import pandas as pd
from flask import Flask, jsonify, request
import numpy as np

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
def get_recommendations(mood_id: int):
    if df_mood.empty:
        return ["Data not loaded. Check server logs."]

    songs_in_mood  = df_mood[df_mood[CLUSTER_COLUMN] == mood_id]

    if songs_in_mood.empty:
        return [f"No songs found for cluster: {mood_id}"]

    # Select 5 random sample of songs
    recommended_songs = songs_in_mood.sample(5)[['song_name', 'uri']]

    return recommended_songs

# --- 3. Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_mood_json = request.get_json()
        selected_mood: int = selected_mood_json['mood']

        recommendations = get_recommendations(selected_mood)

    return jsonify(recommendations.to_dict('records'))

if __name__ == '__main__':
    app.run(debug=True) 
