from flask import Flask, request, render_template
import pandas as pd
import snscrape.modules.twitter as sntwitter
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Load model dan dataset kasar
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
kasar_df = pd.read_csv('kalimat_kasar.csv')
kasar_texts = kasar_df['text'].dropna().astype(str).str.strip().tolist()
kasar_embeddings = model.encode(kasar_texts, convert_to_tensor=True)

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    if request.method == "POST":
        url = request.form["url"]
        tweet_id = url.strip().split("/")[-1]

        # Ambil komentar dari Tweet
        query = f"conversation_id:{tweet_id}"
        comments = []
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
            if tweet.inReplyToTweetId == int(tweet_id):
                comments.append(tweet.content)
            if i > 50: break  # batasi 50 komentar

        # Proses deteksi
        input_embeddings = model.encode(comments, convert_to_tensor=True)
        threshold = 0.7
        for i, emb in enumerate(input_embeddings):
            similarities = util.cos_sim(emb, kasar_embeddings)[0]
            max_sim = float(similarities.max())
            if max_sim >= threshold:
                results.append({
                    'komentar': comments[i],
                    'similarity': round(max_sim, 3)
                })

    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
