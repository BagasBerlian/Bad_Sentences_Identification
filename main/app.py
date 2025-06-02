from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import re
import requests
from bs4 import BeautifulSoup
import time
import random
from urllib.parse import urlparse
import logging
import re


app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

class HateSpeechDetector:
    def __init__(self):
        print("Loading SentenceTransformer model...")
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
        
        print("Loading hate speech dataset...")
        self.kasar_df = pd.read_csv('kalimat_kasar.csv')
        
        self.hate_sentences = self.kasar_df[self.kasar_df['contains_bad_word'] == 1]['sentence'].tolist()
        
        print("Encoding hate speech sentences...")
        self.hate_embeddings = self.model.encode(self.hate_sentences, convert_to_tensor=True)
        
        print("Hate Speech Detector initialized successfully!")
    
    
    def preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
        
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        text = re.sub(r'[@#]\w+', '', text)
        
        text = ' '.join(text.split())
        
        return text.strip()
    
    def detect_hate_speech(self, comments, threshold=0.9):
        if not comments:
            return []
        
        results = []
        
        # Preprocess comments
        clean_comments = [self.preprocess_text(comment) for comment in comments]
        clean_comments = [c for c in clean_comments if len(c) > 5]  # Filter very short comments
        
        if not clean_comments:
            return results
        
        # Encode input comments
        comment_embeddings = self.model.encode(clean_comments, convert_to_tensor=True)
        
        # Calculate similarities
        for i, embedding in enumerate(comment_embeddings):
            similarities = util.cos_sim(embedding, self.hate_embeddings)[0]
            max_similarity = float(similarities.max())
            
            if max_similarity >= threshold:
                # Find the most similar hate speech example
                best_match_idx = similarities.argmax().item()
                best_match = self.hate_sentences[best_match_idx]
                
                results.append({
                    'comment': clean_comments[i],
                    'original_comment': comments[i] if i < len(comments) else clean_comments[i],
                    'similarity_score': round(max_similarity, 3),
                    'matched_pattern': best_match,
                    'severity': self.get_severity_level(max_similarity)
                })
        
        # Sort by similarity score (highest first)
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results
    
    def get_severity_level(self, score):
        if score >= 0.9:
            return "Sangat Tinggi"
        elif score >= 0.8:
            return "Tinggi"
        elif score >= 0.75:
            return "Sedang"
        else:
            return "Rendah"

class SocialMediaScraper:
    def __init__(self, youtube_api_key=None):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.youtube_api_key = youtube_api_key
    
    def extract_comments_from_url(self, url):
        try:
            # Determine platform
            if 'youtube.com' in url or 'youtu.be' in url:
                return self.scrape_youtube_comments(url)
            else:
                return []
        except Exception as e:
            logging.error(f"Error extracting comments: {str(e)}")
            return []
    
    def scrape_youtube_comments(self, url):
        if not self.youtube_api_key:
            logging.error("YouTube API key belum diset")
            return []
        
        video_id = self.extract_youtube_video_id(url)
        if not video_id:
            logging.error("Gagal mengambil video ID dari URL")
            return []
        
        comments = []
        base_url = "https://www.googleapis.com/youtube/v3/commentThreads"
        
        params = {
            'part': 'snippet',
            'videoId': video_id,
            'key': self.youtube_api_key,
            'maxResults': 100,
            'textFormat': 'plainText'
        }
        
        while True:
            response = requests.get(base_url, params=params, timeout=10)
            if response.status_code != 200:
                logging.error(f"YouTube API error: {response.status_code} {response.text}")
                break
            
            data = response.json()
            items = data.get('items', [])
            
            for item in items:
                top_comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(top_comment)
            
            if 'nextPageToken' in data:
                params['pageToken'] = data['nextPageToken']
            else:
                break
        
        return comments

    def extract_youtube_video_id(self, url):
        parsed = urlparse(url)
        
        if 'youtu.be' in parsed.netloc:
            return parsed.path[1:] 
        elif 'youtube.com' in parsed.netloc:
            qs = dict([param.split('=') for param in parsed.query.split('&') if '=' in param])
            return qs.get('v', None)
        return None
    

YOUTUBE_API_KEY = 'AIzaSyDsOT9hRkxf_M9FnonbZIYqc0OirHIV59s'

# Initialize components
detector = HateSpeechDetector()
scraper = SocialMediaScraper(youtube_api_key=YOUTUBE_API_KEY)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        threshold = float(data.get('threshold', 0.75))
        
        if not url:
            return jsonify({'error': 'URL tidak boleh kosong'}), 400
        
        # Extract comments
        comments = scraper.extract_comments_from_url(url)
        
        if not comments:
            return jsonify({
                'total_comments': 0,
                'hate_comments': 0,
                'results': [],
                'message': 'Tidak ada komentar yang ditemukan atau platform tidak didukung'
            })
        
        # Detect hate speech
        hate_results = detector.detect_hate_speech(comments, threshold)
        
        return jsonify({
            'total_comments': len(comments),
            'hate_comments': len(hate_results),
            'results': hate_results,
            'platform': get_platform_name(url)
        })
        
    except Exception as e:
        logging.error(f"Analysis error: {str(e)}")
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500

@app.route('/test')
def test():
    """Test endpoint dengan sample data"""
    test_comments = [
        "Postingan yang bagus sekali!",
        "Males itu kalo kerja pagi trus gak ada yg nganter, anjing",
        "Setuju banget dengan pendapat ini",
        "sok geulis anjing",
        "Terima kasih sudah berbagi",
        "ANJING!! AKUN TOLOL",
        "Wah keren nih informasinya",
        "Lancau anjing.. stressnya aku.."
    ]
    
    hate_results = detector.detect_hate_speech(test_comments, 0.75)
    
    return jsonify({
        'total_comments': len(test_comments),
        'hate_comments': len(hate_results),
        'results': hate_results,
        'platform': 'Test Data'
    })

def get_platform_name(url):
    """Get platform name from URL"""
    if 'twitter.com' in url or 'x.com' in url:
        return 'Twitter/X'
    elif 'instagram.com' in url:
        return 'Instagram'
    elif 'tiktok.com' in url:
        return 'TikTok'
    else:
        return 'Unknown'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)