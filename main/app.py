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

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

class HateSpeechDetector:
    def __init__(self):
        """Initialize the hate speech detector with pre-trained model and dataset"""
        print("Loading SentenceTransformer model...")
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
        
        # Load dataset kata kasar
        print("Loading hate speech dataset...")
        self.kasar_df = pd.read_csv('kalimat_kasar.csv')
        
        # Filter hanya yang mengandung kata kasar (contains_bad_word = 1)
        self.hate_sentences = self.kasar_df[self.kasar_df['contains_bad_word'] == 1]['sentence'].tolist()
        
        # Encode hate speech sentences
        print("Encoding hate speech sentences...")
        self.hate_embeddings = self.model.encode(self.hate_sentences, convert_to_tensor=True)
        
        print("Hate Speech Detector initialized successfully!")
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove mentions and hashtags (but keep the text)
        text = re.sub(r'[@#]\w+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def detect_hate_speech(self, comments, threshold=0.75):
        """Detect hate speech in list of comments"""
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
        """Categorize severity based on similarity score"""
        if score >= 0.9:
            return "Sangat Tinggi"
        elif score >= 0.8:
            return "Tinggi"
        elif score >= 0.75:
            return "Sedang"
        else:
            return "Rendah"

class SocialMediaScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def extract_comments_from_url(self, url):
        """Extract comments from social media URL"""
        try:
            # Determine platform
            if 'twitter.com' in url or 'x.com' in url:
                return self.scrape_twitter_comments(url)
            elif 'instagram.com' in url:
                return self.scrape_instagram_comments(url)
            elif 'tiktok.com' in url:
                return self.scrape_tiktok_comments(url)
            else:
                return []
        except Exception as e:
            logging.error(f"Error extracting comments: {str(e)}")
            return []
    
    def scrape_twitter_comments(self, url):
        """Scrape Twitter comments (simplified version)"""
        # Note: This is a simplified version. In practice, you'd need Twitter API
        # For demo purposes, we'll return some sample comments
        sample_comments = [
            "Postingan yang bagus sekali!",
            "Setuju banget dengan pendapat ini",
            "Wah keren nih informasinya",
            "Terima kasih sudah berbagi",
            "Sangat bermanfaat sekali"
        ]
        
        # Add some potentially offensive comments for testing
        if random.random() > 0.5:  # 50% chance to include test offensive comments
            sample_comments.extend([
                "Males itu kalo kerja pagi trus gak ada yg nganter, anjing",
                "sok geulis anjing",
                "Lancau anjing.. stressnya aku..",
                "ANJING!! AKUN TOLOL"
            ])
        
        return sample_comments[:10]  # Return max 10 comments
    
    def scrape_instagram_comments(self, url):
        """Scrape Instagram comments (placeholder)"""
        # Instagram requires authentication, so this is a placeholder
        return [
            "Beautiful post! 😍",
            "Love this content",
            "Amazing work!",
            "Thanks for sharing"
        ]
    
    def scrape_tiktok_comments(self, url):
        """Scrape TikTok comments (placeholder)"""
        # TikTok also requires special handling
        return [
            "So funny! 😂",
            "Great video",
            "Love it!",
            "Nice content"
        ]

# Initialize components
detector = HateSpeechDetector()
scraper = SocialMediaScraper()

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