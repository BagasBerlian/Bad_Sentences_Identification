# pylint: disable=all
# type: ignore
# noqa

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
from dotenv import load_dotenv
import os

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")


app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

class HateSpeechDetector:
    def __init__(self):
        print("Loading SentenceTransformer model...")
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
        
        print("Loading hate speech dataset...")
        self.kasar_df = pd.read_csv('kalimat_kasar.csv')
        self.hate_sentences = self.kasar_df[self.kasar_df['contains_bad_word'] == 1]['sentence'].tolist()
        self.hate_sentences = self.filter_hate_sentences(self.hate_sentences)
        print("Encoding hate speech sentences...")
        self.hate_embeddings = self.model.encode(self.hate_sentences, convert_to_tensor=True)
        
        self.positive_indicators = [
            # Umum
            'bagus', 'baik', 'benar', 'setuju', 'mantap', 'keren', 'hebat', 'luar biasa',
            'wow', 'keren banget', 'mantul', 'top', 'oke', 'ok', 'sip', 'jos', 'gaskeun',
            'yes', 'betul', 'bener', 'jujur', 'asli', 'terpercaya',

            # Apresiasi / Ucapan terima kasih
            'terima kasih', 'makasih', 'thanks', 'thank you', 'terimakasih banyak', 
            'alhamdulillah', 'syukur', 'puji syukur',

            # Emosi positif
            'senang', 'gembira', 'bahagia', 'suka', 'love', 'happy', 'enjoy', 
            'terharu', 'tersentuh', 'semangat', 'antusias', 'terinspirasi', 'respect',

            # Penilaian / Pujian
            'cantik', 'tampan', 'ganteng', 'beautiful', 'handsome', 'cakep',
            'bagus banget', 'keren banget', 'super', 'amazing', 'perfect', 'excellent',
            'terbaik', 'very good', 'recommended', 'worth it', 'bermanfaat', 'menginspirasi',
            'edukatif', 'informatif', 'berguna', 'top markotop', 'masyaallah', 'subhanallah',

            # Bahasa gaul/slang positif
            'kereeen', 'mantaaap', 'sipp', 'asli keren', 'goks', 'pecah', 'epic', 'gila keren',
            'niat banget', 'niat bgt', 'nice one', 'solid', 'dahsyat', 'gg', 'op banget', 'worth banget',

            # Umum tambahan
            'like', 'favorit', 'best', 'bagus lah', 'bagus sih', 'positif banget', 'powerful',
            'terdepan', 'recommended banget', 'paling keren', 'paling bagus'
        ]
        
        print("Hate Speech Detector initialized successfully!")
    
    def filter_hate_sentences(self, sentences):
        """Filter hate sentences untuk menghapus yang tidak relevan"""
        filtered = []
        for sentence in sentences:
            if len(sentence.split()) < 3:
                continue
            
            lower_sentence = sentence.lower()
            if any(keyword in lower_sentence for keyword in [
                # Konteks produk / promosi
                'jual', 'beli', 'harga', 'diskon', 'promo', 'gratis', 'order', 'pesan sekarang',
                'ready stock', 'stok tersedia', 'preorder', 'tersedia', 'limited edition', 
                'produk terbaru', 'produk unggulan', 'paket hemat', 'official store',

                # Hewan / medis / veteriner
                'makanan anjing', 'dog food', 'petshop', 'hewan peliharaan', 'hewan ternak',
                'vaksin', 'dokter hewan', 'veteriner', 'klinik hewan', 'rawat inap', 'steril',
                'resep dokter', 'obat hewan', 'vitamin', 'grooming hewan',

                # Konteks edukasi / netral
                'fakta', 'penelitian', 'data statistik', 'survey', 'kajian ilmiah',
                'artikel edukasi', 'konten edukatif', 'pembelajaran', 'materi pelajaran',
                'penyuluhan', 'sosialisasi', 'informasi penting', 'info kesehatan',

                # Umum lainnya yang netral / non-hate
                'tips', 'trik', 'cara', 'tutorial', 'rekomendasi', 'ulasan produk', 'review jujur',
                'how to', 'panduan', 'video ini membahas', 'channel ini', 'konten ini', 'gunakan dengan bijak',
                'mari kita pelajari', 'untuk pemula', 'untuk anak-anak', 'untuk hewan',

                # Brand atau kata-kata ambigu yang bukan hate
                'guinness', 'netflix', 'shopee', 'tokopedia', 'lazada', 'blibli', 'ecommerce', 
                'whiskas', 'royal canin', 'pedigree', 'purina'
            ]):
                continue
                
            filtered.append(sentence)
        
        return filtered
    
    def preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
        
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'[@#]\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        
        return text.strip()
    
    def is_likely_positive(self, comment):
        lower_comment = comment.lower()
        return any(positive in lower_comment for positive in self.positive_indicators)
    
    def has_sufficient_context(self, comment):
        words = comment.split()
        
        if len(words) < 3:
            return False
            
        if all(len(word) <= 2 and not word.isalnum() for word in words):
            return False
            
        generic_patterns = [
            r'^(ya|iya|yah|yoi|sip|ok+|oke+|baik+|yap+|yes+|no+|nggak+|tidak+)$',
            r'^(wah+|wow+|wooow+|hebat+|mantap+|keren+|bagus+|top+|jos+|asik+|sip+)$',
            r'^(haha+|hihi+|hehe+|wkwk+|wk+w+|lol+|lmao+)$',
            r'^(first|pertama|kedua|ketiga|keempat|nomor\s+\d+)$',
            r'^(nice+|good+|great+|amazing+|cool+|beautiful+|lovely+|perfect+)$',
            r'^(terima\s+kasih|makasih|thanks+|thank\s+you+|tq+|arigato+)$',
            r'^(mantap\s*(banget|sekali|betul)?|bagus\s*(banget|sekali)?|keren\s*(abis|banget)?)$',
            r'^((video|kontennya)\s*(bagus|keren|mantap|hebat))$',
            r'^((salam|salam\s+hormat|assalamualaikum|salam\s+sejahtera).*)$',
            r'^(semangat|lanjutkan|tetap\s+semangat|good\s+luck|keep\s+going)$',
            r'^(sukses\s+selalu|maju\s+terus|teruskan|lanjutkan)$',
            r'^(test|tes|cek|check|123|321)$',
            r'^(\d{1,2}/\d{1,2}/\d{2,4})$',  
            r'^(\d+)$',  
        ]
        
        comment_lower = comment.lower().strip()
        if any(re.match(pattern, comment_lower) for pattern in generic_patterns):
            return False
            
        return True
    
    def detect_hate_speech(self, comments, threshold=0.85):
        if not comments:
            return []
        
        results = []
        clean_comments = [self.preprocess_text(comment) for comment in comments]
        filtered_pairs = []
        for i, (original, clean) in enumerate(zip(comments, clean_comments)):
            if (len(clean) > 10 and 
                self.has_sufficient_context(clean) and 
                not self.is_likely_positive(clean)):
                filtered_pairs.append((i, original, clean))
        
        if not filtered_pairs:
            return results
        
        indices, original_comments, clean_comments_filtered = zip(*filtered_pairs)
        comment_embeddings = self.model.encode(clean_comments_filtered, convert_to_tensor=True)
        
        for i, embedding in enumerate(comment_embeddings):
            similarities = util.cos_sim(embedding, self.hate_embeddings)[0]
            max_similarity = float(similarities.max())
            
            adjusted_threshold = max(threshold, 0.88) 
            
            if max_similarity >= adjusted_threshold:
                best_match_idx = similarities.argmax().item()
                best_match = self.hate_sentences[best_match_idx]
                
                if self.validate_match(clean_comments_filtered[i], best_match, max_similarity):
                    results.append({
                        'comment': clean_comments_filtered[i],
                        'original_comment': original_comments[i],
                        'similarity_score': round(max_similarity, 3),
                        'matched_pattern': best_match,
                        'severity': self.get_severity_level(max_similarity),
                        'confidence': self.calculate_confidence(max_similarity, clean_comments_filtered[i])
                    })
        
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results
    
    def validate_match(self, comment, matched_sentence, similarity_score):
        comment_lower = comment.lower()
        matched_lower = matched_sentence.lower()
        
        hate_keywords = [
            # Kata makian kasar umum
            'anjing', 'babi', 'bangsat', 'bajingan', 'brengsek', 'tolol', 'goblok', 'bodoh', 'idiot',
            'kontol', 'memek', 'titit', 'pelacur', 'lonte', 'tai', 'sial', 'sialan', 'asu', 'anjrit',
            'monyet', 'kunyuk', 'celeng', 'keparat', 'setan', 'iblis',

            # Kata penghinaan agama, ras, kelompok
            'kafir', 'cina', 'kadrun', 'cebong', 'kampret', 'jancuk', 'gundik', 'parasit', 'binatang',

            # Kata bernuansa seksual/pelecehan
            'gatel', 'genit', 'mesum', 'ngentot', 'binal', 'birahi', 'cabul',

            # Kata alternatif/ejaan umum yang sering disamarkan (obfuscated)
            'anjg', 'b4bi', 'b4ngs4t', 'p3lacur', 'k0ntol', 'mem3k', 'g0blok', 'gblk',
            's!al', 't*l*l', 'b*d*h', 'g*bl*k', 'ng3ntot', 'b*ngs*t', 'b*j*ng*n', 'br*ngs*k', 'k*nt*l'
        ]
        
        comment_has_hate = any(keyword in comment_lower for keyword in hate_keywords)
        matched_has_hate = any(keyword in matched_lower for keyword in hate_keywords)
        
        if similarity_score > 0.9 and not comment_has_hate and matched_has_hate:
            return False
            
        return True
    
    def calculate_confidence(self, similarity_score, comment):
        base_confidence = similarity_score
        
        hate_keywords = ['anjing', 'bangsat', 'tolol', 'goblok', 'asu', 'babi']
        if any(keyword in comment.lower() for keyword in hate_keywords):
            base_confidence = min(1.0, base_confidence + 0.05)
        
        if len(comment.split()) < 5:
            base_confidence *= 0.9
            
        return round(base_confidence, 3)
    
    def get_severity_level(self, score):
        if score >= 0.95:
            return "Sangat Tinggi"
        elif score >= 0.9:
            return "Tinggi"
        elif score >= 0.85:
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
        threshold = float(data.get('threshold', 0.85)) 
        
        if not url:
            return jsonify({'error': 'URL tidak boleh kosong'}), 400
        
        comments = scraper.extract_comments_from_url(url)
        
        if not comments:
            return jsonify({
                'total_comments': 0,
                'hate_comments': 0,
                'results': [],
                'message': 'Tidak ada komentar yang ditemukan atau platform tidak didukung'
            })
        
        hate_results = detector.detect_hate_speech(comments, threshold)
        
        return jsonify({
            'total_comments': len(comments),
            'hate_comments': len(hate_results),
            'results': hate_results,
            'platform': get_platform_name(url),
            'filtered_comments': len([c for c in comments if detector.has_sufficient_context(detector.preprocess_text(c))])
        })
        
    except Exception as e:
        logging.error(f"Analysis error: {str(e)}")
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500

@app.route('/test')
def test():
    test_comments = [
        "Postingan yang bagus sekali!", 
        "Males itu kalo kerja pagi trus gak ada yg nganter, anjing",  
        "Setuju banget dengan pendapat ini",  
        "sok geulis anjing",  
        "Terima kasih sudah berbagi informasi",  
        "ANJING!! AKUN TOLOL",  
        "Wah keren nih informasinya mantap",  
        "Lancau anjing.. stressnya aku..",  
        "Betul juga",  
        "Ya",  
        "Mantap",  
        "Video bagus sekali, terima kasih telah berbagi",  
        "Dasar bangsat tolol goblok anjing",  
    ]
    
    hate_results = detector.detect_hate_speech(test_comments, 0.85)
    
    return jsonify({
        'total_comments': len(test_comments),
        'hate_comments': len(hate_results),
        'results': hate_results,
        'platform': 'Test Data',
        'test_info': 'Improved filtering applied'
    })

def get_platform_name(url):
    if 'youtube.com' in url or 'youtu.be' in url:
        return 'YouTube'
    elif 'twitter.com' in url or 'x.com' in url:
        return 'Twitter/X'
    elif 'instagram.com' in url:
        return 'Instagram'
    elif 'tiktok.com' in url:
        return 'TikTok'
    else:
        return 'Unknown'

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
    # app.run(debug=True, host='0.0.0.0', port=5000)