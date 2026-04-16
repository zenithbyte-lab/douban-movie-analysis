import re
import json
import jieba
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from scipy.sparse import hstack
from snownlp import SnowNLP

STOPWORDS = set(
    '的 了 是 在 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 '
    '自己 这 他 她 它 们 那 被 从 把 让 用 还 个 可以 对 而 吗 呢 吧 啊 哦 呀 嗯 哈 '
    '可 与 之 中 为 以 于 其 或 但 如 这个 那个 什么 怎么 这样 那样 然后 所以 因为 '
    '如果 虽然 但是 不过 只是 已经 一样 一直 一些 这些 那些 没 更 最 又 再 '
    '电影 故事 一部 影片 一段 两个 之间 开始 最后 一次 其中 以及 通过 成为 '
    '导演 片 中 年'.split()
)


class MLService:
    def __init__(self):
        self.df = None
        self.tfidf = None
        self.svm_model = None
        self.label_encoder = None
        self.tfidf_matrix = None
        self.baseline_results = None
        self.improved_results = None
        self.genre_list = []
        self.analysis_cache = {}
        self._sentiment_cache = {}
        self._word_frequency_cache = {}
        self.is_ready = False

    def init_app(self, movie_data_path, baseline_path, improved_path):
        self.df = pd.read_excel(movie_data_path)
        self._clean_data()
        self._train_model()
        self._load_experiment_results(baseline_path, improved_path)
        self.is_ready = True

    def _clean_data(self):
        self.df['评分'] = pd.to_numeric(self.df['评分'], errors='coerce')
        self.df['评分人数'] = pd.to_numeric(self.df['评分人数'], errors='coerce')
        self.df['上映年份'] = pd.to_numeric(self.df['上映年份'], errors='coerce')

        def parse_duration(val):
            if pd.isna(val):
                return np.nan
            m = re.search(r'(\d+)', str(val))
            return int(m.group(1)) if m else np.nan
        self.df['片长_数值'] = self.df['片长'].apply(parse_duration)

        self.df['首要类型'] = self.df['类型'].apply(
            lambda x: str(x).split('/')[0].strip() if pd.notna(x) else None
        )
        self.df['所有类型'] = self.df['类型'].apply(
            lambda x: [g.strip() for g in str(x).split('/') if g.strip()] if pd.notna(x) else []
        )
        self.df['首要国家'] = self.df['国家/地区'].apply(
            lambda x: str(x).split('/')[0].strip() if pd.notna(x) else None
        )
        self.df['所有国家'] = self.df['国家/地区'].apply(
            lambda x: [c.strip() for c in str(x).split('/') if c.strip()] if pd.notna(x) else []
        )
        self.df['首要导演'] = self.df['导演'].apply(
            lambda x: str(x).split('/')[0].strip() if pd.notna(x) else None
        )
        self.genre_list = sorted(set(g for gl in self.df['所有类型'] for g in gl))

    def _tokenize(self, text):
        if pd.isna(text):
            return ''
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', ' ', str(text))
        words = jieba.lcut(text)
        return ' '.join(w for w in words if len(w) >= 2 and w not in STOPWORDS)

    def _train_model(self):
        valid = self.df.dropna(subset=['剧情简介', '首要类型']).copy()
        top_genres = valid['首要类型'].value_counts().nlargest(10).index.tolist()
        valid = valid[valid['首要类型'].isin(top_genres)].copy()

        genre_map = {
            '剧情': '剧情', '喜剧': '喜剧', '爱情': '喜剧',
            '动作': '动作', '动画': '动画', '纪录片': '纪录片',
            '科幻': '科幻悬疑', '恐怖': '科幻悬疑', '惊悚': '科幻悬疑', '悬疑': '科幻悬疑',
        }
        valid['合并类型'] = valid['首要类型'].map(genre_map)
        valid = valid.dropna(subset=['合并类型'])

        valid['分词文本'] = valid['剧情简介'].apply(self._tokenize)

        self.tfidf = TfidfVectorizer(
            max_features=8000, ngram_range=(1, 2),
            min_df=2, max_df=0.95, sublinear_tf=True
        )
        X_text = self.tfidf.fit_transform(valid['分词文本'])

        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(valid['合并类型'])

        self.svm_model = LinearSVC(C=5.0, class_weight='balanced', max_iter=8000)
        self.svm_model.fit(X_text, y)

        self.tfidf_matrix = self.tfidf.transform(self.df['剧情简介'].fillna('').apply(self._tokenize))

    def _load_experiment_results(self, baseline_path, improved_path):
        try:
            with open(baseline_path, 'r', encoding='utf-8') as f:
                self.baseline_results = json.load(f)
        except Exception:
            self.baseline_results = None
        try:
            with open(improved_path, 'r', encoding='utf-8') as f:
                self.improved_results = json.load(f)
        except Exception:
            self.improved_results = None

    def predict_genre(self, text):
        tokenized = self._tokenize(text)
        X = self.tfidf.transform([tokenized])
        decision = self.svm_model.decision_function(X)[0]
        exp_scores = np.exp(decision - np.max(decision))
        probs = exp_scores / exp_scores.sum()
        pred_idx = np.argmax(probs)
        classes = self.label_encoder.classes_
        result = {
            'predicted': classes[pred_idx],
            'confidence': float(probs[pred_idx]),
            'all_scores': {cls: float(p) for cls, p in zip(classes, probs)}
        }
        return result

    def get_similar_movies(self, movie_idx, n=10):
        from sklearn.metrics.pairwise import cosine_similarity
        if movie_idx >= self.tfidf_matrix.shape[0]:
            return []
        sim = cosine_similarity(self.tfidf_matrix[movie_idx], self.tfidf_matrix).flatten()
        sim[movie_idx] = -1
        top_indices = sim.argsort()[-n:][::-1]
        results = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            results.append({
                'movie_id': int(row.get('movie_id', 0)),
                'name': str(row.get('电影名称', '')),
                'rating': float(row['评分']) if pd.notna(row['评分']) else None,
                'genre': str(row.get('类型', '')),
                'year': int(row['上映年份']) if pd.notna(row['上映年份']) else None,
                'similarity': float(sim[idx]),
            })
        return results

    def analyze_sentiment(self, text):
        if not text or pd.isna(text):
            return {'score': 0.5, 'label': '中性'}
        cache_key = str(text)
        if cache_key in self._sentiment_cache:
            return dict(self._sentiment_cache[cache_key])
        try:
            score = SnowNLP(cache_key).sentiments
        except Exception:
            score = 0.5
        if score > 0.6:
            label = '正面'
        elif score < 0.4:
            label = '负面'
        else:
            label = '中性'
        result = {'score': round(score, 4), 'label': label}
        self._sentiment_cache[cache_key] = result
        return dict(result)

    def get_word_frequencies(self, n=200):
        from collections import Counter
        if n in self._word_frequency_cache:
            return list(self._word_frequency_cache[n])
        all_words = []
        for text in self.df['剧情简介'].dropna():
            tokenized = self._tokenize(text)
            all_words.extend(tokenized.split())
        counter = Counter(all_words)
        freqs = counter.most_common(n)
        self._word_frequency_cache[n] = freqs
        return list(freqs)


ml_service = MLService()
