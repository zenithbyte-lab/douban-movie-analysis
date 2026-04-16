import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request
from flask_login import login_required
from ml_service import ml_service

api_bp = Blueprint('api', __name__, url_prefix='/api')


def safe_int(val, default=0):
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def safe_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


@api_bp.route('/dashboard-stats')
@login_required
def dashboard_stats():
    df = ml_service.df
    total = len(df)
    avg_rating = round(df['评分'].mean(), 2) if df['评分'].notna().any() else 0
    total_votes = int(df['评分人数'].sum()) if df['评分人数'].notna().any() else 0
    rated_count = int(df['评分'].notna().sum())

    all_genres = []
    for g_list in df['所有类型']:
        all_genres.extend(g_list)
    genre_counts = pd.Series(all_genres).value_counts()

    rating_bins = pd.cut(df['评分'].dropna(), bins=np.arange(0, 10.5, 0.5))
    rating_dist = rating_bins.value_counts().sort_index()

    top_movies = df.nlargest(10, '评分')[['电影名称', '评分', '评分人数', '类型', '上映年份']].to_dict('records')

    year_counts = df['上映年份'].dropna().astype(int).value_counts().sort_index()
    recent_years = year_counts.tail(20)

    return jsonify({
        'total_movies': total,
        'avg_rating': avg_rating,
        'total_votes': total_votes,
        'rated_count': rated_count,
        'genre_count': len(genre_counts),
        'genre_distribution': {
            'labels': genre_counts.head(15).index.tolist(),
            'values': genre_counts.head(15).values.tolist(),
        },
        'rating_distribution': {
            'labels': [str(b) for b in rating_dist.index],
            'values': rating_dist.values.tolist(),
        },
        'top_movies': top_movies,
        'year_trend': {
            'labels': recent_years.index.tolist(),
            'values': recent_years.values.tolist(),
        },
    })


@api_bp.route('/movies')
@login_required
def movies_list():
    df = ml_service.df
    page = safe_int(request.args.get('page'), 1)
    per_page = safe_int(request.args.get('per_page'), 20)
    search = request.args.get('search', '').strip()
    genre = request.args.get('genre', '').strip()
    sort_by = request.args.get('sort', '评分')
    order = request.args.get('order', 'desc')

    filtered = df.copy()
    if search:
        mask = filtered['电影名称'].str.contains(search, case=False, na=False)
        mask |= filtered['导演'].str.contains(search, case=False, na=False)
        mask |= filtered['主演'].str.contains(search, case=False, na=False)
        filtered = filtered[mask]
    if genre:
        filtered = filtered[filtered['类型'].str.contains(genre, na=False)]

    if sort_by in filtered.columns:
        filtered = filtered.sort_values(sort_by, ascending=(order == 'asc'), na_position='last')

    total = len(filtered)
    start = (page - 1) * per_page
    page_data = filtered.iloc[start:start + per_page]

    movies = []
    for _, row in page_data.iterrows():
        movies.append({
            'index': int(row.name),
            'movie_id': safe_int(row.get('movie_id')),
            'name': str(row.get('电影名称', '')),
            'rating': safe_float(row.get('评分')) if pd.notna(row.get('评分')) else None,
            'votes': safe_int(row.get('评分人数')),
            'year': safe_int(row.get('上映年份')) if pd.notna(row.get('上映年份')) else None,
            'duration': str(row.get('片长', '')),
            'genre': str(row.get('类型', '')),
            'director': str(row.get('导演', '')),
            'actors': str(row.get('主演', '')),
            'country': str(row.get('国家/地区', '')),
            'summary': str(row.get('剧情简介', ''))[:200],
        })

    return jsonify({
        'data': movies,
        'total': total,
        'page': page,
        'per_page': per_page,
        'total_pages': (total + per_page - 1) // per_page,
        'genres': ml_service.genre_list,
    })


@api_bp.route('/rating-analysis')
@login_required
def rating_analysis():
    df = ml_service.df
    ratings = df['评分'].dropna()

    histogram_bins = np.arange(0, 10.1, 0.5)
    hist_counts, _ = np.histogram(ratings, bins=histogram_bins)
    hist_labels = [f'{histogram_bins[i]:.1f}-{histogram_bins[i+1]:.1f}' for i in range(len(hist_counts))]

    stats = {
        'mean': round(ratings.mean(), 3),
        'median': round(ratings.median(), 3),
        'std': round(ratings.std(), 3),
        'min': round(ratings.min(), 3),
        'max': round(ratings.max(), 3),
        'q25': round(ratings.quantile(0.25), 3),
        'q75': round(ratings.quantile(0.75), 3),
    }

    genre_rating = {}
    for _, row in df.dropna(subset=['评分']).iterrows():
        for g in row['所有类型']:
            genre_rating.setdefault(g, []).append(row['评分'])
    genre_box = {}
    for g, vals in sorted(genre_rating.items(), key=lambda x: -len(x[1]))[:12]:
        arr = np.array(vals)
        genre_box[g] = {
            'min': round(float(arr.min()), 2),
            'q1': round(float(np.percentile(arr, 25)), 2),
            'median': round(float(np.median(arr)), 2),
            'q3': round(float(np.percentile(arr, 75)), 2),
            'max': round(float(arr.max()), 2),
            'mean': round(float(arr.mean()), 2),
            'count': len(vals),
        }

    year_rating = df.dropna(subset=['评分', '上映年份']).groupby('上映年份')['评分'].agg(['mean', 'count'])
    year_rating = year_rating[year_rating['count'] >= 5].tail(30)

    votes_rating = df.dropna(subset=['评分', '评分人数']).copy()
    votes_rating['评分人数_log'] = np.log10(votes_rating['评分人数'].clip(lower=1))
    scatter_sample = votes_rating.sample(min(500, len(votes_rating)), random_state=42)

    return jsonify({
        'histogram': {'labels': hist_labels, 'values': hist_counts.tolist()},
        'stats': stats,
        'genre_box': genre_box,
        'year_rating': {
            'years': year_rating.index.astype(int).tolist(),
            'means': year_rating['mean'].round(2).tolist(),
            'counts': year_rating['count'].astype(int).tolist(),
        },
        'scatter': {
            'data': [[round(r['评分人数_log'], 2), round(r['评分'], 2)]
                     for _, r in scatter_sample.iterrows()],
        },
    })


@api_bp.route('/genre-analysis')
@login_required
def genre_analysis():
    df = ml_service.df

    all_genres = []
    for g_list in df['所有类型']:
        all_genres.extend(g_list)
    genre_counts = pd.Series(all_genres).value_counts()

    genre_year = {}
    for _, row in df.dropna(subset=['上映年份']).iterrows():
        year = int(row['上映年份'])
        for g in row['所有类型']:
            genre_year.setdefault(g, {}).setdefault(year, 0)
            genre_year[g][year] += 1
    top_genres_list = genre_counts.head(8).index.tolist()
    years_range = sorted(set(int(y) for y in df['上映年份'].dropna()))
    recent_years = [y for y in years_range if y >= max(years_range) - 25]
    heatmap_data = []
    for gi, g in enumerate(top_genres_list):
        for yi, y in enumerate(recent_years):
            count = genre_year.get(g, {}).get(y, 0)
            heatmap_data.append([yi, gi, count])

    genre_combo = {}
    for g_list in df['所有类型']:
        if len(g_list) >= 2:
            for i in range(len(g_list)):
                for j in range(i + 1, len(g_list)):
                    pair = tuple(sorted([g_list[i], g_list[j]]))
                    genre_combo[pair] = genre_combo.get(pair, 0) + 1
    top_combos = sorted(genre_combo.items(), key=lambda x: -x[1])[:20]

    genre_rating_avg = {}
    for _, row in df.dropna(subset=['评分']).iterrows():
        for g in row['所有类型']:
            genre_rating_avg.setdefault(g, []).append(row['评分'])
    genre_avg = {g: round(np.mean(v), 2) for g, v in genre_rating_avg.items() if len(v) >= 10}
    genre_avg_sorted = dict(sorted(genre_avg.items(), key=lambda x: -x[1])[:15])

    return jsonify({
        'distribution': {
            'labels': genre_counts.head(20).index.tolist(),
            'values': genre_counts.head(20).values.tolist(),
        },
        'heatmap': {
            'genres': top_genres_list,
            'years': recent_years,
            'data': heatmap_data,
        },
        'combos': {
            'labels': [f'{a}+{b}' for (a, b), _ in top_combos],
            'values': [c for _, c in top_combos],
        },
        'avg_rating': {
            'labels': list(genre_avg_sorted.keys()),
            'values': list(genre_avg_sorted.values()),
        },
    })


@api_bp.route('/director-analysis')
@login_required
def director_analysis():
    df = ml_service.df

    director_stats = {}
    for _, row in df.dropna(subset=['导演', '评分']).iterrows():
        directors = [d.strip() for d in str(row['导演']).split('/') if d.strip()]
        for d in directors:
            director_stats.setdefault(d, {'ratings': [], 'votes': 0, 'movies': 0, 'genres': []})
            director_stats[d]['ratings'].append(row['评分'])
            director_stats[d]['votes'] += safe_int(row.get('评分人数'))
            director_stats[d]['movies'] += 1
            director_stats[d]['genres'].extend(row['所有类型'])

    top_by_count = sorted(director_stats.items(), key=lambda x: -x[1]['movies'])[:20]
    top_count = [{'name': d, 'count': s['movies'], 'avg_rating': round(np.mean(s['ratings']), 2)}
                 for d, s in top_by_count]

    qualified = {d: s for d, s in director_stats.items() if s['movies'] >= 3}
    top_by_rating = sorted(qualified.items(), key=lambda x: -np.mean(x[1]['ratings']))[:20]
    top_rating = [{'name': d, 'avg_rating': round(np.mean(s['ratings']), 2),
                   'count': s['movies'], 'total_votes': s['votes']}
                  for d, s in top_by_rating]

    top_by_votes = sorted(director_stats.items(), key=lambda x: -x[1]['votes'])[:20]
    top_votes = [{'name': d, 'total_votes': s['votes'], 'count': s['movies'],
                  'avg_rating': round(np.mean(s['ratings']), 2)}
                 for d, s in top_by_votes]

    dir_genre = {}
    for d, s in list(qualified.items())[:50]:
        genre_count = pd.Series(s['genres']).value_counts().head(3).to_dict()
        dir_genre[d] = genre_count

    return jsonify({
        'top_by_count': top_count,
        'top_by_rating': top_rating,
        'top_by_votes': top_votes,
        'director_genre': dir_genre,
    })


@api_bp.route('/region-analysis')
@login_required
def region_analysis():
    df = ml_service.df

    all_countries = []
    for c_list in df['所有国家']:
        all_countries.extend(c_list)
    country_counts = pd.Series(all_countries).value_counts()

    country_rating = {}
    for _, row in df.dropna(subset=['评分']).iterrows():
        for c in row['所有国家']:
            country_rating.setdefault(c, []).append(row['评分'])
    country_avg = {c: round(np.mean(v), 2) for c, v in country_rating.items() if len(v) >= 10}
    country_avg_sorted = dict(sorted(country_avg.items(), key=lambda x: -x[1])[:20])

    country_genre = {}
    for _, row in df.iterrows():
        for c in row['所有国家']:
            for g in row['所有类型']:
                country_genre.setdefault(c, {}).setdefault(g, 0)
                country_genre[c][g] += 1
    top_countries = country_counts.head(8).index.tolist()
    all_g = sorted(set(g for c in top_countries for g in country_genre.get(c, {}).keys()))[:10]
    cg_matrix = []
    for c in top_countries:
        cg_matrix.append([country_genre.get(c, {}).get(g, 0) for g in all_g])

    country_year = {}
    for _, row in df.dropna(subset=['上映年份']).iterrows():
        year = int(row['上映年份'])
        for c in row['所有国家']:
            country_year.setdefault(c, {}).setdefault(year, 0)
            country_year[c][year] += 1

    years_all = sorted(set(int(y) for y in df['上映年份'].dropna()))
    recent_20 = [y for y in years_all if y >= max(years_all) - 20]
    top5_countries = country_counts.head(5).index.tolist()
    cy_trends = {}
    for c in top5_countries:
        cy_trends[c] = [country_year.get(c, {}).get(y, 0) for y in recent_20]

    return jsonify({
        'distribution': {
            'labels': country_counts.head(20).index.tolist(),
            'values': country_counts.head(20).values.tolist(),
        },
        'avg_rating': {
            'labels': list(country_avg_sorted.keys()),
            'values': list(country_avg_sorted.values()),
        },
        'country_genre': {
            'countries': top_countries,
            'genres': all_g,
            'matrix': cg_matrix,
        },
        'country_year': {
            'years': recent_20,
            'countries': top5_countries,
            'trends': cy_trends,
        },
    })


@api_bp.route('/year-trends')
@login_required
def year_trends():
    df = ml_service.df
    valid = df.dropna(subset=['上映年份']).copy()
    valid['上映年份'] = valid['上映年份'].astype(int)

    year_count = valid.groupby('上映年份').size()
    year_rating = valid.dropna(subset=['评分']).groupby('上映年份')['评分'].mean().round(2)
    year_votes = valid.dropna(subset=['评分人数']).groupby('上映年份')['评分人数'].mean().round(0)
    year_duration = valid.dropna(subset=['片长_数值']).groupby('上映年份')['片长_数值'].mean().round(1)

    year_genre = {}
    top_g = pd.Series([g for gl in df['所有类型'] for g in gl]).value_counts().head(6).index.tolist()
    for _, row in valid.iterrows():
        y = int(row['上映年份'])
        for g in row['所有类型']:
            if g in top_g:
                year_genre.setdefault(g, {}).setdefault(y, 0)
                year_genre[g][y] += 1

    all_years = sorted(year_count.index)
    recent = [y for y in all_years if y >= max(all_years) - 30]

    genre_trends = {}
    for g in top_g:
        genre_trends[g] = [year_genre.get(g, {}).get(y, 0) for y in recent]

    return jsonify({
        'years': recent,
        'count': [int(year_count.get(y, 0)) for y in recent],
        'avg_rating': [float(year_rating.get(y, 0)) for y in recent],
        'avg_votes': [float(year_votes.get(y, 0)) for y in recent],
        'avg_duration': [float(year_duration.get(y, 0)) for y in recent],
        'genre_trends': genre_trends,
        'top_genres': top_g,
    })


@api_bp.route('/sentiment-analysis')
@login_required
def sentiment_analysis():
    cached = ml_service.analysis_cache.get('sentiment_analysis')
    if cached is not None:
        return jsonify(cached)

    df = ml_service.df
    sample = df.dropna(subset=['剧情简介']).sample(min(200, len(df)), random_state=42)

    results = []
    score_dist = {'正面': 0, '中性': 0, '负面': 0}
    genre_sentiment = {}
    year_sentiment = {}

    for _, row in sample.iterrows():
        sa = ml_service.analyze_sentiment(row['剧情简介'])
        score_dist[sa['label']] += 1
        results.append({
            'name': str(row.get('电影名称', '')),
            'score': sa['score'],
            'label': sa['label'],
            'rating': safe_float(row.get('评分')) if pd.notna(row.get('评分')) else None,
        })
        for g in row['所有类型']:
            genre_sentiment.setdefault(g, []).append(sa['score'])
        if pd.notna(row.get('上映年份')):
            y = int(row['上映年份'])
            year_sentiment.setdefault(y, []).append(sa['score'])

    genre_avg = {g: round(np.mean(v), 3) for g, v in genre_sentiment.items() if len(v) >= 5}
    genre_avg_sorted = dict(sorted(genre_avg.items(), key=lambda x: -x[1])[:12])

    year_avg = {y: round(np.mean(v), 3) for y, v in sorted(year_sentiment.items()) if len(v) >= 3}

    results.sort(key=lambda x: x['score'], reverse=True)

    payload = {
        'distribution': score_dist,
        'genre_sentiment': {
            'labels': list(genre_avg_sorted.keys()),
            'values': list(genre_avg_sorted.values()),
        },
        'year_sentiment': {
            'years': list(year_avg.keys()),
            'values': list(year_avg.values()),
        },
        'top_positive': results[:10],
        'top_negative': results[-10:][::-1],
        'sample_count': len(results),
    }
    ml_service.analysis_cache['sentiment_analysis'] = payload
    return jsonify(payload)


@api_bp.route('/wordcloud-data')
@login_required
def wordcloud_data():
    freqs = ml_service.get_word_frequencies(300)
    return jsonify({
        'words': [{'name': w, 'value': c} for w, c in freqs],
    })


@api_bp.route('/model-results')
@login_required
def model_results():
    baseline = ml_service.baseline_results
    improved = ml_service.improved_results
    return jsonify({
        'baseline': baseline,
        'improved': improved,
    })


@api_bp.route('/predict', methods=['POST'])
@login_required
def predict_genre():
    data = request.get_json()
    text = data.get('text', '')
    if not text.strip():
        return jsonify({'error': '请输入电影剧情简介'}), 400
    result = ml_service.predict_genre(text)
    return jsonify(result)


@api_bp.route('/recommend', methods=['POST'])
@login_required
def recommend_movies():
    data = request.get_json()
    movie_idx = data.get('movie_index')
    if movie_idx is None:
        return jsonify({'error': '请选择一部电影'}), 400
    results = ml_service.get_similar_movies(int(movie_idx), n=10)
    return jsonify({'recommendations': results})


@api_bp.route('/search-movies')
@login_required
def search_movies():
    q = request.args.get('q', '').strip()
    if not q:
        return jsonify({'results': []})
    df = ml_service.df
    mask = df['电影名称'].str.contains(q, case=False, na=False)
    matches = df[mask].head(20)
    results = []
    for _, row in matches.iterrows():
        results.append({
            'index': int(row.name),
            'name': str(row.get('电影名称', '')),
            'rating': safe_float(row.get('评分')) if pd.notna(row.get('评分')) else None,
            'year': safe_int(row.get('上映年份')) if pd.notna(row.get('上映年份')) else None,
            'genre': str(row.get('类型', '')),
        })
    return jsonify({'results': results})


@api_bp.route('/movie-detail/<int:idx>')
@login_required
def movie_detail(idx):
    df = ml_service.df
    if idx >= len(df):
        return jsonify({'error': '电影不存在'}), 404
    row = df.iloc[idx]
    sa = ml_service.analyze_sentiment(row.get('剧情简介'))
    return jsonify({
        'movie_id': safe_int(row.get('movie_id')),
        'name': str(row.get('电影名称', '')),
        'rating': safe_float(row.get('评分')) if pd.notna(row.get('评分')) else None,
        'votes': safe_int(row.get('评分人数')),
        'year': safe_int(row.get('上映年份')) if pd.notna(row.get('上映年份')) else None,
        'duration': str(row.get('片长', '')),
        'genre': str(row.get('类型', '')),
        'director': str(row.get('导演', '')),
        'actors': str(row.get('主演', '')),
        'writers': str(row.get('编剧', '')),
        'country': str(row.get('国家/地区', '')),
        'language': str(row.get('语言', '')),
        'summary': str(row.get('剧情简介', '')),
        'sentiment': sa,
    })
