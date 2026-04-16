import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))

SECRET_KEY = os.environ.get('SECRET_KEY', 'douban-movie-analysis-secret-key-2024')
SQLALCHEMY_DATABASE_URI = f"sqlite:///{os.path.join(BASE_DIR, 'app.db')}"
SQLALCHEMY_TRACK_MODIFICATIONS = False

MOVIE_DATA_PATH = os.path.join(PROJECT_DIR, 'thesis_analysis', 'data', 'douban_movies.xlsx')
BASELINE_RESULTS_PATH = os.path.join(PROJECT_DIR, 'thesis_analysis', 'data', 'experiment_results.json')
IMPROVED_RESULTS_PATH = os.path.join(PROJECT_DIR, 'thesis_analysis', 'data', 'experiment_results_improved.json')
