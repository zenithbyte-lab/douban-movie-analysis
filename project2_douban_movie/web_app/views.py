from flask import Blueprint, render_template, redirect, url_for
from flask_login import login_required

views_bp = Blueprint('views', __name__)


@views_bp.route('/')
def index():
    return redirect(url_for('views.dashboard'))


@views_bp.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', page='dashboard')


@views_bp.route('/movies')
@login_required
def movies():
    return render_template('movies.html', page='movies')


@views_bp.route('/rating-analysis')
@login_required
def rating_analysis():
    return render_template('rating_analysis.html', page='rating')


@views_bp.route('/genre-analysis')
@login_required
def genre_analysis():
    return render_template('genre_analysis.html', page='genre')


@views_bp.route('/director-analysis')
@login_required
def director_analysis():
    return render_template('director_analysis.html', page='director')


@views_bp.route('/region-analysis')
@login_required
def region_analysis():
    return render_template('region_analysis.html', page='region')


@views_bp.route('/year-trends')
@login_required
def year_trends():
    return render_template('year_trends.html', page='year')


@views_bp.route('/sentiment')
@login_required
def sentiment():
    return render_template('sentiment.html', page='sentiment')


@views_bp.route('/wordcloud')
@login_required
def wordcloud_page():
    return render_template('wordcloud.html', page='wordcloud')


@views_bp.route('/models')
@login_required
def models_page():
    return render_template('models.html', page='models')


@views_bp.route('/prediction')
@login_required
def prediction():
    return render_template('prediction.html', page='prediction')


@views_bp.route('/recommendation')
@login_required
def recommendation():
    return render_template('recommendation.html', page='recommendation')
