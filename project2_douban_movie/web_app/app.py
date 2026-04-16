import os
import sys
from flask import Flask
from flask_login import LoginManager
from models import db, User
from auth import auth_bp
from views import views_bp
from api import api_bp
from ml_service import ml_service
import config


def _create_demo_user_if_requested():
    if os.environ.get('CREATE_DEMO_USER') != '1':
        return

    username = os.environ.get('DEMO_USERNAME', '123456').strip()
    password = os.environ.get('DEMO_PASSWORD', '123456')
    email = os.environ.get('DEMO_EMAIL', f'{username}@example.local').strip()

    if not username or not password:
        return
    if User.query.filter_by(username=username).first():
        return
    if User.query.filter_by(email=email).first():
        email = f'{username}-{os.getpid()}@example.local'

    user = User(username=username, email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    print(f'已创建默认登录账号：{username} / {password}')


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = config.SECRET_KEY
    app.config['SQLALCHEMY_DATABASE_URI'] = config.SQLALCHEMY_DATABASE_URI
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = config.SQLALCHEMY_TRACK_MODIFICATIONS

    db.init_app(app)

    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.login_message = '请先登录'
    login_manager.login_message_category = 'warning'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    app.register_blueprint(auth_bp)
    app.register_blueprint(views_bp)
    app.register_blueprint(api_bp)

    with app.app_context():
        db.create_all()
        _create_demo_user_if_requested()

    print('正在加载电影数据和训练模型，请稍候...')
    with app.app_context():
        ml_service.init_app(
            config.MOVIE_DATA_PATH,
            config.BASELINE_RESULTS_PATH,
            config.IMPROVED_RESULTS_PATH,
        )
    print(f'数据加载完成！共 {len(ml_service.df)} 部电影')

    return app


if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('APP_PORT', '5001'))
    debug = os.environ.get('FLASK_DEBUG', '1') == '1'
    app.run(debug=debug, port=port, use_reloader=False)
