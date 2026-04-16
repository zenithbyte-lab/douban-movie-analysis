"""
豆瓣电影评论爬虫
运行后生成 data/comments.csv
"""
import time
import random
import os
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def rating_to_label(rating):
    if rating >= 4:
        return '好评'
    if rating == 3:
        return '中评'
    if rating in (1, 2):
        return '差评'
    return None


def init_driver():
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)')
    return webdriver.Chrome(options=options)


def fetch_page(driver, movie_id, start=0, sort='hot'):
    url = (
        f'https://movie.douban.com/subject/{movie_id}/comments'
        f'?start={start}&limit=20&status=P&sort={sort}'
    )
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, '#comments')))
        time.sleep(random.uniform(1, 2))
        return driver.page_source
    except:
        return None


def parse_comments(html, movie_id, movie_title):
    comments = []
    soup = BeautifulSoup(html, 'html.parser')
    for item in soup.select('#comments .comment-item'):
        try:
            content = item.select_one('.short')
            if not content: continue
            content = content.get_text(strip=True)
            rating = 0
            rating_tag = item.select_one('.rating')
            if rating_tag:
                for cls in rating_tag.get('class', []):
                    if cls.startswith('allstar'):
                        rating = int(cls.replace('allstar', '')) // 10
            time_tag = item.select_one('.comment-time')
            comment_time = None
            if time_tag:
                comment_time = time_tag.get('title') or time_tag.get_text(strip=True)
            votes = item.select_one('.votes')
            votes = int(votes.get_text(strip=True)) if votes else 0
            comments.append({
                'movie_id': movie_id,
                'movie_title': movie_title,
                'content': content,
                'rating': rating,
                'rating_label': rating_to_label(rating),
                'comment_time': comment_time,
                'votes': votes
            })
        except: pass
    return comments


def crawl_movie_comments(
    movie,
    target_total=80,
    min_per_label=5,
    max_pages=20,
    sorts=None,
):
    """爬取单部电影评论并尽量覆盖好/中/差评"""
    sorts = sorts or ['hot', 'new_score', 'time', 'low_score']
    print('='*50)
    print('豆瓣电影评论爬虫')
    print('='*50)

    driver = init_driver()
    all_comments = []
    seen = set()
    label_counts = {'好评': 0, '中评': 0, '差评': 0}

    print(f"\n爬取《{movie['title']}》...")
    for sort in sorts:
        for page in range(max_pages):
            start = page * 20
            html = fetch_page(driver, movie['movie_id'], start=start, sort=sort)
            if not html:
                continue
            comments = parse_comments(html, movie['movie_id'], movie['title'])
            for comment in comments:
                key = comment['content']
                if key in seen:
                    continue
                seen.add(key)
                all_comments.append(comment)
                label = comment.get('rating_label')
                if label in label_counts:
                    label_counts[label] += 1
            if len(all_comments) >= target_total and all(label_counts[l] >= min_per_label for l in label_counts):
                break
            time.sleep(random.uniform(1.2, 2.5))
        if len(all_comments) >= target_total and all(label_counts[l] >= min_per_label for l in label_counts):
            break

    driver.quit()
    print(f'\n爬取完成！共 {len(all_comments)} 条评论')
    print(f'好评/中评/差评: {label_counts}')
    return all_comments


if __name__ == '__main__':
    # 选择单一电影，建议挑选评价分布更分散的影片，确保好/中/差评都有
    movie = {'movie_id': '1292052', 'title': '肖申克的救赎'}
    comments = crawl_movie_comments(movie, target_total=80, min_per_label=5, max_pages=30)
    
    # 保存数据（相对于脚本所在目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    df = pd.DataFrame(comments)
    output_path = os.path.join(data_dir, 'comments.csv')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'\n数据已保存到 {output_path}')
