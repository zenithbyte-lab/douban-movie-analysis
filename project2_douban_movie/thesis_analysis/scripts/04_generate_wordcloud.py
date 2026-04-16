import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import jieba
import re
import os
from collections import Counter

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC']
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SCRIPT_DIR, '..', 'diagrams')
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
os.makedirs(OUT, exist_ok=True)

# Load data
df = pd.read_excel(os.path.join(DATA_DIR, 'douban_movies.xlsx'))
df = df[df['剧情简介'].notna() & df['类型'].notna()].copy()
df['primary_genre'] = df['类型'].apply(lambda x: x.split('/')[0].strip())
top_genres = df['primary_genre'].value_counts().head(10).index.tolist()
df = df[df['primary_genre'].isin(top_genres)].copy()

# Jieba word segmentation on all plot summaries
stopwords = set('的 了 是 在 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 自己 这 他 她 它 们 那 里 吗 吧 啊 呢 嗯 哦 呀 哈 嘿 么 与 及 或 等 把 被 让 给 对 从 向 往 于 而 但 却 又 之 其 中 为 所 以 因 如 果 虽然 然而 不过 已经 可以 这个 那个 什么 怎么 为什么 如何 哪 哪里 哪个 谁 多少 几 每 各 某 这些 那些 一些 其他 另 另外 此 该 才 只 仍 再 更 最 还 过 来 去 起来 出来 下去 上来 进 出 回 起 开 关 大 小 多 少 长 短 高 低 前 后 左 右 里 外 内 本 这样 那样 怎样 如此 十分 非常 特别 真 太 比较 一直 已 将 正 正在 之后 之前 之间 关于 通过 进行 开始 成为 可能 需要 应该 能够 必须 时候 时间 其中 之一 以及 因为 所以 虽然 但是 如果 那么 然后 而且 或者 无论 不仅 不但 只要 只有 除了 除非 即使 甚至 并且 因此 于是 至少 至多 同时 一样 一般 当然 后来 以后 以前 曾经 终于 总是 经常 偶尔 任何 全部 所有 整个 每个'.split())
# Add single chars and punctuation
stopwords.update(set('，。！？、；：""''（）…—— 》《【】｛｝～·'))

word_counter = Counter()
for text in df['剧情简介']:
    # Clean: remove punctuation and special chars
    clean = re.sub(r'[^\u4e00-\u9fffA-Za-z]', ' ', str(text))
    words = jieba.lcut(clean)
    for w in words:
        w = w.strip()
        if len(w) >= 2 and w not in stopwords and not re.match(r'^[a-zA-Z]{1,2}$', w):
            word_counter[w] += 1

# Keep top 300 words
top_words = dict(word_counter.most_common(300))
print(f"Top 20 words: {list(top_words.items())[:20]}")

# Find Chinese font
font_candidates = [
    '/System/Library/Fonts/PingFang.ttc',
    '/System/Library/Fonts/STHeiti Medium.ttc',
    '/Library/Fonts/Arial Unicode.ttf',
    '/System/Library/Fonts/Hiragino Sans GB.ttc',
]
font_path = None
for fp in font_candidates:
    if os.path.exists(fp):
        font_path = fp
        break
print(f"Using font: {font_path}")

# Generate proper word cloud
wc = WordCloud(
    font_path=font_path,
    width=1600,
    height=900,
    background_color='white',
    max_words=200,
    max_font_size=160,
    min_font_size=16,
    colormap='Dark2',
    relative_scaling=0.5,
    prefer_horizontal=0.7,
    margin=6,
    contour_width=0,
    collocations=False,
)
wc.generate_from_frequencies(top_words)

fig, ax = plt.subplots(figsize=(16, 9))
ax.imshow(wc, interpolation='bilinear')
ax.axis('off')
plt.tight_layout(pad=0)
plt.savefig(f'{OUT}/feature-importance-wordcloud.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Done: feature-importance-wordcloud.png")
