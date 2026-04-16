import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, label_binarize, LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV
import warnings, os, json
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SCRIPT_DIR, '..', 'diagrams')
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
os.makedirs(OUT, exist_ok=True)

# ========== Load & Prepare ==========
df = pd.read_excel(os.path.join(DATA_DIR, 'douban_movies.xlsx'))
df = df[df['剧情简介'].notna() & df['类型'].notna()].copy()
df['primary_genre'] = df['类型'].apply(lambda x: x.split('/')[0].strip())
top_genres = df['primary_genre'].value_counts().head(10).index.tolist()
df = df[df['primary_genre'].isin(top_genres)].copy()

le = LabelEncoder()
y = le.fit_transform(df['primary_genre'])
classes = le.classes_

tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(1,3), max_features=8000, min_df=2, max_df=0.95, sublinear_tf=True)
X = tfidf.fit_transform(df['剧情简介'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Train models
nb = MultinomialNB(alpha=0.5).fit(X_train, y_train)
dt = DecisionTreeClassifier(max_depth=30, min_samples_split=5, min_samples_leaf=3, random_state=42).fit(X_train, y_train)
svm = LinearSVC(C=0.5, max_iter=5000, random_state=42).fit(X_train, y_train)

y_nb = nb.predict(X_test)
y_dt = dt.predict(X_test)
y_svm = svm.predict(X_test)

print("Models trained. Generating plots...")

# ========== 图3-3: Genre Distribution ==========
fig, ax = plt.subplots(figsize=(12, 6))
genre_counts = df['primary_genre'].value_counts()
colors = sns.color_palette('Blues_d', len(genre_counts))
bars = ax.barh(range(len(genre_counts)), genre_counts.values, color=colors)
ax.set_yticks(range(len(genre_counts)))
ax.set_yticklabels(genre_counts.index, fontsize=14)
ax.invert_yaxis()
ax.set_xlabel('电影数量', fontsize=14)
ax.set_title('豆瓣电影数据集类型分布（共4371部）', fontsize=18, fontweight='bold')
for i, (v, name) in enumerate(zip(genre_counts.values, genre_counts.index)):
    ax.text(v + 20, i, f'{v} ({v/len(df)*100:.1f}%)', va='center', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT}/genre-distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  genre-distribution.png")

# ========== 图3-4: Rating Distribution ==========
fig, ax = plt.subplots(figsize=(10, 6))
ratings = df['评分'].dropna()
ax.hist(ratings, bins=30, color='#4e79a7', edgecolor='white', alpha=0.85)
ax.set_xlabel('豆瓣评分', fontsize=14)
ax.set_ylabel('电影数量', fontsize=14)
ax.set_title('豆瓣电影评分分布直方图', fontsize=18, fontweight='bold')
ax.axvline(ratings.mean(), color='red', linestyle='--', linewidth=2, label=f'均值={ratings.mean():.2f}')
ax.legend(fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT}/rating-distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  rating-distribution.png")

# ========== 图4-1: TF-IDF Feature Word Cloud (bar chart) ==========
feature_names = tfidf.get_feature_names_out()
# Get top features by mean TF-IDF weight
mean_tfidf = X.mean(axis=0).A1
top_indices = mean_tfidf.argsort()[-30:][::-1]
top_words = [feature_names[i] for i in top_indices]
top_vals = [mean_tfidf[i] for i in top_indices]

fig, ax = plt.subplots(figsize=(14, 8))
y_pos = range(len(top_words))
ax.barh(y_pos, top_vals, color=sns.color_palette('viridis', len(top_words)))
ax.set_yticks(y_pos)
ax.set_yticklabels(top_words, fontsize=12)
ax.invert_yaxis()
ax.set_xlabel('平均 TF-IDF 权重', fontsize=14)
ax.set_title('TF-IDF 高权重特征（Top 30）', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/feature-importance-wordcloud.png', dpi=150, bbox_inches='tight')
plt.close()
print("  feature-importance-wordcloud.png")

# ========== 图4-2: Algorithm Comparison ==========
models = ['朴素贝叶斯', '决策树', 'SVM(优化)']
accs = [accuracy_score(y_test, y_nb), accuracy_score(y_test, y_dt), accuracy_score(y_test, y_svm)]
from sklearn.metrics import f1_score as f1s
f1s_vals = [f1s(y_test, y_nb, average='weighted'), f1s(y_test, y_dt, average='weighted'), f1s(y_test, y_svm, average='weighted')]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(models))
w = 0.35
bars1 = ax.bar(x - w/2, accs, w, label='准确率', color='#4e79a7', alpha=0.85)
bars2 = ax.bar(x + w/2, f1s_vals, w, label='F1值', color='#f28e2b', alpha=0.85)
ax.set_ylabel('分数', fontsize=14)
ax.set_title('三种分类算法性能对比', fontsize=18, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=14)
ax.set_ylim(0, 0.8)
ax.legend(fontsize=13)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{bar.get_height():.4f}', ha='center', fontsize=11)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{bar.get_height():.4f}', ha='center', fontsize=11)
plt.tight_layout()
plt.savefig(f'{OUT}/algorithm-comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  algorithm-comparison.png")

# ========== 图4-3: Training Time ==========
import time
times = []
for name, clf_class, params in [
    ('朴素贝叶斯', MultinomialNB, {'alpha': 0.5}),
    ('决策树', DecisionTreeClassifier, {'max_depth': 30, 'min_samples_split': 5, 'min_samples_leaf': 3, 'random_state': 42}),
    ('SVM(优化)', LinearSVC, {'C': 0.5, 'max_iter': 5000, 'random_state': 42}),
]:
    t0 = time.time()
    clf_class(**params).fit(X_train, y_train)
    times.append(time.time() - t0)

fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#4e79a7', '#f28e2b', '#e15759']
bars = ax.bar(models, times, color=colors, alpha=0.85, edgecolor='white')
ax.set_ylabel('训练时间 (秒)', fontsize=14)
ax.set_title('三种算法建模耗时对比', fontsize=18, fontweight='bold')
for bar, t in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{t:.4f}s', ha='center', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT}/training-time.png', dpi=150, bbox_inches='tight')
plt.close()
print("  training-time.png")

# ========== 图4-4: Cross Validation ==========
cv_nb = cross_val_score(nb, X, y, cv=5, scoring='accuracy')
cv_dt = cross_val_score(dt, X, y, cv=5, scoring='accuracy')
cv_svm = cross_val_score(svm, X, y, cv=5, scoring='accuracy')

fig, ax = plt.subplots(figsize=(10, 6))
bp = ax.boxplot([cv_nb, cv_dt, cv_svm], labels=models, patch_artist=True, widths=0.5)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('准确率', fontsize=14)
ax.set_title('5折交叉验证结果对比', fontsize=18, fontweight='bold')
for i, (cv, name) in enumerate(zip([cv_nb, cv_dt, cv_svm], models)):
    ax.text(i+1, cv.mean() + 0.005, f'μ={cv.mean():.4f}±{cv.std():.4f}', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/cross-validation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  cross-validation.png")

# ========== 图5-1~5-3: Confusion Matrices ==========
for pred, name, fname in [(y_nb, '朴素贝叶斯', 'nb'), (y_dt, '决策树', 'dt'), (y_svm, 'SVM(优化)', 'svm')]:
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax, linewidths=0.5)
    ax.set_xlabel('预测类型', fontsize=14)
    ax.set_ylabel('真实类型', fontsize=14)
    ax.set_title(f'{name}分类混淆矩阵', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT}/{fname}-confusion-matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {fname}-confusion-matrix.png")

# ========== 图5-4: ROC Curves ==========
y_test_bin = label_binarize(y_test, classes=range(len(classes)))
n_classes = len(classes)

fig, ax = plt.subplots(figsize=(10, 8))

# NB has predict_proba
nb_proba = nb.predict_proba(X_test)
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], nb_proba[:, i])
    roc_auc = auc(fpr, tpr)

# Compute macro-average ROC for each model
def plot_macro_roc(y_score, label, color, linestyle='-'):
    fpr_all, tpr_all = {}, {}
    for i in range(n_classes):
        fpr_all[i], tpr_all[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    all_fpr = np.unique(np.concatenate([fpr_all[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr_all[i], tpr_all[i])
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, color=color, linestyle=linestyle, linewidth=2, label=f'{label} (AUC={macro_auc:.4f})')

# NB
plot_macro_roc(nb_proba, '朴素贝叶斯', '#4e79a7')

# DT
dt_proba = dt.predict_proba(X_test)
plot_macro_roc(dt_proba, '决策树', '#f28e2b')

# SVM (need CalibratedClassifierCV for proba)
svm_cal = CalibratedClassifierCV(LinearSVC(C=0.5, max_iter=5000, random_state=42), cv=3)
svm_cal.fit(X_train, y_train)
svm_proba = svm_cal.predict_proba(X_test)
plot_macro_roc(svm_proba, 'SVM(优化)', '#e15759')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='随机基线')
ax.set_xlabel('假阳性率 (FPR)', fontsize=14)
ax.set_ylabel('真阳性率 (TPR)', fontsize=14)
ax.set_title('三种算法宏平均 ROC 曲线对比', fontsize=18, fontweight='bold')
ax.legend(fontsize=13, loc='lower right')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig(f'{OUT}/roc-curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  roc-curves.png")

print("\nAll 12 plots generated successfully!")
