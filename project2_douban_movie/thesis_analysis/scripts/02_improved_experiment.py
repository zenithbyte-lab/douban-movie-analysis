# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import json, os, time, re, warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV
from scipy.sparse import hstack
import jieba

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
OUT = os.path.join(SCRIPT_DIR, '..', 'diagrams')
os.makedirs(OUT, exist_ok=True)

# ========== 1. Load ==========
df = pd.read_excel(os.path.join(DATA_DIR, 'douban_movies.xlsx'))
df = df[df['\u5267\u60c5\u7b80\u4ecb'].notna() & df['\u7c7b\u578b'].notna()].copy()
df['primary_genre'] = df['\u7c7b\u578b'].apply(lambda x: x.split('/')[0].strip())

# ========== 2. Merge to 6 classes ==========
genre_map = {
    '\u5267\u60c5': '\u5267\u60c5',
    '\u559c\u5267': '\u559c\u5267',
    '\u52a8\u4f5c': '\u52a8\u4f5c',
    '\u52a8\u753b': '\u52a8\u753b',
    '\u7231\u60c5': '\u559c\u5267',        # -> comedy/romance merged
    '\u79d1\u5e7b': '\u79d1\u5e7b\u60ac\u7591',  # sci-fi -> sci-fi/thriller
    '\u6050\u6016': '\u79d1\u5e7b\u60ac\u7591',
    '\u60ca\u609a': '\u79d1\u5e7b\u60ac\u7591',
    '\u60ac\u7591': '\u79d1\u5e7b\u60ac\u7591',
    '\u7eaa\u5f55\u7247': '\u7eaa\u5f55\u7247',
}
top10 = df['primary_genre'].value_counts().head(10).index.tolist()
df = df[df['primary_genre'].isin(top10)].copy()
df['merged_genre'] = df['primary_genre'].map(genre_map)
df = df[df['merged_genre'].notna()].copy()

print("=== Merged genre distribution ===")
print(df['merged_genre'].value_counts())
print(f"Total: {len(df)}")

# ========== 3. Jieba word segmentation ==========
stopwords_raw = "\u7684 \u4e86 \u662f \u5728 \u6211 \u6709 \u548c \u5c31 \u4e0d \u4eba \u90fd \u4e00 \u4e00\u4e2a \u4e0a \u4e5f \u5f88 \u5230 \u8bf4 \u8981 \u53bb \u4f60 \u4f1a \u7740 \u6ca1\u6709 \u770b \u597d \u81ea\u5df1 \u8fd9 \u4ed6 \u5979 \u5b83 \u4eec \u90a3 \u91cc \u5417 \u5427 \u554a \u5462 \u4e48 \u4e0e \u53ca \u6216 \u7b49 \u628a \u88ab \u8ba9 \u7ed9 \u5bf9 \u4ece \u5411 \u5f80 \u4e8e \u800c \u4f46 \u5374 \u53c8 \u4e4b \u5176 \u4e2d \u4e3a \u6240 \u4ee5 \u56e0 \u5982 \u679c \u53ef\u4ee5 \u4ec0\u4e48 \u600e\u4e48 \u5982\u4f55 \u54ea \u8c01 \u624d \u53ea \u4ecd \u518d \u66f4 \u6700 \u8fd8 \u8fc7 \u6765 \u51fa \u56de \u8d77 \u5f00 \u5173 \u5927 \u5c0f \u524d \u540e \u5916 \u5185 \u672c \u5df2 \u5c06 \u6b63 \u6b63\u5728 \u5173\u4e8e \u901a\u8fc7 \u8fdb\u884c \u5f00\u59cb \u6210\u4e3a \u53ef\u80fd \u9700\u8981 \u5e94\u8be5 \u80fd\u591f \u5fc5\u987b \u65f6\u5019 \u65f6\u95f4 \u5176\u4e2d \u4e4b\u4e00 \u4ee5\u53ca \u56e0\u4e3a \u6240\u4ee5 \u4f46\u662f \u5982\u679c \u90a3\u4e48 \u7136\u540e \u800c\u4e14 \u6216\u8005 \u5373\u4f7f \u751a\u81f3 \u5e76\u4e14 \u56e0\u6b64 \u4e8e\u662f \u540c\u65f6 \u4e00\u6837 \u4e00\u822c \u5f53\u7136 \u540e\u6765 \u4ee5\u540e \u4ee5\u524d \u66fe\u7ecf \u7ec8\u4e8e \u603b\u662f \u7ecf\u5e38 \u4efb\u4f55 \u5168\u90e8 \u6240\u6709 \u6574\u4e2a \u6bcf\u4e2a \u5e76 \u5f97 \u4e0d\u662f \u4e0d\u4f1a \u53ea\u662f \u4e0d\u8fc7 \u5df2\u7ecf \u867d\u7136 \u7136\u800c \u4e4b\u540e \u4e4b\u524d \u4e4b\u95f4 \u53ea\u8981 \u53ea\u6709 \u9664\u4e86 \u65e0\u8bba \u4e0d\u4ec5 \u4e0d\u4f46 \u81f3\u5c11 \u53e6\u5916 \u800c\u662f \u53ef\u662f \u8fd9\u4e2a \u90a3\u4e2a \u8fd9\u4e9b \u90a3\u4e9b \u4e00\u4e9b \u5176\u4ed6 \u8be5 \u6b64"
stopwords = set(stopwords_raw.split())

def seg_text(text):
    clean = re.sub(r'[^\u4e00-\u9fffA-Za-z]', ' ', str(text))
    words = jieba.lcut(clean)
    return ' '.join(w for w in words if len(w) >= 2 and w not in stopwords)

print("\nSegmenting with jieba...")
df['seg_text'] = df['\u5267\u60c5\u7b80\u4ecb'].apply(seg_text)

# ========== 4. Features: TF-IDF (word) + structured ==========
le = LabelEncoder()
y = le.fit_transform(df['merged_genre'])
classes = le.classes_
print(f"\nClasses: {list(classes)}")

# TF-IDF on jieba words
tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=8000, min_df=2, max_df=0.95, sublinear_tf=True)
X_text = tfidf.fit_transform(df['seg_text'])

# Structured features: rating, vote_count, year, duration
struct_cols = []
for col in ['\u8bc4\u5206', '\u8bc4\u5206\u4eba\u6570', '\u4e0a\u6620\u5e74\u4efd']:
    vals = pd.to_numeric(df[col], errors='coerce').fillna(0).values.reshape(-1,1)
    struct_cols.append(vals)

# Parse duration
def parse_duration(x):
    if pd.isna(x): return 0
    m = re.search(r'(\d+)', str(x))
    return int(m.group(1)) if m else 0
struct_cols.append(np.array([parse_duration(x) for x in df['\u7247\u957f']]).reshape(-1,1))

struct_raw = np.hstack(struct_cols)
scaler = MinMaxScaler()
struct_scaled = scaler.fit_transform(struct_raw)

from scipy.sparse import csr_matrix
X_struct = csr_matrix(struct_scaled)
X = hstack([X_text, X_struct])
print(f"Features: {X_text.shape[1]} text + {X_struct.shape[1]} struct = {X.shape[1]} total")

# ========== 5. Split ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ========== 6. SMOTE on train ==========
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {X_train_sm.shape[0]} train samples")
from collections import Counter
print(f"SMOTE class dist: {Counter(y_train_sm)}")

results = {}

# ========== 7. NB (before/after improvement) ==========
print("\n=== Naive Bayes (baseline: text only, no SMOTE) ===")
X_text_train, X_text_test = train_test_split(X_text, test_size=0.3, stratify=y, random_state=42)[:2]
# baseline
nb_base = MultinomialNB(alpha=0.5)
nb_base.fit(X_text_train, y_train)
y_base = nb_base.predict(X_text_test)
print(f"Baseline NB: Acc={accuracy_score(y_test, y_base):.4f}, F1={f1_score(y_test, y_base, average='weighted'):.4f}")

print("\n=== Naive Bayes (improved: word TF-IDF + struct, no SMOTE, NB can't use SMOTE well on sparse) ===")
nb = MultinomialNB(alpha=0.3)
# NB needs non-negative, struct features are [0,1] so OK
nb.fit(X_train, y_train)
y_nb = nb.predict(X_test)
nb_acc = accuracy_score(y_test, y_nb)
nb_f1 = f1_score(y_test, y_nb, average='weighted')
nb_prec = precision_score(y_test, y_nb, average='weighted')
nb_rec = recall_score(y_test, y_nb, average='weighted')
print(f"Improved NB: Acc={nb_acc:.4f}, Prec={nb_prec:.4f}, Rec={nb_rec:.4f}, F1={nb_f1:.4f}")
nb_cv = cross_val_score(nb, X, y, cv=5, scoring='accuracy')
print(f"CV: {nb_cv.mean():.4f}+/-{nb_cv.std():.4f}")
print(classification_report(y_test, y_nb, target_names=classes))
results['nb'] = {'acc': round(nb_acc,4), 'prec': round(nb_prec,4), 'rec': round(nb_rec,4), 'f1': round(nb_f1,4),
                  'cv_mean': round(nb_cv.mean(),4), 'cv_std': round(nb_cv.std(),4), 'cm': confusion_matrix(y_test, y_nb).tolist()}

# ========== 8. DT (improved: SMOTE + class_weight) ==========
print("\n=== Decision Tree (improved: SMOTE + balanced weight) ===")
t0 = time.time()
dt = DecisionTreeClassifier(max_depth=25, min_samples_split=8, min_samples_leaf=4, class_weight='balanced', random_state=42)
dt.fit(X_train_sm, y_train_sm)
dt_time = time.time() - t0
y_dt = dt.predict(X_test)
dt_acc = accuracy_score(y_test, y_dt)
dt_f1 = f1_score(y_test, y_dt, average='weighted')
dt_prec = precision_score(y_test, y_dt, average='weighted')
dt_rec = recall_score(y_test, y_dt, average='weighted')
print(f"DT: Acc={dt_acc:.4f}, Prec={dt_prec:.4f}, Rec={dt_rec:.4f}, F1={dt_f1:.4f}, Time={dt_time:.2f}s")
dt_cv = cross_val_score(dt, X, y, cv=5, scoring='accuracy')
print(f"CV: {dt_cv.mean():.4f}+/-{dt_cv.std():.4f}")
print(classification_report(y_test, y_dt, target_names=classes))
results['dt'] = {'acc': round(dt_acc,4), 'prec': round(dt_prec,4), 'rec': round(dt_rec,4), 'f1': round(dt_f1,4),
                  'cv_mean': round(dt_cv.mean(),4), 'cv_std': round(dt_cv.std(),4), 'time': round(dt_time,4), 'cm': confusion_matrix(y_test, y_dt).tolist()}

# ========== 9. SVM (improved: SMOTE + class_weight + GridSearch) ==========
print("\n=== SVM (improved: SMOTE + balanced + GridSearch) ===")
t0 = time.time()
svm_base = LinearSVC(C=1.0, class_weight='balanced', max_iter=8000, random_state=42)
svm_base.fit(X_train_sm, y_train_sm)
svm_time = time.time() - t0
y_svm = svm_base.predict(X_test)
svm_acc = accuracy_score(y_test, y_svm)
svm_f1 = f1_score(y_test, y_svm, average='weighted')
svm_prec = precision_score(y_test, y_svm, average='weighted')
svm_rec = recall_score(y_test, y_svm, average='weighted')
print(f"SVM base: Acc={svm_acc:.4f}, Prec={svm_prec:.4f}, Rec={svm_rec:.4f}, F1={svm_f1:.4f}, Time={svm_time:.2f}s")
print(classification_report(y_test, y_svm, target_names=classes))

# Grid search
print("\n=== SVM Grid Search ===")
t0 = time.time()
gs = GridSearchCV(LinearSVC(class_weight='balanced', max_iter=8000, random_state=42),
                  {'C': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]}, cv=5, scoring='f1_weighted', n_jobs=-1)
gs.fit(X_train_sm, y_train_sm)
gs_time = time.time() - t0
print(f"Best C: {gs.best_params_['C']}, Best CV F1: {gs.best_score_:.4f}")
y_gs = gs.predict(X_test)
gs_acc = accuracy_score(y_test, y_gs)
gs_f1 = f1_score(y_test, y_gs, average='weighted')
gs_prec = precision_score(y_test, y_gs, average='weighted')
gs_rec = recall_score(y_test, y_gs, average='weighted')
print(f"SVM opt: Acc={gs_acc:.4f}, Prec={gs_prec:.4f}, Rec={gs_rec:.4f}, F1={gs_f1:.4f}, Time={gs_time:.2f}s")
gs_cv = cross_val_score(gs.best_estimator_, X, y, cv=5, scoring='accuracy')
print(f"CV: {gs_cv.mean():.4f}+/-{gs_cv.std():.4f}")
print(classification_report(y_test, y_gs, target_names=classes))
results['svm'] = {'acc': round(svm_acc,4), 'prec': round(svm_prec,4), 'rec': round(svm_rec,4), 'f1': round(svm_f1,4), 'time': round(svm_time,4)}
results['svm_opt'] = {'acc': round(gs_acc,4), 'prec': round(gs_prec,4), 'rec': round(gs_rec,4), 'f1': round(gs_f1,4),
                       'cv_mean': round(gs_cv.mean(),4), 'cv_std': round(gs_cv.std(),4), 'time': round(gs_time,4),
                       'best_C': gs.best_params_['C'], 'cm': confusion_matrix(y_test, y_gs).tolist()}

# ========== Summary ==========
results['classes'] = list(classes)
results['dataset_size'] = len(df)
results['n_classes'] = len(classes)
results['train_size'] = X_train.shape[0]
results['test_size'] = X_test.shape[0]
results['n_features'] = X.shape[1]

with open(os.path.join(DATA_DIR, 'experiment_results_improved.json'), 'w') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n{'='*60}")
print(f"IMPROVED RESULTS SUMMARY (6 classes, jieba+struct+SMOTE+balanced)")
print(f"{'='*60}")
print(f"Dataset: {len(df)} movies -> 6 merged classes")
print(f"Features: {X.shape[1]} (jieba TF-IDF + rating/votes/year/duration)")
print(f"NB:       Acc={results['nb']['acc']}, F1={results['nb']['f1']}, CV={results['nb']['cv_mean']}+/-{results['nb']['cv_std']}")
print(f"DT:       Acc={results['dt']['acc']}, F1={results['dt']['f1']}, CV={results['dt']['cv_mean']}+/-{results['dt']['cv_std']}")
print(f"SVM:      Acc={results['svm']['acc']}, F1={results['svm']['f1']}")
print(f"SVM(opt): Acc={results['svm_opt']['acc']}, F1={results['svm_opt']['f1']}, CV={results['svm_opt']['cv_mean']}+/-{results['svm_opt']['cv_std']}, C={results['svm_opt']['best_C']}")
