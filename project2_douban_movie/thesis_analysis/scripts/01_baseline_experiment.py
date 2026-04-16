import pandas as pd
import numpy as np
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import time
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
OUT_DIR = os.path.join(SCRIPT_DIR, '..', 'diagrams')
os.makedirs(OUT_DIR, exist_ok=True)

# Load data
df = pd.read_excel(os.path.join(DATA_DIR, 'douban_movies.xlsx'))
df = df[df['剧情简介'].notna() & df['类型'].notna()].copy()

# Extract primary genre (first listed genre)
df['primary_genre'] = df['类型'].apply(lambda x: x.split('/')[0].strip())

# Keep top 10 genres
top_genres = df['primary_genre'].value_counts().head(10).index.tolist()
df = df[df['primary_genre'].isin(top_genres)].copy()

print(f"Dataset: {len(df)} movies, {len(top_genres)} genres")
print(f"Genres: {top_genres}")
print(f"\nGenre distribution:")
for g in top_genres:
    print(f"  {g}: {len(df[df['primary_genre']==g])}")

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df['primary_genre'])

# TF-IDF on plot summaries (character-level for Chinese)
tfidf = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(1, 3),
    max_features=8000,
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)
X = tfidf.fit_transform(df['剧情简介'])
print(f"\nTF-IDF features: {X.shape[1]}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

results = {}

# 1. Naive Bayes
print("\n=== Naive Bayes ===")
t0 = time.time()
nb = MultinomialNB(alpha=0.5)
nb.fit(X_train, y_train)
nb_train_time = time.time() - t0
y_pred_nb = nb.predict(X_test)
nb_acc = accuracy_score(y_test, y_pred_nb)
nb_prec = precision_score(y_test, y_pred_nb, average='weighted')
nb_rec = recall_score(y_test, y_pred_nb, average='weighted')
nb_f1 = f1_score(y_test, y_pred_nb, average='weighted')
nb_cm = confusion_matrix(y_test, y_pred_nb)
print(f"Accuracy: {nb_acc:.4f}, Precision: {nb_prec:.4f}, Recall: {nb_rec:.4f}, F1: {nb_f1:.4f}")
print(f"Training time: {nb_train_time:.4f}s")
nb_cv = cross_val_score(nb, X, y, cv=5, scoring='accuracy')
print(f"CV: {nb_cv.mean():.4f} ± {nb_cv.std():.4f}")
results['nb'] = {
    'acc': round(nb_acc, 4), 'prec': round(nb_prec, 4), 
    'rec': round(nb_rec, 4), 'f1': round(nb_f1, 4),
    'time': round(nb_train_time, 4), 'cv_mean': round(nb_cv.mean(), 4), 
    'cv_std': round(nb_cv.std(), 4), 'cm': nb_cm.tolist()
}
print("\nPer-class report:")
print(classification_report(y_test, y_pred_nb, target_names=le.classes_))

# 2. Decision Tree
print("\n=== Decision Tree ===")
t0 = time.time()
dt = DecisionTreeClassifier(max_depth=30, min_samples_split=5, min_samples_leaf=3, random_state=42)
dt.fit(X_train, y_train)
dt_train_time = time.time() - t0
y_pred_dt = dt.predict(X_test)
dt_acc = accuracy_score(y_test, y_pred_dt)
dt_prec = precision_score(y_test, y_pred_dt, average='weighted')
dt_rec = recall_score(y_test, y_pred_dt, average='weighted')
dt_f1 = f1_score(y_test, y_pred_dt, average='weighted')
dt_cm = confusion_matrix(y_test, y_pred_dt)
print(f"Accuracy: {dt_acc:.4f}, Precision: {dt_prec:.4f}, Recall: {dt_rec:.4f}, F1: {dt_f1:.4f}")
print(f"Training time: {dt_train_time:.4f}s")
dt_cv = cross_val_score(dt, X, y, cv=5, scoring='accuracy')
print(f"CV: {dt_cv.mean():.4f} ± {dt_cv.std():.4f}")
results['dt'] = {
    'acc': round(dt_acc, 4), 'prec': round(dt_prec, 4),
    'rec': round(dt_rec, 4), 'f1': round(dt_f1, 4),
    'time': round(dt_train_time, 4), 'cv_mean': round(dt_cv.mean(), 4),
    'cv_std': round(dt_cv.std(), 4), 'cm': dt_cm.tolist()
}
print("\nPer-class report:")
print(classification_report(y_test, y_pred_dt, target_names=le.classes_))

# 3. SVM (linear for speed on this size)
print("\n=== SVM (Linear) ===")
from sklearn.svm import LinearSVC
t0 = time.time()
svm = LinearSVC(C=1.0, max_iter=5000, random_state=42)
svm.fit(X_train, y_train)
svm_train_time = time.time() - t0
y_pred_svm = svm.predict(X_test)
svm_acc = accuracy_score(y_test, y_pred_svm)
svm_prec = precision_score(y_test, y_pred_svm, average='weighted')
svm_rec = recall_score(y_test, y_pred_svm, average='weighted')
svm_f1 = f1_score(y_test, y_pred_svm, average='weighted')
svm_cm = confusion_matrix(y_test, y_pred_svm)
print(f"Accuracy: {svm_acc:.4f}, Precision: {svm_prec:.4f}, Recall: {svm_rec:.4f}, F1: {svm_f1:.4f}")
print(f"Training time: {svm_train_time:.4f}s")
svm_cv = cross_val_score(svm, X, y, cv=5, scoring='accuracy')
print(f"CV: {svm_cv.mean():.4f} ± {svm_cv.std():.4f}")
results['svm'] = {
    'acc': round(svm_acc, 4), 'prec': round(svm_prec, 4),
    'rec': round(svm_rec, 4), 'f1': round(svm_f1, 4),
    'time': round(svm_train_time, 4), 'cv_mean': round(svm_cv.mean(), 4),
    'cv_std': round(svm_cv.std(), 4), 'cm': svm_cm.tolist()
}
print("\nPer-class report:")
print(classification_report(y_test, y_pred_svm, target_names=le.classes_))

# 4. SVM with Grid Search optimization
print("\n=== SVM Grid Search ===")
param_grid = {'C': [0.1, 0.5, 1.0, 5.0, 10.0]}
t0 = time.time()
gs = GridSearchCV(LinearSVC(max_iter=5000, random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
gs.fit(X_train, y_train)
gs_train_time = time.time() - t0
y_pred_gs = gs.predict(X_test)
gs_acc = accuracy_score(y_test, y_pred_gs)
gs_prec = precision_score(y_test, y_pred_gs, average='weighted')
gs_rec = recall_score(y_test, y_pred_gs, average='weighted')
gs_f1 = f1_score(y_test, y_pred_gs, average='weighted')
gs_cm = confusion_matrix(y_test, y_pred_gs)
print(f"Best C: {gs.best_params_['C']}")
print(f"Accuracy: {gs_acc:.4f}, Precision: {gs_prec:.4f}, Recall: {gs_rec:.4f}, F1: {gs_f1:.4f}")
print(f"Training time: {gs_train_time:.4f}s")
gs_cv = cross_val_score(gs.best_estimator_, X, y, cv=5, scoring='accuracy')
print(f"CV: {gs_cv.mean():.4f} ± {gs_cv.std():.4f}")
results['svm_opt'] = {
    'acc': round(gs_acc, 4), 'prec': round(gs_prec, 4),
    'rec': round(gs_rec, 4), 'f1': round(gs_f1, 4),
    'time': round(gs_train_time, 4), 'cv_mean': round(gs_cv.mean(), 4),
    'cv_std': round(gs_cv.std(), 4), 'cm': gs_cm.tolist(),
    'best_C': gs.best_params_['C']
}

# Save results
results['genres'] = le.classes_.tolist()
results['dataset_size'] = len(df)
results['train_size'] = X_train.shape[0]
results['test_size'] = X_test.shape[0]
results['n_features'] = X.shape[1]

with open(os.path.join(DATA_DIR, 'experiment_results.json'), 'w') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n=== SUMMARY ===")
print(f"Dataset: {len(df)} movies, {len(top_genres)} genres")
print(f"NB:         Acc={results['nb']['acc']}, F1={results['nb']['f1']}, CV={results['nb']['cv_mean']}±{results['nb']['cv_std']}")
print(f"DT:         Acc={results['dt']['acc']}, F1={results['dt']['f1']}, CV={results['dt']['cv_mean']}±{results['dt']['cv_std']}")
print(f"SVM:        Acc={results['svm']['acc']}, F1={results['svm']['f1']}, CV={results['svm']['cv_mean']}±{results['svm']['cv_std']}")
print(f"SVM(opt):   Acc={results['svm_opt']['acc']}, F1={results['svm_opt']['f1']}, CV={results['svm_opt']['cv_mean']}±{results['svm_opt']['cv_std']}")
print(f"\nResults saved to {os.path.join(DATA_DIR, 'experiment_results.json')}")
