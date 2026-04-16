# 豆瓣电影爬虫与情感分析

## 项目结构

```
├── comment_crawler.py        # 爬虫程序（Selenium + BeautifulSoup）
├── sentiment_analysis.ipynb  # 课程实验：情感分析 Notebook
├── requirements.txt
├── data/
│   └── comments.csv          # 爬取的评论数据（75条，5部电影）
├── output/                   # Notebook 生成的可视化图表
│   ├── sentiment_distribution.png
│   ├── model_comparison.png
│   ├── sentiment_score_trend.png
│   ├── rating_trend.png
│   ├── rating_distribution_pie.png
│   └── wordcloud.png
└── thesis_analysis/          # 论文级深度实验
    ├── data/
    │   ├── douban_movies.xlsx              # 豆瓣电影数据集（4371部）
    │   ├── experiment_results.json         # 基线实验结果
    │   └── experiment_results_improved.json # 改进实验结果
    ├── scripts/
    │   ├── 01_baseline_experiment.py       # 基线：TF-IDF + NB/DT/SVM
    │   ├── 02_improved_experiment.py       # 改进：jieba分词 + 结构化特征 + SMOTE
    │   ├── 03_generate_plots.py            # 生成论文图表
    │   └── 04_generate_wordcloud.py        # 词云生成
    └── diagrams/                           # 论文用图表（20+张）
```

## 使用方法

1. 安装依赖
```bash
pip install -r requirements.txt
```

2. 运行爬虫
```bash
python comment_crawler.py
```

3. 打开 Jupyter Notebook 进行分析
```bash
jupyter notebook sentiment_analysis.ipynb
```

4. 运行论文实验
```bash
cd thesis_analysis/scripts
python 01_baseline_experiment.py
python 02_improved_experiment.py
python 03_generate_plots.py
python 04_generate_wordcloud.py
```

## 实验内容

### 课程实验（sentiment_analysis.ipynb）
- SnowNLP 情感分析（均值、中位数、分布）
- 手写朴素贝叶斯 + 逻辑回归
- 模型对比：准确率、精确率、召回率、F1分数
- 评价分布饼图 + 词频统计 + 词云展示

### 论文实验（thesis_analysis/）
- 基于剧情简介的电影类型分类（10类/6类）
- TF-IDF 特征提取（字符级 / jieba 分词）
- 朴素贝叶斯、决策树、SVM 三种算法对比
- SMOTE 过采样 + GridSearchCV 超参优化
- 混淆矩阵、ROC 曲线、交叉验证等论文级图表
