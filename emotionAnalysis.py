# coding:utf-8
import pandas as pd
from snownlp import SnowNLP
import jieba.analyse
from wordcloud import WordCloud
from PIL import Image
import numpy as np
from collections import Counter
import jieba.posseg as psg

def sentiment_analyse(v_cmt_list):
    """
    情感打分
    :param v_cmt_list:
    :return:
    """
    score_list = []
    tag_list = []
    pos_count = 0
    neg_count = 0
    index = 0
    for comment in v_cmt_list:
        tag = ''
        semtiments_score = SnowNLP(comment).sentiments
        if semtiments_score < 0.5:
            tag = '消极'
            neg_count += 1
        else:
            tag = '积极'
            pos_count += 1
        score_list.append(semtiments_score)
        tag_list.append(tag)
        index += 1
        print(index)
    print("积极评价占比", round(pos_count / (pos_count + neg_count), 4))
    print("消极评价占比", round(neg_count / (pos_count + neg_count), 4))
    raw_data['情感得分'] = score_list
    raw_data['情感分类'] = tag_list
    #raw_data.to_excel("middle_result.xlsx", index = None)
    print("情感分析完成")


def make_wordcloud(v_str, v_stopwords,v_outfile):
    try:
        stopWords = v_stopwords
        wc = WordCloud(
            background_color="white",
            width=1500,
            height=1200,
            max_words=500,
            stopwords=stopWords,
            font_path="C:\Windows\Fonts\simhei.ttf"
        )
        jieba_text = " ".join(jieba.lcut(v_str))
        wc.generate(jieba_text)
        wc.to_file(v_outfile)
        print("词云图保存成功")
    except Exception as e:
        print(e)


def print_top_words(model, feature_names, n_top_words):
    tword = []
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        topic_w = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        tword.append(topic_w)
        print(topic_w)
    return tword




raw_data = pd.read_csv('./high.csv')#在此此修改产品等级


v_cmt_list = raw_data['描述'].values.tolist()
v_cmt_list = [str(i) for i in v_cmt_list]
v_cmt_str = ''.join(str(i) for i in v_cmt_list)

#进行情感分析
sentiment_analyse(v_cmt_list)

"""
数据清洗
"""

reviews = raw_data.copy()
reviews = reviews[['类别', '描述']]
print("去重之前", reviews.shape[0])
reviews = reviews.drop_duplicates()
print("去重之后",reviews.shape[0])

# 检查缺失值
print(reviews.isnull().sum())

# 删除包含缺失值的行
reviews = reviews.dropna()
print("去掉缺少数据之后", reviews.shape[0])

#数据清洗
import re
content = reviews['描述']
info = re.compile('[0-9a-zA-Z]|手机|vivo|')
content = content.apply(lambda x: info.sub("", str(x)))
#导入停用词
stop_path=open('./stoplist.txt','r',encoding='UTF-8')
stop_words=stop_path.readlines()
len(stop_words)
print(stop_words[0:5])

#停用词，预处理
stop_words=[word.strip('\n') for word in stop_words]
print(stop_words[0:5])


keywords_with_weight = jieba.analyse.extract_tags(v_cmt_str, withWeight=True, topK= 100)
# 过滤停用词
filtered_keywords = [(word, weight) for word, weight in keywords_with_weight if word not in stop_words]

# 获取 top 10 高频词
top_10_keywords = Counter(dict(filtered_keywords)).most_common(15)

# 打印结果
for word, weight in top_10_keywords:
    print(f"{word}: {weight}")

make_wordcloud(v_str=v_cmt_str,v_stopwords=stop_words,v_outfile="wordcloud.jpg")

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

custom_stop_words = set(stop_words)

# 合并默认英语停用词和自定义停用词
my_stop_words =list( ENGLISH_STOP_WORDS.union(custom_stop_words))

n_features = 1000 #提取1000个特征词语
tf_vectorizer = CountVectorizer(strip_accents = 'unicode',
                                max_features=n_features,
                                stop_words=my_stop_words,
                                max_df = 0.5,
                                min_df = 10)
tf = tf_vectorizer.fit_transform(content)

n_topics = 8
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=50,
                                learning_method='batch',
                                learning_offset=50,
                                 doc_topic_prior=0.1,
                                topic_word_prior=0.01,
                               random_state=0)
lda.fit(tf)

n_top_words = 25
tf_feature_names = tf_vectorizer.get_feature_names_out()
topic_word = print_top_words(lda, tf_feature_names, n_top_words)