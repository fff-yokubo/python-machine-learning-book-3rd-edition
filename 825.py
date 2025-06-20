import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

df=pd.read_csv("ch08/movie_data.csv")



import re
def preprocessor(text):#記号削除 (ただし、顔文字を除く)
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    return text
  
from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()


def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
  
  
  
from nltk.corpus import stopwords

stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] if w not in stop]


#訓練データ　Posotive率: 50.1%
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
#テストデータ Positive率 49.9%
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values



from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop],
               'vect__tokenizer': [tokenizer_porter],
               'clf__penalty': ['l2'],
               'clf__C': [10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop],
               'vect__tokenizer': [tokenizer],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty': ['l2'],
               'clf__C': [10.0, 100.0]},
              ]

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0, solver='liblinear'))])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy',
                           cv=3,
                           verbose=2,
                           n_jobs=-1)



gs_lr_tfidf.fit(X_train, y_train)


print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)
clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))