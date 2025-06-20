# %%
import numpy as np
import pandas as pd

df=pd.read_csv("ch08/movie_data.csv")



from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words = 'english',max_df = .1, max_features = 6000)


X =count.fit_transform(df["review"].values)
print(X)



from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components=10, random_state=123, learning_method='online')

X_topics = lda.fit_transform(X)


print(lda.components_.shape)

n_top_words = 5
feature_names = count.get_feature_names_out()

for topic_idx, topic in enumerate(lda.components_):
  print("Topic[%d]: "%(1+topic_idx))
  print(" ".join([feature_names[i] for i in topic.argsort() [:-n_top_words -1 :-1]]))