# import libraries
#https://goodboychan.github.io/python/datacamp/natural_language_processing/2020/07/17/04-TF-IDF-and-similarity-scores.html
#help taken from ChatGpt

import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
import matplotlib.pyplot as plt
import seaborn as sns


#load and inspect the dataset
#improve from chatgpt

df = pd.read_csv('C:/Users/DELL/Downloads/FASDH25-portfolio3/dataset/FASDHportfolio3/articles.csv')


# inspect the dataframe
print(df.head())
print(df.columns)

#extract the text column
#drop any missing entries
articles = df['transcript'].dropna()

#create the TF-IDF Vectors
#create the TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

#generate the TF - IDF matrix
tfidf_matrix = vectorizer.fit_transform(articles)

#print the shape of the matrix
print(tfidf_matrix.shape) #num_articles, num_unique_words)

#cosine similarity matrix
#start timer
start = time.time()

#compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

#print time taken
print("Time taken: %s seconds"% (time.time() - start))
#cosine_sim = kernel(tfidf_matrix, tfidf, tfidf_matrix)

#creating a dataframe for similarity matrix
titles = df.loc[articles.index, 'title']
cosine_sim_df = pd.DataFrame(cosine_sim, index=title, columns=title)
print(cosine_sim_df.head())

#visualize the cosine similarity
plt.figure(figsize=(12, 10))
sns.heatmap(cosine_sim_df, cmap='viridis')
plt.title("cosine Similarity between Articles (TF-IDF)")
plt.xlabel("Articles")
plt.ylabel("Articles")
plt.tight_layout()
plt.show()
                            




    
