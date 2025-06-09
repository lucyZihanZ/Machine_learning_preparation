import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import string
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

rst = pd.read_excel('../data/ERR.xlsx', sheet_name='Restaurants')
print(rst.shape)
rst['Augmented Description'] = rst['Brief Description'] + 'The cuisine is ' + rst['Cuisine']

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(rst['Augmented Description'])

tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
res_matrix = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_feature_names, index=rst['Restaurant Name'])

def find_top_restaurant(term):
    term = term.lower()  # all transform into the lower cases.
    if term not in res_matrix.columns:
        print(f"'{term}' not found in vocabulary.")
        return None
    restaurants = res_matrix[term]
    # scores = res_matrix[term]
    return restaurants

cozy_rst = find_top_restaurant('cozy').reset_index()
cozy_rst.columns = ['Restaurant Name', 'cozy_scores']
pd.set_option('display.max_columns', None)
cozy_rst = pd.merge(cozy_rst, rst[['Restaurant Name', 'Augmented Description']], on = 'Restaurant Name')
cozy_rst  = cozy_rst.sort_values(by = 'cozy_scores', ascending=False)
print(cozy_rst.head())

chi_rst = find_top_restaurant('chinese').reset_index()
chi_rst.columns = ['Restaurant Name', 'scores']
pd.set_option('display.max_columns', None)
chi_rst = pd.merge(chi_rst, rst[['Restaurant Name', 'Augmented Description']], on = 'Restaurant Name')
chinese_rst  = chi_rst.sort_values(by = 'scores', ascending=False)


# Find Top 100 Most Frequent Words 
# use methods to drop the stop words
def word_preprocess(text):
    text = text.lower()
# remove the punctuation, because it may affects our final analysis
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
# use a list to store all the stop words
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)
rst['Augmented Description'] = rst['Augmented Description'].apply(word_preprocess)

count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(rst['Augmented Description'])

word_counts = pd.DataFrame({
    'word': count_vectorizer.get_feature_names_out(),
    'count': count_matrix.toarray().sum(axis=0)
})

# Top 100 frequent words
top_words = word_counts.sort_values(by='count', ascending=False).head(100)['word'].tolist()

print("\nTop frequent words:")
print(word_counts.sort_values(by='count', ascending=False).head(10))

# Recompute TF-IDF using only Top 100 Words
# Limit TF-IDF to top words, no need for 2 loops, because tf-idf can calculate the matrix directly.
tfidf_vectorizer_top = TfidfVectorizer(vocabulary=top_words)
tfidf_matrix_top = tfidf_vectorizer_top.fit_transform(rst['Augmented Description'])
#  (a) Burger King and Edzo’s Burger Shop (b) Burger King and Oceanique (c) Lao Sze Chuan and Kabul House
# be_score = rst.loc["Burger King", "Edzo s Burger Shop"]

cosine_sim = cosine_similarity(tfidf_matrix_top)

# Euclidean Distance
euclidean_dist = euclidean_distances(tfidf_matrix_top)

cosine_sim_df = pd.DataFrame(cosine_sim, index=rst['Restaurant Name'], columns=rst['Restaurant Name'])
euclidean_dist_df = pd.DataFrame(euclidean_dist, index=rst['Restaurant Name'], columns=rst['Restaurant Name'])

be_score = euclidean_dist_df.loc["Burger King", "Edzo's Burger Shop"] 
be_score_cos = cosine_sim_df.loc["Burger King", "Edzo's Burger Shop"]
print(be_score, be_score_cos)
bo_score = euclidean_dist_df.loc['Burger King', 'Oceanique']
bo_cos = cosine_sim_df.loc['Burger King', 'Oceanique']
print(bo_score, bo_cos)
lk_score = euclidean_dist_df.loc['Lao Sze Chuan', 'Kabul House']
lk_score_cos = cosine_sim_df.loc['Lao Sze Chuan', 'Kabul House']
print(lk_score, lk_score_cos)


# plt.figure(figsize=(8, 6))
# sns.heatmap(cosine_sim_df, annot=False, cmap='coolwarm')
# plt.title('Euclidean Distance Matrix (Restaurants)')
# plt.xlabel('Restaurants')
# plt.ylabel('Restaurants')
# plt.show()

# plt.figure(figsize=(8, 6))
# sns.heatmap(euclidean_dist_df, annot=False, cmap='coolwarm')
# plt.title('Euclidean Distance Matrix (Restaurants)')
# plt.xlabel('Restaurants')
# plt.ylabel('Restaurants')
# plt.show()


# print("\nCosine Similarity Matrix:")
# print(cosine_sim_df)

# print("\nEuclidean Distance Matrix:")
# print(euclidean_dist_df)