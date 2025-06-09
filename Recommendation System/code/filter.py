import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
# distance

def euclidean_dist(lat1, lon1, lat2, lon2):
    return (lat1 - lat2)**2 + (lon1 - lon2) ** 2

def cosine_dist(lat1, lon1, lat2, lon2):
    return np.dot(lat1 - lat2, lon1 - lon2) / (np.linalg.norm(lat1 - lat2) * np.linalg.norm(lon1 - lon2))

# Note that you will need to compute a numeric embedding of the categorical variables.
rst = pd.read_excel('../data/ERR.xlsx', sheet_name='Restaurants')
rev = pd.read_excel('../data/ERR.xlsx', sheet_name='Reviews')
missing_vals = ['La Principal', 'Todoroki Sushi','World Market',"Clare's Korner"]
rev = rev[~rev['Restaurant Name'].isin(missing_vals)]
print(rev.head())
# use the one-hot encoder methods for tokenization.
categorical_col = ['Cuisine', 'Open After 8pm?']
rst = pd.get_dummies(rst, columns=categorical_col)
# Negative longitude means the location is west of the Prime Meridian (0° longitude line, which passes through Greenwich, London)
rst.loc[rst['Longitude'] == ', -87.67915254400617', 'Longitude'] = -87.6791
rst['Longitude'] = rst['Longitude'].astype('float64')
rst['Longitude'] = rst['Longitude'].apply(lambda x: -x if x > 0 else x)
# standardize for numerical data
scaler = StandardScaler()
numerical_col = ['Latitude', 'Longitude', 'Average Cost']
rst.loc[:, numerical_col] = scaler.fit_transform(rst.loc[:, numerical_col])
# compute the 
n = len(rst)
distance_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        distance_matrix[i, j] = euclidean_dist(
            rst.loc[i, 'Latitude'], rst.loc[i, 'Longitude'],
            rst.loc[j, 'Latitude'], rst.loc[j, 'Longitude']
        )
rst_dis = pd.DataFrame(distance_matrix, index=rst['Restaurant Name'], columns=rst['Restaurant Name'])

# # 1. Plot Euclidean Distance Heatmap
# plt.figure(figsize=(14, 10))
# sns.set_palette("flare")
# sns.heatmap(
#     rst_dis,
#     annot=False,
#     cmap='flare',
#     xticklabels=True,
#     yticklabels=True,
#     cbar_kws={"shrink": 0.7}
# )
# plt.title('Euclidean Distance Matrix (Restaurants)', fontsize=16)
# plt.xlabel('Restaurants', fontsize=12)
# plt.ylabel('Restaurants', fontsize=12)
# plt.xticks(fontsize=8, rotation=90)
# plt.yticks(fontsize=8)
# plt.tight_layout()
# plt.show()

# 2. Plot Pearson Similarity Heatmap
cos_rst = rst.copy()
cos_rst.drop(columns='Brief Description', inplace=True)
cos_rst = cos_rst.set_index('Restaurant Name')

# Compute Pearson correlation (similarity)
pearson_similarity = cos_rst.T.corr()

# plt.figure(figsize=(14, 12))
# sns.set_palette("flare")
# sns.heatmap(
#     pearson_similarity,
#     annot=False,
#     cmap='flare',
#     xticklabels=True,
#     yticklabels=True,
#     cbar_kws={"shrink": 0.7}
# )
# plt.title('Pearson Similarity Between Restaurants', fontsize=12)
# plt.xlabel('Restaurants', fontsize=8)
# plt.ylabel('Restaurants', fontsize=8)
# plt.xticks(fontsize=5, rotation=90)
# plt.yticks(fontsize=5)
# plt.tight_layout()
# plt.show()

# report the results for the pairs ('Peppercorns Kitchen', 'Epic Burger') and ('Peppercorns Kitchen', 'Lao Sze Chuan').
# location methods, index = 'pepp'
euci_pepp = rst_dis.loc['Peppercorns Kitchen', ['Epic Burger', 'Lao Sze Chuan']]
# print(euci_pepp)
pear_pepp = pearson_similarity.loc['Peppercorns Kitchen', ['Epic Burger', 'Lao Sze Chuan']]
compared_cols = ['Peppercorns Kitchen', 'Epic Burger', 'Lao Sze Chuan']
pd.set_option('display.max_columns', None)
pbl_data = rst[rst['Restaurant Name'].isin(compared_cols)]
# print(pbl_data)



# build a recommendation engine
# top_10_rst = euc_rst.head(10).index
def recommend_rst(name, df, rev, ascending: bool):
    review_list = rev[rev['Reviewer Name'] == name]
    
    if review_list.empty:
        print(f"No reviews found for {name}.")
        return None

    review_list = review_list.sort_values(by='Rating', ascending=False)
    best_rst = review_list.loc[:, 'Restaurant Name'].values[0]  # 取具体字符串
    print(f"Best restaurant liked by {name}: {best_rst}")
    

    euc_rst = rst_dis.loc[:, best_rst].reset_index()
    euc_rst.columns = ['Name', 'Similarity']
    
    euc_rst = euc_rst.sort_values(by='Similarity', ascending=ascending)
    
    return euc_rst.head(11)

# 例子：使用
name = 'Willie Jacobsen'

euc_recommend = recommend_rst(name, rst_dis, rev, ascending=True)
cos_recommend = recommend_rst(name, cos_rst, rev, ascending=False)

# print("\nTop-10 based on Euclidean Distance:")
# print(euc_recommend)

# print("\nTop-10 based on Cosine Similarity:")
# print(cos_recommend)

# Step 1: Pick Top-N (e.g., Top-5)
N = 10

rev_rating = rev.groupby('Restaurant Name')['Rating'].mean().reset_index()
rev_rating.columns = ['Name', 'Avg_Rating']
topN_euclidean = euc_recommend.sort_values(by='Similarity', ascending=True).head(N)
topN_cosine = cos_recommend.sort_values(by='Similarity', ascending=False).head(N)

topN_euclidean = pd.merge(topN_euclidean, rev_rating, on = 'Name', how = 'right')
topN_euclidean.dropna(inplace=True)
topN_cosine = pd.merge(topN_cosine, rev_rating, on = 'Name', how = 'right')
topN_cosine.dropna(inplace=True)
print('topN_cosine', topN_cosine.isnull().sum())
euclidean_avg_rating = topN_euclidean['Avg_Rating'].mean()
cosine_avg_rating = topN_cosine['Avg_Rating'].mean()

print(f"Top-{N} Average Rating (Euclidean Distance): {euclidean_avg_rating:.3f}")
print(f"Top-{N} Average Rating (Cosine Similarity): {cosine_avg_rating:.3f}")

# Determine which method generates better recommendations based on average rating
if euclidean_avg_rating > cosine_avg_rating:
    print(f"Euclidean Distance method generates better recommendations with an average rating of {euclidean_avg_rating:.3f}.")
elif cosine_avg_rating > euclidean_avg_rating:
    print(f"Cosine Similarity method generates better recommendations with an average rating of {cosine_avg_rating:.3f}.")
else:
    print("Both methods generate recommendations with the same average rating.")

result_table = pd.DataFrame({
    'Method': ['Euclidean', 'Cosine'],
    'Top-N Average Rating': [euclidean_avg_rating, cosine_avg_rating]
})

print("\nComparison Table:")
print(result_table)

# Step 4: Plot results
plt.bar(result_table['Method'], result_table['Top-N Average Rating'])
plt.title(f"Comparison of Top-{N} Recommendation Quality")
plt.ylabel("Average Rating")
plt.show()






