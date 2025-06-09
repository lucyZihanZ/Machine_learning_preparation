import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# load the file

rev = pd.read_csv('../data/cleaned_err.csv', index_col=0)
rev2 = pd.read_csv('../data/match.csv')

highest_score = rev2['avg_Scores'].max()
highest_rst = rev2.loc[rev2['avg_Scores'] == highest_score, ['Restaurant Name', 'avg_Scores']]
avg_score = rev2['avg_Scores'].median()

count_rev = rev.groupby('Restaurant Name')['Review Text'].count().reset_index()
count_rev.columns = ['Restaurant Name', 'Number of Reviews']
count_rev = count_rev.sort_values(by='Number of Reviews', ascending=False)
print(count_rev)
median_rev = count_rev['Number of Reviews'].median()
sorted_score = sorted(rev2['avg_Scores'].values)

rev['rst_scores'] = rev['Restaurant Name'].map(rev.groupby('Restaurant Name')['Rating'].mean())

def popularity_scores(avg_score, num_of_reviews, total_review=919) -> float:
    return avg_score * (num_of_reviews / total_review)

rev = pd.merge(rev, count_rev, on='Restaurant Name')
rev['pscores'] = rev.apply(lambda x: popularity_scores(x['rst_scores'], x['Number of Reviews']), axis=1)
rev_groups = rev.groupby(['Restaurant Name'])['pscores'].mean()

rev = rev.sort_values(by='pscores', ascending=False)

rec_styles = ['Spanish', 'Chinese', 'Mexican', 'Coffee']
filtered_rev = rev[rev['Cuisine'].isin(rec_styles)]

top_2_per_cuisine = (
    filtered_rev
    .groupby(['Cuisine', 'Restaurant Name'])
    .agg(
        pscores=('pscores', 'mean'),
        Rating_mean=('Rating', 'mean'),
        Rating_count=('Rating', 'count')
    )
    .reset_index()
)

top_2_per_cuisine = (
    top_2_per_cuisine
    .sort_values(['Cuisine', 'pscores'], ascending=[True, False])
    .groupby('Cuisine', group_keys=False)
    .head(2)
)
print(top_2_per_cuisine)

shrkg_rev = rev.groupby(['Cuisine', 'Restaurant Name'])['Rating'].agg(['mean', 'count']).reset_index()
shrkg_rev.columns = ['Cuisine', 'Restaurant Name', 'Mean Rating', 'Num Ratings']

N_mu = shrkg_rev['Num Ratings'].mean()
mu_s = shrkg_rev['Mean Rating'].mean()

shrkg_rev['Shrinkage Rating'] = (
    (N_mu * mu_s + shrkg_rev['Num Ratings'] * shrkg_rev['Mean Rating']) /
    (N_mu + shrkg_rev['Num Ratings'])
)

shrkg_rev['Percentage of change'] = (shrkg_rev['Shrinkage Rating'] - shrkg_rev['Mean Rating']) / shrkg_rev['Mean Rating']
shrkg_rev = shrkg_rev.sort_values(by='Percentage of change', ascending=False)

top_3_increase = shrkg_rev.head(5)
top_3_decrease = shrkg_rev.iloc[-7:-2]
print(top_3_decrease)
print(top_3_increase)


plt.figure(figsize=(15,6))
sns.color_palette("Spectral")
sns.barplot(x='Percentage of change', y='Restaurant Name', hue = 'Restaurant Name', data=top_3_increase)
plt.title('Top 5 Increase in Rating')
plt.xlabel('Percentage of Change')
plt.ylabel('Restaurant Name')
plt.show()

plt.figure(figsize=(15,6))
sns.barplot(x='Percentage of change', y='Restaurant Name', hue = 'Restaurant Name', data=top_3_decrease)
plt.title('Top 5 Decrease in Rating')
plt.xlabel('Percentage of Change')
plt.ylabel('Restaurant Name')
plt.show()
















