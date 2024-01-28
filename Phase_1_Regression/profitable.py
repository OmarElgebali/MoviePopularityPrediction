import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Relation between profit, Profitable and vote_average
# """
movies_dataset = pd.read_csv('movies-regression-dataset.csv')
bonus_dataset = pd.read_csv('movies-credit-students-train.csv')
movies = bonus_dataset.merge(movies_dataset, left_on='movie_id', right_on='id', how='left')
movies = movies.drop(['movie_id', 'title_x'], axis=1)   # drop extra redundant columns
movies = movies.rename(columns={'title_y': 'title'})
movies['profit'] = movies['revenue'] - movies['budget']
movies['profitable'] = (movies['profit'] > 0).astype(int)
money = movies[['revenue', 'budget', 'profit', 'profitable', 'vote_average']]
corr = money.corr()  # Get the correlation between the features
plt.subplots(figsize=(12, 8))  # Correlation plot
sns.heatmap(corr, annot=True)
plt.show()
# """


