import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Assuming your DataFrame is already loaded and is named movies_dataset:
# Create a new column called 'homepage_exists' initialized with 0's
movies_dataset = pd.read_csv('movies-regression-dataset.csv')

movies_dataset['homepage_exists'] = 0
movies_dataset.loc[~movies_dataset['homepage'].isnull(), 'homepage_exists'] = 1
movies_dataset['homepage'] = movies_dataset['homepage_exists']
movies_dataset.drop(columns='homepage_exists')
movies_dataset.to_csv("movies_dataset.csv", index=False)

# corr_matrix = movies_dataset[['homepage_exists', 'vote_average']].corr()
# import seaborn as sns

# Create the heatmap using Seaborn
# sns.heatmap(corr_matrix, annot=True)

# Show the plot
# plt.show()
