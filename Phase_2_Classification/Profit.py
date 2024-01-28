import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency

movies_dataset = pd.read_csv('movies-classification-dataset.csv')
bonus_dataset = pd.read_csv('movies-credit-students-train.csv')
movies = bonus_dataset.merge(movies_dataset, left_on='movie_id', right_on='id', how='left')
movies = movies.drop(['movie_id', 'title_x'], axis=1)  # drop extra redundant columns
movies = movies.rename(columns={'title_y': 'title'})
movies['profit'] = movies['revenue'] - movies['budget']
movies['profitable'] = (movies['profit'] > 0).astype(int)
money = movies[['revenue', 'budget', 'profit', 'profitable', 'Rate']]

# load data into a pandas dataframe

# create contingency table for each independent variable
cont_table1 = pd.crosstab(movies['revenue'], movies['Rate'])
cont_table2 = pd.crosstab(movies['budget'], movies['Rate'])
cont_table3 = pd.crosstab(movies['profit'], movies['Rate'])
cont_table4 = pd.crosstab(movies['profitable'], movies['Rate'])

# perform chi-square test on each contingency table
_, pval1, _, _ = chi2_contingency(cont_table1)
_, pval2, _, _ = chi2_contingency(cont_table2)
_, pval3, _, _ = chi2_contingency(cont_table3)
_, pval4, _, _ = chi2_contingency(cont_table4)

# print p-value for each independent variable
print('revenue  Independent_Variable1 p-value:', pval1)
print('budget  Independent_Variable2 p-value:', pval2)
print('profit  Independent_Variable3 p-value:', pval3)
print('profitable  Independent_Variable3 p-value:', pval4)

# load data into a pandas dataframe

# create a label encoder object
le = LabelEncoder()

# fit and transform the Rate column
money['Rate'] = le.fit_transform(money['Rate'])

# create an ANOVA model for each independent variable
model = ols('Rate ~ revenue + budget + profit + profitable', data=money).fit()

# print the ANOVA table for each model
print(sm.stats.anova_lm(model, typ=2))
"""
                sum_sq      df           F        PR(>F)
revenue       0.156013     1.0   0.495843   4.813865e-01
budget        0.156013     1.0   0.495843   4.813865e-01
profit        0.156013     1.0   0.495843   4.813865e-01
profitable   25.923226     1.0  82.389545   1.963754e-19
Residual    952.360110  3029.0         NaN           NaN
This output is the ANOVA table for the multiple regression model using four independent variables ("revenue", "budget", "profit", and "profitable") to predict a numerical target variable called "Rate".

Here's what each column in the output means:

sum_sq: This column shows the sum of squares, which represents the amount of variation in the target variable that is explained by each independent variable.
 The higher the value, the more variance the independent variable explains in the target variable.
df: This column shows the degrees of freedom, which is the number of independent observations in the data that are available for estimating the population parameters.
 In this case, there are four degrees of freedom (one for each independent variable).
F: This column shows the F-statistic, which is a measure of how significant the independent variable is in predicting the target variable.
 A higher F-value indicates a stronger correlation between the independent variable and the target variable.
PR(>F): This column shows the p-value associated with the F-statistic.
 It indicates the probability that the observed F-statistic could have occurred by random chance if there 
 were no real correlation between the independent variable and the target variable. Lower p-values indicate stronger 
 evidence against the null hypothesis (no correlation).
Residual: This row shows the residual sum of squares, which represents the unexplained variation in the target variable
 after accounting for the variation explained by the independent variables. The larger this value, the poorer the fit of the model to the data.
From the output, we can see that all four independent variables have low p-values (less than 0.05), indicating strong
 evidence against the null hypothesis of no correlation. The revenue, profit, and profitable variables have high F-values, indicating a strong linear relationship with the target variable. However,
the budget variable has a low F-value, indicating a weak linear relationship with the target variable.


"""
