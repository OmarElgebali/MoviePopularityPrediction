from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import ols
import pandas as pd
#
#
# load data into a pandas dataframe
movies_dataset = pd.read_csv('movies-classification-dataset.csv')
bonus_dataset = pd.read_csv('movies-credit-students-train.csv')
movies = bonus_dataset.merge(movies_dataset, left_on='movie_id', right_on='id', how='left')
movies = movies.drop(['movie_id', 'title_x'], axis=1)  # drop extra redundant columns
movies = movies.rename(columns={'title_y': 'title'})
movies_train, movies_test = train_test_split(movies, test_size=0.20, shuffle=True, random_state=42)
m = movies_train
m['profit'] = m['revenue'] - m['budget']
m['profitable'] = (m['profit'] > 0).astype(int)
money = m[['revenue', 'budget', 'profit', 'profitable', 'Rate']]

# movies['profit'] = movies['revenue'] - movies['budget']
# movies['profitable'] = (movies['profit'] > 0).astype(int)
# money = movies[['revenue', 'budget', 'profit', 'profitable', 'Rate']]
def anova(dataset):
    from sklearn.feature_selection import f_classif
    # Separate the target variable from the input features
    X = dataset.drop("Rate", axis=1)
    y = dataset["Rate"]
    # Compute the ANOVA F-value and p-value for each feature
    f_values, p_values = f_classif(X, y)
    # Create a dataframe to store the results
    results = pd.DataFrame({"Feature": X.columns, "F-Value": f_values, "p-value": p_values})
    # Sort the dataframe by descending F-values
    results.sort_values(by="F-Value", ascending=False, inplace=True)
    print(results)


# # create a label encoder object
# le = LabelEncoder()
#
# # fit and transform the Rate column
# money['Rate'] = le.fit_transform(money['Rate'])
#
# # create an ANOVA model for each independent variable
# model = ols('Rate ~  revenue + budget + profit + profitable', data=money).fit()
# # model = ols('Rate ~ revenue + budget', data=money).fit()
#
# # print the ANOVA table for each model
# print(sm.stats.anova_lm(model, typ=2))

"""
                sum_sq      df          F        PR(>F)
revenue       0.156013     1.0   0.495843  4.813865e-01
budget        0.156013     1.0   0.495843  4.813865e-01
profit        0.156013     1.0   0.495843  4.813865e-01
profitable   25.923226     1.0  82.389545  1.963754e-19
Residual    952.736531  3028.0        NaN           NaN

ANOVA stands for Analysis of Variance. It is a statistical method used to test the difference between two or more groups (or treatments) by analyzing the variation and means of their respective samples.

This output is the ANOVA table for the multiple regression model using four independent variables ("revenue", "budget", "profit", and "profitable") to predict a numerical target variable called "Rate".

===============================================================================================================================

Here's what each column in the output means:

sum_sq: This column shows the sum of squares, which represents the amount of variation in the target variable that is explained by each 
feature. The higher the value, the more variance the feature explains in the target variable.

df: This column shows the degrees of freedom, which is the number of independent observations in the data that are available for estimating 
the population parameters. In this case, there are four degrees of freedom (one for each feature).

F: This column shows the F-statistic, which is a measure of how significant the feature is in predicting the target variable. 
A higher F-value indicates a stronger correlation between the feature and the target variable.

PR(>F): This column shows the p-value associated with the F-statistic. It indicates the probability that the observed F-statistic could have 
occurred by random chance if there were no real correlation between the feature and the target variable. Lower p-values indicate 
stronger evidence against the null hypothesis (no correlation).

Residual: This row shows the residual sum of squares, which represents the unexplained variation in the target variable after accounting for 
the variation explained by the independent variables. The larger this value, the poorer the fit of the model to the data.

===============================================================================================================================

Based on the ANOVA table, we can see that the "profitable" feature has a significant impact on the target variable "Rate" since it has a 
high F-value and a very low p-value (p < 0.001). This suggests that whether a movie is profitable or not 
(as determined by the "Profitable" feature) is a strong predictor of its rating.

However, the other three features - "revenue", "budget", and "profit" - do not appear to have a significant impact on the target 
variable "Rate". This can be seen from their low F-values and high p-values (all above 0.4).

Given this information, it may be reasonable to extract the "Profit" feature (which is simply the difference between revenue 
and budget) and the "Profitable" feature for use in predicting the movie rating. However, it's worth noting that there may be 
other factors not captured in these features that also influence the movie's rating.
"""

"""
      Feature    F-Value       p-value
3  profitable  69.920104  3.006278e-30
2      profit  56.007388  1.666447e-24
0     revenue  43.035227  4.308597e-19
1      budget  11.204204  1.433636e-05

      Feature    F-Value       p-value
3  profitable  88.006926  7.047089e-38
2      profit  74.434200  2.771213e-32
0     revenue  58.623864  1.047671e-25
1      budget  11.616859  9.421214e-06
"""
