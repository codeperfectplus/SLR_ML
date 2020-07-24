This article was originally published at [Codeperfectplus](http://codeperfectplus.herokuapp.com/what-is-simple-linear-regression).

## Introduction

The term regression was first applied to statistics by the polymath Francis Galton. Galton is a major figure in the development of statistics and genetics.

Linear Regression is one of the simplest machine learning algorithms that map the relationship between two variable by fitting a linear equation to observed data. One variable is considered to be an explanatory variable, and the other is considered to be a dependent variable.

## Simple Linear Regression

In simple linear regression when we have a single input. and we have to obtain a line that best fits the data. The best fit line is the one for which total prediction error (all data points) are as small as possible. Error is the distance between the point to the regression line.

E.g: Relationship between hours of study and marks obtained. The goal is to find the relationship between the hours of study and marks obtained by the student. for that, we have to find a linear equation between these two variables.

$ y_(predict) = b_0 + b_1.x $

Error is the difference between predicted and actual value. to reduce the error and find the best fit line we have to find value for bo and b1.

$$ Error = (Y_p - Y_a) ^ 2 $$

for finding a best fit line value of bo and b1 must be that minimize the error.error is the difference between predicted and actual output.

<div class="container text-center my-3">
<img src="https://images.pexels.com/photos/3815585/pexels-photo-3815585.jpeg?auto=compress&cs=tinysrgb&dpr=3&h=750&w=1260" class="img-fluid border border-primary text-center" alt="machine learning">
</div>

## Assumptions of simple linear regression
Simple linear regression is a parametric test, meaning that it makes certain assumptions about the data. These assumptions are:

- Homogeneity of variance (homoscedasticity): the size of the error in our prediction doesn’t change significantly across the values of the independent variable.
- Independence of observations: the observations in the dataset were collected using statistically valid sampling methods and there are no hidden relationships among observations.
- Normality: The data follows a normal distribution.
- The relationship between the independent and dependent variable is linear: the line of best fit through the data points is a straight line (rather than a curve or some sort of grouping factor).
If your data do not meet the assumptions of homoscedasticity or normality, you may be able to use a nonparametric test instead, such as the Spearman rank test.

## Linear Regression example

Let us now apply Machine Learning to train a dataset to predict the **_Salary_** from **_Years of Experience_**.

## Importing Libraries and Datasets

Three python libraries will be used in the code.

-  pandas
- matplotlib
- sklearn

```python
## Importing the necessary Libraries
import pandas as pd
import matplotlib.pyplot as plt

## Importing the datasert
data = pd.read_csv("https://raw.githubusercontent.com/codePerfectPlus/DataAnalysisWithJupyter/master/SalaryVsExperinceModel/Salary.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
data.head()
```

## Split the Data 

In this step, Split the dataset into the Training set, on which the Linear Regression model will be trained and the Test set, on which the trained model will be applied to visualize the results. In this the `test_size=0.3` denotes that **30%** of the data will be kept as the **Test set** and the remaining **70%** will be used for training as the **Training set**.

```python
## Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state=12)
```

## Fit and Predict
Import `LinearRegression` from linear_model and assigned it to the variable lr. `lr.fit()` used to learn from data and `lr.predict()` used to predict basis on learn data.

```python
## Fit and Predict Model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_X, train_y)

predicted_y = lr.predict(test_X)
```

## Evaluate and Visualize

Create a pandas dataframe of predict and actual values and visualize the dataset.

```python
df = pd.DataFrame({'Real Values':test_y,"Predict Value":predicted_y})
df.head()

## Step 6: Visualising the Test set results

plt.scatter(test_X, test_y, color = 'red')
plt.scatter(test_X, predicted_y, color = 'green')
plt.plot(train_X, lr.predict(train_X), color = 'black')
plt.title('Salary vs Experience (Result)')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()
```

<div class="container text-center my-3">
<img src="https://raw.githubusercontent.com/codePerfectPlus/code/master/slr.jpeg" alt="simple linear regression" class="img-fluid border border-primary text-center">
</div>


## Conclusion

Thus in this story, we have successfully been able to build a  **_Simple Linear Regression_**  model that predicts the ‘Salary’ of an employee based on their ‘Years of Experience’ and visualize the results.

Download full code [GitHub](https://github.com/codeperfectplus/code).

```python
## Importing the Libraries
import pandas as pd
import matplotlib.pyplot as plt

## Importing the dataset
data = pd.read_csv("https://raw.githubusercontent.com/codePerfectPlus/DataAnalysisWithJupyter/master/SalaryVsExperinceModel/Salary.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
data.head()

## Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state=12)

## Fit and Predict Model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_X, train_y)

## Predicting the Test set results
predicted_y = lr.predict(test_X)

## Comparing the Test Set with Predicted Values
df = pd.DataFrame({'Real Values':test_y,"Predict Value":predicted_y})
df.head()

## Visualising the Test set results

plt.scatter(test_X, test_y, color = 'red')
plt.scatter(test_X, predicted_y, color = 'green')
plt.plot(train_X, lr.predict(train_X), color = 'black')
plt.title('Salary vs Experience (Result)')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()
```

### More Articles by Author

- [Deploy Your First Django App With Heroku](http://codeperfectplus.herokuapp.com/deploy-your-first-django-app-with-heroku)
- [Logistic Regression for Machine Learning Problem](http://codeperfectplus.herokuapp.com/logistic-regression-for-machine-learning-problem)
- [5 Tips for Computer Programming Beginners](http://codeperfectplus.herokuapp.com/5-tips-for-computer-programming-beginners)
- [What is Simple Linear Regression?](http://codeperfectplus.herokuapp.com/what-is-simple-linear-regression)
- [Introduction to Machine Learning and it's Type.](http://codeperfectplus.herokuapp.com/introduction-to-machine-learning-and-its-type)
- [Difference Between Machine Learning and Artificial Intelligence](http://codeperfectplus.herokuapp.com/difference-between-machine-learning-and-AI)
