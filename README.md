



# Titanic: Machine Learning from Disaster


### Predict survival on the Titanic and get familiar with ML basics

Using Titanic Data Set from kaggle (https://www.kaggle.com/c/titanic/data)


```python
import pandas as pd
import matplotlib.pyplot as plt

test=pd.read_csv("test.csv")

test_shape= test.shape

train=pd.read_csv("train.csv")

train_shape= train.shape


```


```python
test.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>Wirz, Mr. Albert</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>8.6625</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>12.2875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
Sex_pivot= train.pivot_table(index="Sex", values="Survived")
Sex_pivot.plot.bar()
plt.show()
```





```python
Pclass_pivot= train.pivot_table(index="Pclass", values="Survived")
Pclass_pivot.plot.bar()
plt.show()
```




```python
print(train["Age"].describe())
```

    count    714.000000
    mean      29.699118
    std       14.526497
    min        0.420000
    25%       20.125000
    50%       28.000000
    75%       38.000000
    max       80.000000
    Name: Age, dtype: float64
    


```python
survived= train[train["Survived"]==1]
died=train[train["Survived"]==0]
survived["Age"].plot.hist(alpha=0.5,color="green", bins=50)

died["Age"].plot.hist(alpha=0.5,color="red", bins=50)
plt.legend(["Survived","Died"])
plt.show()
```



Missing  -1 to 0
Infant 0 to 5
Child 5 to 12
Teenage 12 to 18
Young Adult 18 to 35
Adult 35 to 60
Senior 60 to 100


```python
def process_age(df,cut_points,label_names):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df
cut_points=[-1,0,5,12,18,35,60,100]
label_names= ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]
train= process_age(train,cut_points,label_names)
test= process_age(test,cut_points,label_names)

pivot=train.pivot_table(index="Age_categories", values="Survived")
pivot.plot.bar()
plt.show()
```


![png](output_10_0.png)



```python
train["Pclass"].value_counts()
```




    3    491
    1    216
    2    184
    Name: Pclass, dtype: int64



While the class of each passenger certainly has some sort of ordered relationship, the relationship between each class is not the same as the relationship between the numbers 1, 2, and 3. For instance, class 2 isn't "worth" double what class 1 is, and class 3 isn't "worth" triple what class 1 is.

In order to remove this relationship, we can create dummy columns for each unique value in Pclass.


The code below creates a function to create the dummy columns for the Pclass column and add it back to the original dataframe. It then applies that function the train and test dataframes.


```python
def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df

train = create_dummies(train,"Pclass")
test = create_dummies(test,"Pclass")

train = create_dummies(train,"Sex")
test = create_dummies(test,"Sex")

train = create_dummies(train,"Age_categories")
test = create_dummies(test,"Age_categories")
```

Now that our data has been prepared, we are ready to train our first model. The first model we will use is called Logistic Regression, which is often the first model you will train when performing classification.

We will be using the scikit-learn library as it has many tools that make performing machine learning easier.


Each model in scikit-learn is implemented as a separate class and the first step is to identify the class we want to create an instance of. In this case, we  use the LogisticRegression class.


```python
columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
       'Age_categories_Missing','Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior']

from sklearn.linear_model import LogisticRegression
lr= LogisticRegression()
lr.fit(train[columns], train['Survived'])
```

    C:\Users\USER\Anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='warn',
              n_jobs=None, penalty='l2', random_state=None, solver='warn',
              tol=0.0001, verbose=0, warm_start=False)



We have trained the  machine learning model. The next step is to find out how accurate our model is, and to do that, I have to make some predictions.


The convention in machine learning is to call these two parts train and test. This can become confusing, since I already have our test dataframe that I will eventually use to make predictions to submit. To avoid confusion, from here on, I am going to call this Kaggle 'test' data holdout data, which is the technical name given to this type of data used for final predictions.


The scikit-learn library has a handy `model_selection.train_test_split()` function that we can use to split our data. `train_test_split()` accepts two parameters, X and y, which contain all the data we want to train and test on, and returns four objects: `train_X`, `train_y`, `test_X`, `test_y`.


`test_size`, which lets us control what proportions our data are split into, and random_state. The `train_test_split()` function randomizes observations before dividing them, and setting a random seed means that our results will be reproducible, which is important if you are collaborating, or need to produce consistent results each time


```python
holdout = test #  from now on we will refer to this dataframe as the holdout data
from sklearn.model_selection import train_test_split


columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
       'Age_categories_Missing','Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior']
all_X=train[columns]
all_y=train['Survived']

train_X,test_X,train_y,test_y= train_test_split(all_X,all_y, test_size=0.2, random_state=0)


```

Now we have fit our model, we can use the `LogisticRegression.predict()` method to make predictions.


The `predict()` method takes a single parameter X, a two dimensional array of features for the observations I wish to predict. X must have the exact same features as the array we used to fit our model. The method returns single dimensional array of predictions.


Measuring the Accuracy:


```python
from sklearn.metrics import accuracy_score
lr= LogisticRegression()
lr.fit(train_X,train_y)
predictions=lr.predict(test_X)
accuracy = accuracy_score(test_y, predictions)
print(accuracy)
```

    0.8100558659217877
    

    C:\Users\USER\Anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    

The model has an accuracy score of 81.0% when tested against our 20% test set.

To  better understand the real performance of our model, we can use a technique called cross validation to train and test our model on different splits of our data, and then average the accuracy scores.

The most common form of cross validation, and the one we will be using, is called k-fold cross validation. 'Fold' refers to each different iteration that we train our model on, and 'k' just refers to the number of folds. 

I will use scikit-learn's model_selection.cross_val_score() function to automate the process. The basic syntax for cross_val_score() is:

`cross_val_score(estimator, X, y, cv=None)`

- estimator is a scikit-learn estimator object, like the LogisticRegression() objects we have been creating.
- X is all features from our data set.
- y is the target variables.
- cv specifies the number of folds.


The function returns a numpy ndarray of the accuracy scores of each fold.


```python
from sklearn.model_selection import cross_val_score
import numpy as np
lr= LogisticRegression()
scores=cross_val_score(lr, all_X, all_y, cv=10)
accuracy=np.mean(scores)
print(scores)
print(accuracy)
```

    C:\Users\USER\Anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    C:\Users\USER\Anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    C:\Users\USER\Anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    C:\Users\USER\Anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    C:\Users\USER\Anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    C:\Users\USER\Anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    C:\Users\USER\Anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    C:\Users\USER\Anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    C:\Users\USER\Anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    

    [0.8        0.81111111 0.7752809  0.87640449 0.80898876 0.78651685
     0.76404494 0.76404494 0.83146067 0.80681818]
    0.8024670865963002
    

    C:\Users\USER\Anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    

From the results of the k-fold validation, you can see that the accuracy number varies with each fold - ranging between 76.4% (0.764) and 87.6% (0.876). This demonstrates why cross validation is important.

From observation, the  average accuracy score was 80.2%, which is not far from the 81.0% we got from our simple train/test split


I am now ready to use the model we have built to train our final model and then make predictions on our unseen holdout data, or what Kaggle calls the 'test' data set.


```python
columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
       'Age_categories_Missing','Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior']
lr= LogisticRegression()
lr.fit(all_X,all_y)
holdout_predictions=lr.predict(holdout[columns])
```

    C:\Users\USER\Anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    


```python
holdout_ids = holdout["PassengerId"]
submission_df = {"PassengerId": holdout_ids,
                 "Survived": holdout_predictions}
submission = pd.DataFrame(submission_df)

submission.to_csv("submission.csv", index=False)

```

I have successfully made the prediction. The `submission.csv` can be found in this [link](https://github.com/makozi/Titanic-Machine-Learning-from-Disaster).

### Conclusion


There are many things that can be done to  improve the accuracy of our model. Here are some that we will cover in the next two missions of this course:

#### Improving the features:
- Feature Engineering: Create new features from the existing data.
- Feature Selection: Select the most relevant features to reduce noise and overfitting.

#### Improving the model:
- Model Selection: Try a variety of models to improve performance.
- Hyperparameter Optimization: Optimize the settings within each particular machine learning model.


```python

```




Follow me on [Twitter](https://twitter.com/marizu_makozi).
