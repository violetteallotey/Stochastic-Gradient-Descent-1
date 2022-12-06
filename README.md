# Stochastic-Gradient-Descent-1
Week 11 Project

# Data Preparation

### Read Data


```python
# importing libraries and reading the housing data

import pandas as pd

import numpy as np

housing_data=pd.read_csv('housing.csv')

housing_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-122.23</td>
      <td>37.88</td>
      <td>41.0</td>
      <td>880.0</td>
      <td>129.0</td>
      <td>322.0</td>
      <td>126.0</td>
      <td>8.3252</td>
      <td>452600.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-122.22</td>
      <td>37.86</td>
      <td>21.0</td>
      <td>7099.0</td>
      <td>1106.0</td>
      <td>2401.0</td>
      <td>1138.0</td>
      <td>8.3014</td>
      <td>358500.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-122.24</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1467.0</td>
      <td>190.0</td>
      <td>496.0</td>
      <td>177.0</td>
      <td>7.2574</td>
      <td>352100.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1274.0</td>
      <td>235.0</td>
      <td>558.0</td>
      <td>219.0</td>
      <td>5.6431</td>
      <td>341300.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1627.0</td>
      <td>280.0</td>
      <td>565.0</td>
      <td>259.0</td>
      <td>3.8462</td>
      <td>342200.0</td>
      <td>NEAR BAY</td>
    </tr>
  </tbody>
</table>
</div>



### Spliting the data


```python
#spliting the dataset into train and test sets
#import the train_test_split() method from sklearn.model_selection
from sklearn.model_selection import train_test_split
train,test=train_test_split(housing_data,test_size=0.2,random_state=42)
train.shape,test.shape
```




    ((16512, 10), (4128, 10))



# Cleaning the data


```python
# cleaning the training set by handling missing values

train.isnull().sum()
```




    longitude             0
    latitude              0
    housing_median_age    0
    total_rooms           0
    total_bedrooms        0
    population            0
    households            0
    median_income         0
    median_house_value    0
    ocean_proximity       0
    dtype: int64




```python
#using imputer method to replace missing numerical values with median
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy='median')

#dropping the categorical value column
train_num=train.drop('ocean_proximity',axis=1)

#fitting the imputer to the numerical training dataset
imputer.fit(train_num)
```




    SimpleImputer(strategy='median')




```python
#applying the strategy to our numerical training data using the transform()
x=imputer.transform(train_num)
```


```python
#the results is an array to convert it to a dataframe apply pd
train_prepared=pd.DataFrame(x,columns=train_num.columns)
train_prepared.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-117.03</td>
      <td>32.71</td>
      <td>33.0</td>
      <td>3126.0</td>
      <td>627.0</td>
      <td>2300.0</td>
      <td>623.0</td>
      <td>3.2596</td>
      <td>103000.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-118.16</td>
      <td>33.77</td>
      <td>49.0</td>
      <td>3382.0</td>
      <td>787.0</td>
      <td>1314.0</td>
      <td>756.0</td>
      <td>3.8125</td>
      <td>382100.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-120.48</td>
      <td>34.66</td>
      <td>4.0</td>
      <td>1897.0</td>
      <td>331.0</td>
      <td>915.0</td>
      <td>336.0</td>
      <td>4.1563</td>
      <td>172600.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-117.11</td>
      <td>32.69</td>
      <td>36.0</td>
      <td>1421.0</td>
      <td>367.0</td>
      <td>1418.0</td>
      <td>355.0</td>
      <td>1.9425</td>
      <td>93400.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-119.80</td>
      <td>36.78</td>
      <td>43.0</td>
      <td>2382.0</td>
      <td>431.0</td>
      <td>874.0</td>
      <td>380.0</td>
      <td>3.5542</td>
      <td>96500.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#handling categorical data

train_ctg=train[['ocean_proximity']]
train_ctg.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14196</th>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>8267</th>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>17445</th>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>14265</th>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>2271</th>
      <td>INLAND</td>
    </tr>
  </tbody>
</table>
</div>




```python
# import onehotencoder from sklearn.preprocessing,to convert the categoriccal data to one hot vector
from sklearn.preprocessing import OneHotEncoder
ctg_encoder=OneHotEncoder()
train_ctg_1hot=ctg_encoder.fit_transform(train_ctg)
train_ctg_1hot
```




    <16512x5 sparse matrix of type '<class 'numpy.float64'>'
    	with 16512 stored elements in Compressed Sparse Row format>




```python
#convert the resulting sparse matrix to a numpy array
train_ctg_1hot.toarray()
```




    array([[0., 0., 0., 0., 1.],
           [0., 0., 0., 0., 1.],
           [0., 0., 0., 0., 1.],
           ...,
           [1., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0.],
           [0., 0., 0., 1., 0.]])



### Feature Engineering


```python
train['rooms_per_household'] =train['total_rooms']/train['households']
train['bedrooms_per_rooms']=train['total_bedrooms']/train['total_rooms']
train['population_per_household']=train['population']/train['households']
```


```python
#checking the correlation of the new created columns to the median house value
corr_matrix= train.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)
```




    median_house_value          1.000000
    median_income               0.690647
    rooms_per_household         0.158485
    total_rooms                 0.133989
    housing_median_age          0.103706
    households                  0.063714
    total_bedrooms              0.047980
    population_per_household   -0.022030
    population                 -0.026032
    longitude                  -0.046349
    latitude                   -0.142983
    bedrooms_per_rooms         -0.257419
    Name: median_house_value, dtype: float64



### Feature Scaling


```python
#spliting data into input and output(i.e features and labels )
#creating the output data.Making a copy so that any changes made in this o not affect the original dataset
train_labels=train['median_house_value'].copy()

#creating the input data
train=train.drop('median_house_value',axis=1)

#droping the categorical value column so as to work with the numerical values including the added columns
train_num=train.drop('ocean_proximity',axis=1)
train_labels.head()
```




    14196    103000.0
    8267     382100.0
    17445    172600.0
    14265     93400.0
    2271      96500.0
    Name: median_house_value, dtype: float64




```python
# since there are many  data transformation steps that need to be executed 
# in the right order, the Pipeline class in sklearn will help with such sequence of transformation

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline=Pipeline([('imputer',SimpleImputer(strategy='median')),('std_scaler',StandardScaler())])
train_num_tr=num_pipeline.fit_transform(train_num)
```


```python
'''scikit has introduced a new class called the ColumnTransfomer 
that transforms both the numerical and categorical values at the same time'''
from sklearn.compose import ColumnTransformer
num_attributes=list(train_num)
ctg_attributes=['ocean_proximity']

full_pipeline=ColumnTransformer([('num',num_pipeline,num_attributes),('ctg',OneHotEncoder(),ctg_attributes)])
train_prepared=full_pipeline.fit_transform(train)
train_prepared
```




    array([[ 1.27258656, -1.3728112 ,  0.34849025, ...,  0.        ,
             0.        ,  1.        ],
           [ 0.70916212, -0.87669601,  1.61811813, ...,  0.        ,
             0.        ,  1.        ],
           [-0.44760309, -0.46014647, -1.95271028, ...,  0.        ,
             0.        ,  1.        ],
           ...,
           [ 0.59946887, -0.75500738,  0.58654547, ...,  0.        ,
             0.        ,  0.        ],
           [-1.18553953,  0.90651045, -1.07984112, ...,  0.        ,
             0.        ,  0.        ],
           [-1.41489815,  0.99543676,  1.85617335, ...,  0.        ,
             1.        ,  0.        ]])




```python
train_prepared.shape
```




    (16512, 16)



# Model Training

### Random Forest Model


```python
#Random forest regression model

from sklearn.ensemble import RandomForestRegressor
forest_reg=RandomForestRegressor()
forest_reg.fit(train_prepared, train_labels)
```




    RandomForestRegressor()




```python
#Measuring the Root Mean Squared Error of this model

from sklearn.metrics import mean_squared_error
train_predictions=forest_reg.predict(train_prepared)
forest_mse=mean_squared_error(train_labels, train_predictions)
forest_rmse=np.sqrt(forest_mse)
forest_rmse
```




    18552.659159028022



### Stochastic Gradient Descent Model


```python
from sklearn.linear_model import SGDRegressor
sgd_mdl=SGDRegressor()
sgd_mdl.fit(train_prepared,train_labels)
```




    SGDRegressor()




```python
#use some of the trainig data to test our model
some_data=train.iloc[:5]
some_data_labels=train_labels[:5]

#perform a full pipeline and transform some_data
some_data_prepared=full_pipeline.transform(some_data)

#print the predictions of our model
print('predictions:',sgd_mdl.predict(some_data_prepared))
```

    predictions: [182125.81030629 289766.87462683 246787.55072702 146237.04496245
     163585.58693537]
    


```python
#to measure rmse of this sgd model
train_predictions=sgd_mdl.predict(train_prepared)
sgd_mse=mean_squared_error(train_labels,train_predictions)
sgd_rmse= np.sqrt(sgd_mse)
sgd_rmse
```




    67691.15556449741




```python
#performing cross validation on our SGD model

from sklearn.model_selection import cross_val_score
sgd_scores=cross_val_score(sgd_mdl, train_prepared,train_labels,scoring="neg_mean_squared_error", cv=10)
sgd_rmse_scores=np.sqrt(-sgd_scores)
print('scores:',sgd_rmse_scores)
print('mean',sgd_rmse_scores.mean())
print('standard deviation',sgd_rmse_scores.std())
```

    scores: [6.51587840e+04 2.03921328e+05 9.23222784e+04 6.62083207e+04
     8.14803385e+04 6.78311271e+04 2.12782615e+05 8.61011142e+04
     1.13027706e+08 7.23380829e+04]
    mean 11397585.039783126
    standard deviation 33876748.46845418
    


```python
#performing cross validation on our random forest regression model

forest_scores=cross_val_score(forest_reg, train_prepared,train_labels,scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores=np.sqrt(-forest_scores)
print('scores:',forest_rmse_scores)
print('mean',forest_rmse_scores.mean())
print('standard deviation',forest_rmse_scores.std())
```

    scores: [47224.92182302 51681.36468686 49864.00113439 51799.27588945
     52449.7032766  47024.82114377 47452.65120166 50222.29040321
     49139.73669868 49997.55548551]
    mean 49685.63217431541
    standard deviation 1868.872853756544
    

# Model Tuning


```python
#fine tuning a model
#using the grid search
from sklearn.model_selection import GridSearchCV

param_grid=[ {'alpha': [0.01,0.04,0.07,0.1],'max_iter':[200,400,600,800,1000]}]

forest_reg=SGDRegressor()

grid_search=GridSearchCV(sgd_mdl, param_grid,cv=5,scoring='neg_mean_squared_error',return_train_score=True)

grid_search.fit(train_prepared, train_labels)
```




    GridSearchCV(cv=5, estimator=SGDRegressor(),
                 param_grid=[{'alpha': [0.01, 0.04, 0.07, 0.1],
                              'max_iter': [200, 400, 600, 800, 1000]}],
                 return_train_score=True, scoring='neg_mean_squared_error')




```python
estimator=SGDRegressor()
estimator.get_params().keys()

```




    dict_keys(['alpha', 'average', 'early_stopping', 'epsilon', 'eta0', 'fit_intercept', 'l1_ratio', 'learning_rate', 'loss', 'max_iter', 'n_iter_no_change', 'penalty', 'power_t', 'random_state', 'shuffle', 'tol', 'validation_fraction', 'verbose', 'warm_start'])




```python
grid_search.best_params_
```




    {'alpha': 0.1, 'max_iter': 1000}




```python
grid_search.best_estimator_
```




    SGDRegressor(alpha=0.1)




```python
cvres=grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
```

    12087620.272264155 {'alpha': 0.01, 'max_iter': 200}
    27717875.018455766 {'alpha': 0.01, 'max_iter': 400}
    16815044.839521367 {'alpha': 0.01, 'max_iter': 600}
    27863584.290531427 {'alpha': 0.01, 'max_iter': 800}
    15846740.401995681 {'alpha': 0.01, 'max_iter': 1000}
    6780231.98355839 {'alpha': 0.04, 'max_iter': 200}
    3443247.2923271577 {'alpha': 0.04, 'max_iter': 400}
    4080711.9834982534 {'alpha': 0.04, 'max_iter': 600}
    3666168.99341223 {'alpha': 0.04, 'max_iter': 800}
    4499915.191164249 {'alpha': 0.04, 'max_iter': 1000}
    694984.6728840792 {'alpha': 0.07, 'max_iter': 200}
    329409.67288712336 {'alpha': 0.07, 'max_iter': 400}
    100292.00258391847 {'alpha': 0.07, 'max_iter': 600}
    523322.0367335065 {'alpha': 0.07, 'max_iter': 800}
    1165871.4404877583 {'alpha': 0.07, 'max_iter': 1000}
    271226.18737003475 {'alpha': 0.1, 'max_iter': 200}
    74834.36132849634 {'alpha': 0.1, 'max_iter': 400}
    128454.19206022762 {'alpha': 0.1, 'max_iter': 600}
    121169.50014579562 {'alpha': 0.1, 'max_iter': 800}
    71469.48608381976 {'alpha': 0.1, 'max_iter': 1000}
    


```python
#saving our model in a file
import pickle
filename='sgd_housing_model.pkl'
pickle.dump(grid_search.best_estimator_,open(filename,'wb'))

```

# Model Testing

### We will first clean our test dataset before we do the testing

## Cleaning Test dataset


```python
#checking for null values in the test dataset
test.isnull().sum()
```




    longitude               0
    latitude                0
    housing_median_age      0
    total_rooms             0
    total_bedrooms        207
    population              0
    households              0
    median_income           0
    median_house_value      0
    ocean_proximity         0
    dtype: int64




```python
#handling numerical data
'''replacing the attributes missing value with the median,
we use median instead of mean because some of our data has outliers'''
#importing a SimpleImputer class to replace missing values using impute method
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy='median')
#remove the ocean proximity column since the imputer method works with numerical data
test_df=test.drop('ocean_proximity',axis=1)
#fitting the imputer instance in the test data using fit() method
imputer.fit(test_df)
```




    SimpleImputer(strategy='median')




```python
#checking all the median values of our dataset
test_df.median().values
```




    array([-1.1847e+02,  3.4230e+01,  2.9000e+01,  2.1100e+03,  4.2800e+02,
            1.1600e+03,  4.0600e+02,  3.5000e+00,  1.7865e+05])




```python
#using the imputer to transform the test data with the learnt median
x=imputer.transform(test_df)
#to convert the array result to a dataframe
test_trained=pd.DataFrame(x,columns=test_df.columns)
test_trained.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-119.01</td>
      <td>36.06</td>
      <td>25.0</td>
      <td>1505.0</td>
      <td>428.0</td>
      <td>1392.0</td>
      <td>359.0</td>
      <td>1.6812</td>
      <td>47700.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-119.46</td>
      <td>35.14</td>
      <td>30.0</td>
      <td>2943.0</td>
      <td>428.0</td>
      <td>1565.0</td>
      <td>584.0</td>
      <td>2.5313</td>
      <td>45800.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-122.44</td>
      <td>37.80</td>
      <td>52.0</td>
      <td>3830.0</td>
      <td>428.0</td>
      <td>1310.0</td>
      <td>963.0</td>
      <td>3.4801</td>
      <td>500001.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-118.72</td>
      <td>34.28</td>
      <td>17.0</td>
      <td>3051.0</td>
      <td>428.0</td>
      <td>1705.0</td>
      <td>495.0</td>
      <td>5.7376</td>
      <td>218600.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-121.93</td>
      <td>36.62</td>
      <td>34.0</td>
      <td>2351.0</td>
      <td>428.0</td>
      <td>1063.0</td>
      <td>428.0</td>
      <td>3.7250</td>
      <td>278000.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#handling categorical data
test_ctg=test[['ocean_proximity']]
test_ctg.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20046</th>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>3024</th>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>15663</th>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>20484</th>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>9814</th>
      <td>NEAR OCEAN</td>
    </tr>
  </tbody>
</table>
</div>




```python
#use OneHotEncoder to convert the categorical data to one_hot vector
from sklearn.preprocessing import OneHotEncoder
ctg_encoder=OneHotEncoder()
#fitting and transforming the ctg_encoder instance to the test categorical data
test_ctg_1hot=ctg_encoder.fit_transform(test_ctg)
test_ctg_1hot
```




    <4128x5 sparse matrix of type '<class 'numpy.float64'>'
    	with 4128 stored elements in Compressed Sparse Row format>




```python
#converting the spicy sparse matrix into a numpy array using toarray() method
test_ctg_1hot.toarray()
```




    array([[0., 1., 0., 0., 0.],
           [0., 1., 0., 0., 0.],
           [0., 0., 0., 1., 0.],
           ...,
           [1., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0.],
           [1., 0., 0., 0., 0.]])




```python
#list of the encoder categories
ctg_encoder.categories_
```




    [array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
           dtype=object)]



### Feature Engineering


```python
test['rooms_per_household'] =test['total_rooms']/test['households']
test['bedrooms_per_rooms']=test['total_bedrooms']/test['total_rooms']
test['population_per_household']=test['population']/test['households']
```


```python
#checking the correlation of the new created columns to the median house value
corr_matrix= test.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)
```




    median_house_value          1.000000
    median_income               0.677502
    total_rooms                 0.134697
    rooms_per_household         0.130928
    housing_median_age          0.113585
    households                  0.074249
    total_bedrooms              0.056667
    population                 -0.019003
    longitude                  -0.044062
    population_per_household   -0.121853
    latitude                   -0.149295
    bedrooms_per_rooms         -0.249196
    Name: median_house_value, dtype: float64



### Feature Scaling


```python
#spliting data into input and output(i.e features and labels )
#creating the output data.Making a copy so that any changes made in this o not affect the original dataset
test_labels=test['median_house_value'].copy()
#creating the input data
test=test.drop('median_house_value',axis=1)
#droping the categorical value column so as to work with the numerical values including the added columns
test_df=test.drop('ocean_proximity',axis=1)
test_labels.head()
```




    20046     47700.0
    3024      45800.0
    15663    500001.0
    20484    218600.0
    9814     278000.0
    Name: median_house_value, dtype: float64




```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline=Pipeline([('imputer',SimpleImputer(strategy='median')),('std_scaler',StandardScaler())])
test_df_tr=num_pipeline.fit_transform(test_df)
```


```python
from sklearn.compose import ColumnTransformer
num_attributes=list(test_df)
ctg_attributes=['ocean_proximity']

full_pipeline=ColumnTransformer([('num',num_pipeline,num_attributes),('cat',OneHotEncoder(),ctg_attributes)])
test_prepared =full_pipeline.fit_transform(test)
test_prepared
```




    array([[ 0.25541734,  0.22194113, -0.30073951, ...,  0.        ,
             0.        ,  0.        ],
           [ 0.02976613, -0.20947715,  0.098724  , ...,  0.        ,
             0.        ,  0.        ],
           [-1.46454628,  1.03788441,  1.85636346, ...,  0.        ,
             1.        ,  0.        ],
           ...,
           [-1.2689819 ,  0.80810728, -0.30073951, ...,  0.        ,
             0.        ,  0.        ],
           [-0.120668  ,  0.5548835 ,  0.57808022, ...,  0.        ,
             0.        ,  0.        ],
           [ 0.57634349, -0.64089543, -0.93988113, ...,  0.        ,
             0.        ,  0.        ]])




```python
test_prepared.shape
```




    (4128, 16)



## Loading the model


```python
#re_loading the model
import pickle
model =pickle.load(open('sgd_housing_model.pkl','rb'))
model
```




    SGDRegressor(alpha=0.1)



## Predicting Values


```python
#making predictions using the test dataset
predictions=model.predict(test_prepared)
predictions
```




    array([ 62046.29639417, 127685.36148466, 261803.87352789, ...,
           422812.78033295, 130714.53429998, 193015.15467117])



## Evaluating Model


```python
from sklearn.metrics import mean_squared_error
predictions=model.predict(test_prepared)
test_mse=mean_squared_error(test_labels,predictions)
test_rmse=np.sqrt(test_mse)
test_rmse
```




    70470.27116567994



## Comparing Models and drawing conclusion

From the results we had during the training of the data, the  Random Forest Model had a lower root mean squared error than the  Stochastic Gradient Model. While on the testing data, the  Random Forest Model still had a lower root mean squared error than the  Stochastic Gradient Model.  

But then, when you compare the results of both models during training and testing, you will realize that the training error of both models were lower than the generalization error which means that the models are overfitting the training data. 

The Random Forest Model performed best compared to the Stochastic Gradient Model. 
