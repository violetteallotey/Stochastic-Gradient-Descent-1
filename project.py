#!/usr/bin/env python
# coding: utf-8

# # Data Preparation

# ### Read Data




# importing libraries and reading the housing data

import pandas as pd

import numpy as np

housing_data=pd.read_csv('housing.csv')

housing_data.head()


# ### Spliting the data




#spliting the dataset into train and test sets
#import the train_test_split() method from sklearn.model_selection
from sklearn.model_selection import train_test_split
train,test=train_test_split(housing_data,test_size=0.2,random_state=42)
train.shape,test.shape


# # Cleaning the data




# cleaning the training set by handling missing values

train.isnull().sum()




#using imputer method to replace missing numerical values with median
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy='median')

#dropping the categorical value column
train_num=train.drop('ocean_proximity',axis=1)

#fitting the imputer to the numerical training dataset
imputer.fit(train_num)





#applying the strategy to our numerical training data using the transform()
x=imputer.transform(train_num)





#the results is an array to convert it to a dataframe apply pd
train_prepared=pd.DataFrame(x,columns=train_num.columns)
train_prepared.head()





#handling categorical data

train_ctg=train[['ocean_proximity']]
train_ctg.head()





# import onehotencoder from sklearn.preprocessing,to convert the categoriccal data to one hot vector
from sklearn.preprocessing import OneHotEncoder
ctg_encoder=OneHotEncoder()
train_ctg_1hot=ctg_encoder.fit_transform(train_ctg)
train_ctg_1hot





#convert the resulting sparse matrix to a numpy array
train_ctg_1hot.toarray()


# ### Feature Engineering



train['rooms_per_household'] =train['total_rooms']/train['households']
train['bedrooms_per_rooms']=train['total_bedrooms']/train['total_rooms']
train['population_per_household']=train['population']/train['households']




#checking the correlation of the new created columns to the median house value
corr_matrix= train.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)


# ### Feature Scaling




#spliting data into input and output(i.e features and labels )
#creating the output data.Making a copy so that any changes made in this o not affect the original dataset
train_labels=train['median_house_value'].copy()

#creating the input data
train=train.drop('median_house_value',axis=1)

#droping the categorical value column so as to work with the numerical values including the added columns
train_num=train.drop('ocean_proximity',axis=1)
train_labels.head()





# since there are many  data transformation steps that need to be executed 
# in the right order, the Pipeline class in sklearn will help with such sequence of transformation

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline=Pipeline([('imputer',SimpleImputer(strategy='median')),('std_scaler',StandardScaler())])
train_num_tr=num_pipeline.fit_transform(train_num)





'''scikit has introduced a new class called the ColumnTransfomer 
that transforms both the numerical and categorical values at the same time'''
from sklearn.compose import ColumnTransformer
num_attributes=list(train_num)
ctg_attributes=['ocean_proximity']

full_pipeline=ColumnTransformer([('num',num_pipeline,num_attributes),('ctg',OneHotEncoder(),ctg_attributes)])
train_prepared=full_pipeline.fit_transform(train)
train_prepared





train_prepared.shape


# # Model Training

# ### Random Forest Model




#Random forest regression model

from sklearn.ensemble import RandomForestRegressor
forest_reg=RandomForestRegressor()
forest_reg.fit(train_prepared, train_labels)





#Measuring the Root Mean Squared Error of this model

from sklearn.metrics import mean_squared_error
train_predictions=forest_reg.predict(train_prepared)
forest_mse=mean_squared_error(train_labels, train_predictions)
forest_rmse=np.sqrt(forest_mse)
forest_rmse


# ### Stochastic Gradient Descent Model




from sklearn.linear_model import SGDRegressor
sgd_mdl=SGDRegressor()
sgd_mdl.fit(train_prepared,train_labels)





#use some of the trainig data to test our model
some_data=train.iloc[:5]
some_data_labels=train_labels[:5]

#perform a full pipeline and transform some_data
some_data_prepared=full_pipeline.transform(some_data)

#print the predictions of our model
print('predictions:',sgd_mdl.predict(some_data_prepared))





#to measure rmse of this sgd model
train_predictions=sgd_mdl.predict(train_prepared)
sgd_mse=mean_squared_error(train_labels,train_predictions)
sgd_rmse= np.sqrt(sgd_mse)
sgd_rmse





#performing cross validation on our SGD model

from sklearn.model_selection import cross_val_score
sgd_scores=cross_val_score(sgd_mdl, train_prepared,train_labels,scoring="neg_mean_squared_error", cv=10)
sgd_rmse_scores=np.sqrt(-sgd_scores)
print('scores:',sgd_rmse_scores)
print('mean',sgd_rmse_scores.mean())
print('standard deviation',sgd_rmse_scores.std())





#performing cross validation on our random forest regression model

forest_scores=cross_val_score(forest_reg, train_prepared,train_labels,scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores=np.sqrt(-forest_scores)
print('scores:',forest_rmse_scores)
print('mean',forest_rmse_scores.mean())
print('standard deviation',forest_rmse_scores.std())


# # Model Tuning




#fine tuning a model
#using the grid search
from sklearn.model_selection import GridSearchCV

param_grid=[ {'alpha': [0.01,0.04,0.07,0.1],'max_iter':[200,400,600,800,1000]}]

forest_reg=SGDRegressor()

grid_search=GridSearchCV(sgd_mdl, param_grid,cv=5,scoring='neg_mean_squared_error',return_train_score=True)

grid_search.fit(train_prepared, train_labels)





estimator=SGDRegressor()
estimator.get_params().keys()





grid_search.best_params_





grid_search.best_estimator_





cvres=grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)





#saving our model in a file
import pickle
filename='sgd_housing_model.pkl'
pickle.dump(grid_search.best_estimator_,open(filename,'wb'))


# # Model Testing

# ### We will first clean our test dataset before we do the testing

# ## Cleaning Test dataset




#checking for null values in the test dataset
test.isnull().sum()





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




#checking all the median values of our dataset
test_df.median().values





#using the imputer to transform the test data with the learnt median
x=imputer.transform(test_df)
#to convert the array result to a dataframe
test_trained=pd.DataFrame(x,columns=test_df.columns)
test_trained.head()





#handling categorical data
test_ctg=test[['ocean_proximity']]
test_ctg.head()





#use OneHotEncoder to convert the categorical data to one_hot vector
from sklearn.preprocessing import OneHotEncoder
ctg_encoder=OneHotEncoder()
#fitting and transforming the ctg_encoder instance to the test categorical data
test_ctg_1hot=ctg_encoder.fit_transform(test_ctg)
test_ctg_1hot




#converting the spicy sparse matrix into a numpy array using toarray() method
test_ctg_1hot.toarray()




#list of the encoder categories
ctg_encoder.categories_


# ### Feature Engineering



test['rooms_per_household'] =test['total_rooms']/test['households']
test['bedrooms_per_rooms']=test['total_bedrooms']/test['total_rooms']
test['population_per_household']=test['population']/test['households']





#checking the correlation of the new created columns to the median house value
corr_matrix= test.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)


# ### Feature Scaling



#spliting data into input and output(i.e features and labels )
#creating the output data.Making a copy so that any changes made in this o not affect the original dataset
test_labels=test['median_house_value'].copy()
#creating the input data
test=test.drop('median_house_value',axis=1)
#droping the categorical value column so as to work with the numerical values including the added columns
test_df=test.drop('ocean_proximity',axis=1)
test_labels.head()





from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline=Pipeline([('imputer',SimpleImputer(strategy='median')),('std_scaler',StandardScaler())])
test_df_tr=num_pipeline.fit_transform(test_df)





from sklearn.compose import ColumnTransformer
num_attributes=list(test_df)
ctg_attributes=['ocean_proximity']

full_pipeline=ColumnTransformer([('num',num_pipeline,num_attributes),('cat',OneHotEncoder(),ctg_attributes)])
test_prepared =full_pipeline.fit_transform(test)
test_prepared




test_prepared.shape


# ## Loading the model




#re_loading the model
import pickle
model =pickle.load(open('sgd_housing_model.pkl','rb'))
model


# ## Predicting Values




#making predictions using the test dataset
predictions=model.predict(test_prepared)
predictions


# ## Evaluating Model


from sklearn.metrics import mean_squared_error
predictions=model.predict(test_prepared)
test_mse=mean_squared_error(test_labels,predictions)
test_rmse=np.sqrt(test_mse)
test_rmse



