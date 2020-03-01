#Cab Analysis of Boston 


Objective:
•	Compare Lyft and Uber
•	Predict Price of cab
•	Predict Surge multiplier

Datasets: 
Data Source: https://www.kaggle.com/ravi72munde/uber-lyft-cab-prices#weather.csv

Two datasets are used:
•	Cab rides - This dataset has all the records of the trips taken by Uber and Lyft in November and December 2018. Shape - (693071 X 10)
•	Weather Data – This file has hourly weather data of Boston for 2018. Shape (6276 X 8)
Libraries Used:
•	Numpy
•	Pandas
•	sklearn.preprocessing – StandardScaler, LabelEncoder
•	sklearn.metrics - confusion_matrix, accuracy_score
•	matplotlib.pyplot
•	sklearn.model_selection - train_test_split
•	sklearn.linear_model - LogisticRegression
•	sklearn.ensemble – RandomForestRegressor, RandomForestClassifier

Data loading and Processing steps:
1.	.csv files for weather and cab data are loaded in the notebook
2.	Data Processing – Cab dataset
a.	Replace null price values by the mean of the column
b.	Add new column – category- type of cab service(shared, regular, premium, black premium, black premium SUV and others)
c.	Day and Hour columns are also extracted from the timestamp for analysis
3.	Data Processing – Weather data
a.	Replace all the null values in rain column by 0. This means there was no rain that day
4.	New columns, date_time is added to both the datasets to create a primary key to join the two datasets
5.	Uber v/s Lyft comparison
a.	Cab Dataset is grouped into 2 dataframes, one for Lyft and the other for Uber
b.	1st Comparison: Price. Both the companies are compared on the average cost of each trip.
 
c.	2nd Comparison: Compared on the average price of trip of each category
 
d.	3rd Comparison: Comparison of days and hours, when most of the cabs are booked for each organization
 
6.	Analyse surge multiplier. At which hour is the multiplier added.
 
7.	Merge the two datasets using join on primary key date_time
8.	 The merged dataframe is processed
a.	All the null values are dropped from the dataframe
9.	Model 1: Predicting Price based on day, distance, hour, temp, clouds, pressure, humidity, wind and rain using Random Forest Regressor
a.	Filter the data to reduce the size of the dataframe. For this model we will only be taking Share Ubers data.
b.	Spilt data into test training the ration of 1:4.
c.	Use random regressor classier to predict price
d.	Calculate accuracy
 
e.	Compute feature importance:
 
10.	Model 2 - Predict Surge Multiplier based on day, distance, hour, temp, clouds, pressure, humidity, wind and rain using Logistic Regression Classifier and Random Forest Classifier
a.	For this model we will use dataframe for all the Lyft regular cabs
b.	Label encode the surge multiplier
c.	Spilt the data into test and train in ratio 1:4
d.	Predict surge multiplier using Logistic Regression
e.	Compute Accuracy
f.	Predict surge multiplier using Random forest
g.	Compute accuracy
h.	Compare accuracy – Both the models have same accuracy
 
i.	Compute Feature Importance
 



