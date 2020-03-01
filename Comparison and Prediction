#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


# # Installing Dataset

# In[2]:


df_cab=pd.read_csv('cab_rides.csv')
df_weather = pd.read_csv('weather.csv')


# # Data Processing

# In[3]:


df_cab.info()


# In[4]:


df_weather.info()


# In[5]:


df_cab.columns


# In[6]:


df_cab.head(5)


# In[5]:


df_weather.columns


# In[6]:


df_weather.head(5)


# In[7]:


df_cab.isnull().sum()


# In[8]:


df_cab["price"] = df_cab["price"].fillna(df_cab["price"].mean())


# In[9]:


df_weather.isnull().sum()


# In[10]:


df_weather.shape


# In[11]:


df_weather["rain"] = df_weather["rain"].fillna(0) 


# In[12]:


df_cab.isnull().sum()


# In[13]:


df_weather.isnull().sum()


# In[14]:


np.unique(df_cab["name"])
df_cab["category"] = ""
df_cab["category"][df_cab['name'].str.contains("Shared")] = "Shared"
df_cab["category"][df_cab['name'].str.contains("UberPool")] = "Shared"
df_cab["category"][df_cab['name'].str.contains("UberX")] = "Regular"
df_cab["category"][df_cab['name'].str.contains("Lyft")] = "Regular"
df_cab["category"][df_cab['name'].str.contains("Lux")] = "Premium"
df_cab["category"][df_cab['name'].str.contains('Lyft XL')] = "Premium"
df_cab["category"][df_cab['name'].str.contains("UberXL")] = "Premium"
df_cab["category"][df_cab['name'].str.contains("Lux Black")] = "Black_Premium"
df_cab["category"][df_cab['name'].str.contains("Black")] = "Black_Premium"
df_cab["category"][df_cab['name'].str.contains("Lux Black XL")] = "Black_Premium_SUV"
df_cab["category"][df_cab['name'].str.contains("Black SUV")] = "Black_Premium_SUV"
df_cab["category"][df_cab['name'].str.contains("WAV")] = "Others"
df_cab["category"][df_cab['name'].str.contains("Taxi")] = "Others"


# In[15]:


df_cab['date_time'] = pd.to_datetime(df_cab['time_stamp']/1000, unit='s')
df_weather['date_time'] = pd.to_datetime(df_weather['time_stamp'], unit='s')


# In[16]:


df_cab['day'] = df_cab.date_time.dt.dayofweek
df_cab['hour'] = df_cab.date_time.dt.hour


# # Uber v/s Lyft Compariosn

# In[17]:


grouped = df_cab.groupby("cab_type")
df_lyft = grouped.get_group("Lyft").reset_index()
df_lyft.shape


# In[18]:


df_uber = grouped.get_group("Uber").reset_index()
df_uber.shape


# In[58]:


df_uber["name"].value_counts()


# In[20]:


df_lyft["name"].value_counts()


# In[21]:


print("Average distance of a Uber trip Lyft in Boston" , round(np.mean(df_uber["distance"]),2), "miles.")


# In[22]:


print("Average distance of a Lyft trip Lyft in Boston" , round(np.mean(df_lyft["distance"]),2), "miles.")


# In[25]:


print("Average Uber Trip costs $", round(np.mean(df_uber["price"]),2))


# In[26]:


print("Average Lyft Trip costs $", round(np.mean(df_lyft["price"]),2))


# In[27]:


plt.figure(figsize=(20,10))

uber_price = df_uber["price"].value_counts().sort_index()
lyft_price = df_lyft["price"].value_counts().sort_index()
plt.bar(uber_price.index,uber_price.values, color = "blue", label = "Uber")
plt.bar(lyft_price.index,lyft_price.values, color = "red", label = "Lyft")
plt.ylabel('Frequency')
plt.xlabel('Trip Price')
plt.title("Price Comparison")
plt.legend()
plt.xlim((0,60))


# In[23]:


plt.figure(figsize=(10,5))
df_uber.groupby("category").mean()["price"].plot(kind = "line", label = "Uber")
df_lyft.groupby("category").mean()["price"].plot(kind = "line", label = "Lyft")
plt.legend()
plt.xlabel("Type of Cab")
plt.ylabel("Average Price")
plt.title("Price compariosn for each cab category")


# In[28]:


plt.figure(figsize=(10,5))
df_uber["hour"].value_counts().sort_index().plot(kind = "line", label = "Uber")
df_lyft["hour"].value_counts().sort_index().plot(kind = "line", label = "Lyft")
plt.legend()
plt.xlabel("Hour")
plt.ylabel("Number of Cabs")
plt.title("Price compariosn for each cab category")


# In[32]:


plt.figure(figsize=(10,5))
df_uber["day"].value_counts().sort_index().plot(kind = "line", label = "Uber")
df_lyft["day"].value_counts().sort_index().plot(kind = "line", label = "Lyft")
plt.legend()
plt.xlabel("Day")
plt.ylabel("Number of Cabs")
plt.title("Price compariosn for each cab category")


# In[19]:


filter_df = df_cab[df_cab["surge_multiplier"]>1]


# In[21]:


plt.figure(figsize=(10,5))
filter_df.groupby("hour").mean()["surge_multiplier"].sort_index().plot(kind = "line", label = "Surge_multiplier")
plt.legend()
plt.xlabel("Hour")
plt.ylabel("Average Surge multiplier")
plt.title("Average surge multiplier across the day")


# In[57]:


multip_cab = df_cab.groupby("source").mean()["surge_multiplier"].sort_values(ascending = False).head(10)


# In[63]:


plt.figure(figsize=(15,5))
df_cab.groupby("source").mean()["surge_multiplier"].sort_values(ascending = False).head(10).plot(kind = "line", label = "Surge_multiplier")
plt.legend()
plt.xlabel("Source")
plt.ylabel("Average Surge multiplier")
plt.title("Average multiplier for different sources")


# # Merge the datasets

# In[29]:


df_cab['merge_date'] = df_cab.source.astype(str) +" - "+ df_cab.date_time.dt.date.astype("str") +" - "+ df_cab.date_time.dt.hour.astype("str")
df_weather['merge_date'] = df_weather.location.astype(str) +" - "+ df_weather.date_time.dt.date.astype("str") +" - "+ df_weather.date_time.dt.hour.astype("str")
df_weather.index = df_weather['merge_date']
final_df = df_cab.join(df_weather,on=['merge_date'],rsuffix ='_w')


# In[30]:


final_df['day'] = final_df.date_time.dt.dayofweek
final_df['hour'] = final_df.date_time.dt.hour
final_df.isnull().sum()


# In[31]:


final_df = final_df.dropna()


# In[32]:


final_df.isnull().sum()


# # Model 1 - Predicting Price based on day,distance,hour,temp,clouds, pressure,humidity, wind and rain using Random Forest Regressor

# In[33]:


df2 = final_df[final_df["category"]== "Shared"]
df2 = df2[df2["cab_type"] == "Uber"]


# In[34]:


X = df2[['day','distance','hour','temp','clouds', 'pressure','humidity', 'wind', 'rain']]
Y = np.array(df2["price"].reset_index(drop = True))
X.reset_index(inplace=True)
X = X.drop(columns=['index'])
X = np.array(pd.get_dummies(X))


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42)

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(X_train, Y_train)
predictions = rf.predict(X_test)


# In[166]:


errors = abs(predictions - Y_test)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# In[167]:


accuracy = rf.score(X_train, Y_train)
print('Accuracy:', round((100 * accuracy),2), '%.')


# In[168]:


f_list = list(['day','distance','hour','temp','clouds', 'pressure','humidity', 'wind', 'rain'])
importances = list(rf.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(f_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


# In[169]:


indices = np.argsort(importances)
imp = rf.feature_importances_
plt.title('Feature Importances')
plt.barh(range(len(indices)), imp[indices], color='b', align='center')
plt.yticks(range(len(indices)), [f_list[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# ## 2nd Model = Predicting Surge Multiplier  based on day,distance,hour,temp,clouds, pressure,humidity, wind and rain using Logistic Regression Classifier

# In[41]:


final_df.columns 


# In[28]:


np.unique(final_df[final_df["category"] == "Regular"]["surge_multiplier"])


# In[43]:


final_df[final_df["category"] == "Regular"].shape


# In[37]:


df3 = final_df[final_df["name"] == "Lyft"]
df3 = df3[df3["category"] == "Regular"]


# In[38]:


df3.shape


# In[39]:


X = df3[['day','distance','hour','temp','clouds', 'pressure','humidity', 'wind', 'rain']]
X.reset_index(inplace=True)
X = X.drop(columns=['index'])
X = np.array(pd.get_dummies(X))


# In[40]:


Y = np.array(df3["surge_multiplier"].reset_index(drop = True))


# In[41]:


le = LabelEncoder()


# In[42]:


Y = Y = le.fit_transform(Y)


# In[43]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42)


# In[51]:


model = LogisticRegression()
model.fit(X_train, Y_train)
prediction = model.predict(X_test)


# In[55]:


accuracy = np. mean(prediction == Y_test )
print("Accuracy", round((100 * accuracy),2), '%.')


# In[56]:


model_rf = RandomForestClassifier(n_estimators =2, max_depth =4,criterion ='entropy', random_state = 100)
model_rf.fit(X_train, Y_train)
pred = model_rf.predict(X_test) 
accuracy = accuracy_score(Y_test, pred)


# In[57]:


print("Accuracy", round((100*accuracy),2))


# In[52]:


feature_importance = abs(model.coef_[0])
feature_importance = 100.0 * (feature_importance / feature_importance.max())


# In[53]:


f_list = list(['day','distance','hour','temp','clouds', 'pressure','humidity', 'wind', 'rain'])
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(f_list, feature_importance)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


# In[54]:


indices = np.argsort(feature_importance)
imp = feature_importance
plt.title('Feature Importances')
plt.barh(range(len(indices)), imp[indices], color='b', align='center')
plt.yticks(range(len(indices)), [f_list[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# In[ ]:




