#!/usr/bin/env python
# coding: utf-8

# #### THE PROJECT IS PREDICT  BIKE RENTAL DEMAND. #####

# In[146]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[147]:


# Load the training data
train_data = pd.read_csv('C:/Users/Dell/OneDrive/Desktop/Vicky/Top Mentor Class Lectures/train.csv')
train_data


# In[148]:


print(train_data.info())


# In[149]:


# Display basic statistics of numerical columns
print(train_data.describe())


# In[150]:


# Explore numerical features
sns.histplot(train_data['count'], bins=30, kde=True)
plt.title('Distribution of Bike Rental Counts')
plt.show()


# In[151]:


# Scatter plot of temperature (temp) vs. bike rentals (count)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='temp', y='count', data=train_data)
plt.title('Temperature vs. Bike Rentals')
plt.xlabel('Temperature (Celsius)')
plt.ylabel('Count of Bike Rentals')
plt.show()


# In[152]:


# Visualize the correlation between numerical features
correlation_matrix = train_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# ##### Prepare Data for Training

# In[153]:


# Select features and target variable
features = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed']
target_variable = 'count'
print("Selected Features:", features)
print("Target Variable:", target_variable)


# In[154]:


X = train_data[features]
y = train_data[target_variable]
print("Features:")
print(X.head())  # Display the first few rows of the features DataFrame

print("\nTarget Variable:")
print(y.head())  # Display the first few rows of the target variable Series


# ##### Train a Machine Learning Model

# In[155]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# In[156]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# In[157]:


from sklearn.neural_network import MLPClassifier


# In[158]:


clf = MLPClassifier(hidden_layer_sizes=(6,5), random_state=3, verbose=True)
clf


# In[159]:


clf.fit(X_train, y_train)


# In[160]:


ypred = clf.predict(X_test)


# In[161]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[162]:


accuracy_score(y_test,ypred)


# In[163]:


confusion_matrix(y_test,ypred)


# In[164]:


# Train a RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# In[165]:


# Make predictions on the test set
predictions = model.predict(X_test)
print(predictions)


# In[166]:


from sklearn.ensemble import RandomForestClassifier 
def random_forest_model(X_train,X_test,y_train,y_test):
    model1 = RandomForestClassifier(criterion='entropy')
    model1.fit(X_train, y_train)
    pred1= model1.predict(X_test)
    results = confusion_matrix(y_test,pred1)
    acc = accuracy_score(y_test,pred1)
    return acc


# In[167]:


accuracy = random_forest_model(X_train,X_test,y_train,y_test)
print("Accuracy of Random Forest is - ", accuracy)


# In[168]:


# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')


# In[169]:


# Load the test dataset
test_data = pd.read_csv('C:/Users/Dell/OneDrive/Desktop/Vicky/Top Mentor Class Lectures/test.csv')
test_data


# In[170]:


# Select the same features used for training
X_test_data = test_data[features]
print(X_test_data)


# In[171]:


# Make predictions on the test set
test_predictions = model.predict(X_test_data)
print(test_predictions)


# In[172]:


# Add predictions to the test dataset
test_data['predicted_count'] = test_predictions
print(test_data)


# In[173]:


# Display the predictions
print(test_data[['datetime', 'predicted_count']])


# In[174]:


# Calculate RMSLE
rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred))
print(f'RMSLE: {rmsle}')


# ############################################## END ##############################################

# In[ ]:




