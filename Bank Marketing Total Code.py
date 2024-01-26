#!/usr/bin/env python
# coding: utf-8

# In[6]:


#!pip install pygwalker
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
#import pygwalker as pyg
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from collections import Counter


# # Code for Analysis (Including internal analysis graphs)

# In[120]:


data = pd.read_csv(r'C:\PDS Project/Analysis.csv', sep=';')


# In[3]:


data.head()


# In[4]:


data.shape


# In[37]:


bins = list(range(15, 100, 5))  # Starting from 15, ending at 100, with a step of 5

# Create a new column 'age_group' with the corresponding bin labels
data['age_group'] = pd.cut(data['age'], bins=bins, right=False, include_lowest=True)

# Count the number of rows in each age group
age_group_counts = data['age_group'].value_counts().sort_index()

# Plot the bar graph
plt.figure(figsize=(9, 5))
age_group_counts.plot(kind='bar', width=0.8)

plt.xlabel('Age Group')
plt.ylabel('Number of Calls')
plt.title('Number of Calls in Each Age Group')
plt.xticks(rotation=30, ha='right')

plt.show()


# In[31]:


data_age = data[data['y'] == 'yes']


# In[32]:


data_age.shape


# In[38]:


data_age = data[data['y'] == 'yes']
bins = list(range(15, 100, 5))  # Starting from 15, ending at 100, with a step of 5

# Create a new column 'age_group' with the corresponding bin labels
data_age['age_group'] = pd.cut(data_age['age'], bins=bins, right=False, include_lowest=True)

# Count the number of rows in each age group
age_group_counts = data_age['age_group'].value_counts().sort_index()

# Plot the bar graph
plt.figure(figsize=(9, 5))
age_group_counts.plot(kind='bar', width=0.8, color='g')

plt.xlabel('Age Group')
plt.ylabel('Number of Conversions')
plt.title('Conversion of Calls in Each Age Group')
plt.xticks(rotation=30, ha='right')

plt.show()


# In[39]:


job_graph = data['job'].value_counts().to_frame().reset_index()

plt.figure(figsize=(9, 5))
#plt.bar(count_ones_zeros['Activity_Tile'], count_ones_zeros['Count_Ones'], label='Count of Ones', alpha=0.7)
plt.bar(job_graph['job'], job_graph['count'], label='Number of calls done to different job holders')
plt.xlabel('Job type')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Number of Calls')
plt.title('Number of calls done to different job holders')
plt.legend()
plt.show()


# In[40]:


dj = data[data['y'] == 'yes']
job_graph_yes = dj['job'].value_counts().to_frame().reset_index()
plt.figure(figsize=(9, 5))

#plt.bar(count_ones_zeros['Activity_Tile'], count_ones_zeros['Count_Ones'], label='Count of Ones', alpha=0.7)
plt.bar(job_graph_yes['job'], job_graph['count'], label='Number of conversions from different job holders', color = 'g')
plt.xlabel('Job type')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Number of Conversions')
plt.title('Number of conversions from different job holders')
plt.legend()
plt.show()


# In[42]:


# Plot the bar graph with hue encoding
plt.figure(figsize=(9, 5))
sns.countplot(x='job', hue='y', data=data, palette='Set2')

plt.xlabel('Job Category')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Count')
plt.title('Count of Customers in Each Job Category with Hue Encoding for Joining')

plt.show()


# In[51]:


# Calculate the percentage of 'Yes' and 'No' for each job category
percentage_df = data.groupby(['job', 'y']).size().unstack().div(data.groupby('job').size(), axis=0) * 100

# Plot the stacked bar graph
plt.figure(figsize=(10, 6))
colors = ['#001F3F', 'g']
percentage_df.plot(kind='bar', stacked=True, color = colors, edgecolor='black')

plt.xlabel('Job Category')
plt.ylabel('Percentage')
plt.title('Percentage of Conversion in Each Job Category')

plt.legend(title='Joining (y)', loc='upper right', bbox_to_anchor=(1.2, 1))

plt.show()


# In[52]:


data.columns


# In[53]:


count_df = data.groupby(['education', 'y']).size().unstack().fillna(0)

# Plot the grouped bar graph
plt.figure(figsize=(9, 5))
count_df.plot(kind='bar', stacked=True, color=['#3498db', '#2ecc71'])

plt.xlabel('Education Category')
plt.ylabel('Count')
plt.title('Count of Rows and "Yes" in Each Education Category')

plt.legend(title='Joining (y)', loc='upper right', bbox_to_anchor=(1.25, 1))

plt.show()


# In[56]:


count_df = data.groupby(['education', 'y']).size().unstack().fillna(0)

# Plot the grouped bar graph side by side
plt.figure(figsize=(9, 5))
count_df.plot(kind='bar', width=0.8)

plt.xlabel('Education Category')
plt.ylabel('Count')
plt.title('Count of Calls and Subscribers in Each Education Category')

plt.legend(title='Joining (y)', loc='upper right', bbox_to_anchor=(1.25, 1))

plt.show()


# In[115]:


# Calculate the count of 'Yes' and 'No'
count_df = data['y'].value_counts()

# Create a 3D bar chart
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Data for bar chart
jobs = data['job']
categories = count_df.index
values = count_df.values
bottom = [0, 0]
width = 0.8

# Plot 3D bars
ax.bar(categories, values, bottom=bottom, width=width, color=['#3498db', '#2ecc71'], alpha=0.8)

ax.set_xlabel('Category')
ax.set_ylabel('Count')
ax.set_zlabel('job')

ax.set_title('3D Bar Chart for "Yes" and "No" Counts')

plt.show()


# In[63]:


# Calculate the count of 'Yes' and 'No'
count_df = data['y'].value_counts()

# Create a pie chart with two separate slices
explode = [0.1, 0]  # Separating the first slice
colors = ['#3498db', '#2ecc71']
labels = count_df.index
sizes = count_df.values

plt.figure(figsize=(5, 5))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.2f%%', startangle=90, shadow=True)
plt.title('Pie Chart for "Yes" and "No" Counts')

plt.show()


# In[64]:


data.columns


# In[ ]:





# In[125]:


# Set color palette for PowerPoint-friendly colors
ppt_colors = ['#1f78b4', '#33a02c']

# Create subplots for 'default', 'housing', and 'loan'
fig, axes = plt.subplots(3, 1, figsize=(6, 15))

# Function to plot percentage bars with 'yes' and 'no' side by side
def plot_percentage_bars(category, ax):
    percentage_data = data.groupby(category)['y'].value_counts(normalize=True).unstack() * 100
    percentage_data.plot(kind='bar', stacked=True, color=ppt_colors, ax=ax)
    ax.set_title(f'{category.capitalize()} vs. y')
    ax.set_ylabel('Percentage')
    ax.legend(title='y', bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot for 'default'
plot_percentage_bars('default', axes[0])

# Plot for 'housing'
plot_percentage_bars('housing', axes[1])

# Plot for 'loan'
plot_percentage_bars('loan', axes[2])

plt.tight_layout()
plt.show()


# In[75]:


# Create subplots for 'default', 'housing', and 'loan'
fig, axes = plt.subplots(3, 1, figsize=(6, 10))

# Set color palette for PowerPoint-friendly colors
ppt_colors = ['#1f78b4', '#33a02c']

# Bar plot for 'default' with 'y' as hue
sns.countplot(x='default', hue='y', data=data, palette=ppt_colors, ax=axes[0])
axes[0].set_title('Default vs. y')

# Bar plot for 'housing' with 'y' as hue
sns.countplot(x='housing', hue='y', data=data, palette=ppt_colors, ax=axes[1])
axes[1].set_title('Housing vs. y')

# Bar plot for 'loan' with 'y' as hue
sns.countplot(x='loan', hue='y', data=data, palette=ppt_colors, ax=axes[2])
axes[2].set_title('Loan vs. y')

plt.tight_layout()
plt.show()


# In[78]:


data.columns


# In[79]:


data['balance'].value_counts()


# In[80]:


# Create a box plot
plt.figure(figsize=(8, 6))
sns.boxplot(x='y', y='balance', data=data, palette=['#1f78b4', '#33a02c'])
plt.title('Box Plot of Bank Balance for Different Values of Y')
plt.show()


# In[82]:


# Create a line plot
plt.figure(figsize=(8, 6))
sns.lineplot(x=data.index, y='balance', hue='y', data=data, marker='o', palette=['#1f78b4', '#33a02c'])
plt.title('Line Plot of Bank Balances for Yes and No')
plt.xlabel('Individuals')
plt.ylabel('Bank Balance')
plt.show()


# In[84]:


# Create a line plot
plt.figure(figsize=(8, 6))
sns.lineplot(x=data.index, y='balance', data=data, marker='o', color='blue')
plt.title('Line Plot of Bank Balances')
plt.xlabel('Individuals')
plt.ylabel('Bank Balance')
plt.show()


# In[96]:


data['y'].value_counts()


# In[97]:


# Filter the DataFrame to include only records where 'Y' is 'Yes'
df_yes = data[data['y'] == 'yes']

# Plotting the line graph
plt.plot(df_yes['balance'], marker='o', linestyle='-', color='b', label='Y = Yes')

# Adding labels and title
plt.xlabel('Record Index')
plt.ylabel('Bank Balance')
plt.title('Bank Balance for Records with Y = Yes')
plt.legend()

# Show the plot
plt.show()


# In[101]:


# Convert 'y' values to numeric for plotting
data['y_numeric'] = pd.Categorical(data['y']).codes

# Create a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='balance', y='y_numeric', data=data, alpha=0.5)
plt.yticks([0, 1], ['No', 'Yes'])  # Labeling y-axis ticks
plt.xlabel('Balance')
plt.title('Scatter Plot of Balance with Respect to Y')
plt.show()


# In[102]:


plt.figure(figsize=(10, 6))
sns.violinplot(x='y', y='balance', data=data)
plt.xlabel('y')
plt.ylabel('balance')
plt.title('Violin Plot of Balance with Respect to Y')
plt.show()


# In[113]:


df = data[data['balance'] >= 0]
# Create a heatmap
plt.figure(figsize=(8, 4))
heatmap_data = df.groupby(['y', pd.cut(df['balance'], bins=15)]).size().unstack().fillna(0)
sns.heatmap(heatmap_data, annot=True, fmt='g', cmap='viridis')
plt.xlabel('Balance Bins')
plt.ylabel('Y')
plt.title('Heatmap of Balance with Respect to Y')
plt.show()


# # Code for dashboard
# 

# In[3]:


gwalker = pyg.walk(data)


# # Code of Modelling (Logistic Regression)

# In[7]:


data = pd.read_csv(r'C:\PDS Project/Analysis.csv', sep=';')


# In[8]:


# load X and y
X = data.drop(columns=['y'])
y = data['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)


# In[9]:


onehot = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact','month','day','poutcome']


# In[10]:


def dummy_df(df, todummy_list):
    for x in todummy_list:
        dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False) # prefix give name
        df = df.drop(x, axis = 1)
        df = pd.concat([df, dummies], axis=1)
    return df


# In[11]:


X_train_one = dummy_df(X_train, onehot)
print(X_train_one.shape)


# In[12]:


X_test_one = dummy_df(X_test, onehot)
print(X_test_one.shape)


# In[13]:


#Create an Logistic classifier and train it on 70% of the data set.
from sklearn import svm

clf = LogisticRegression()
clf


# In[14]:


print(X.shape)
print(X_test.shape)
print(y.shape)
print(y_test.shape)


# In[15]:


clf.fit(X_train_one, y_train)


# In[16]:


#Prediction using test data
y_pred = clf.predict(X_test_one)


# In[17]:


from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))


# In[18]:


predictions = clf.predict(X_test_one)


# In[19]:


predictions = clf.predict(X_test_one)
# Imports
from sklearn.metrics import confusion_matrix, classification_report

# Confusion matrix
cm=confusion_matrix(y_test, predictions)
print(cm)

# New line
print('\n')

# Classification report
print(classification_report(y_test,predictions))


# In[20]:


# plot confusion matrix to describe the performance of classifier.
import seaborn as sns
import matplotlib.pyplot as plt
class_label = ["No", "Yes"]
df_cm = pd.DataFrame(cm, index = class_label, columns = class_label)
sns.heatmap(df_cm, annot = True, fmt = "d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()


# # Over Sampling

# In[21]:


from imblearn.over_sampling import SMOTE

# summarize class distribution
counter = Counter(y_train)
print(counter)


# In[22]:


# transform the dataset
oversample = SMOTE()
X_train_smote, y_train = oversample.fit_resample(X_train_one, y_train)


# In[23]:


# summarize the new class distribution
counter = Counter(y_train)
print(counter)


# In[24]:


#Create an Logistic classifier and train it on 70% of the data set.
from sklearn import svm

clf = LogisticRegression()
clf


# In[25]:


clf.fit(X_train_smote, y_train)


# In[26]:


#Prediction using test data
y_pred = clf.predict(X_test_one)


# In[27]:


from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))


# In[28]:


predictions = clf.predict(X_test_one)
# Imports
from sklearn.metrics import confusion_matrix, classification_report

# Confusion matrix
cm=confusion_matrix(y_test, predictions)
print(cm)

# New line
print('\n')

# Classification report
print(classification_report(y_test,predictions))


# In[29]:


# plot confusion matrix to describe the performance of classifier.
import seaborn as sns
import matplotlib.pyplot as plt
class_label = ["No", "Yes"]
df_cm = pd.DataFrame(cm, index = class_label, columns = class_label)
sns.heatmap(df_cm, annot = True, fmt = "d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()


# In[ ]:




