#!/usr/bin/env python
# coding: utf-8

# In[12]:




import re
import numpy as np
import pandas as pd

import nltk
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
nltk.download('punkt')
nltk.download('wordnet')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import WordNetLemmatizer
# sklearn
from wordcloud import WordCloud
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB,MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# In[13]:


data=pd.read_csv("complaints.csv") 


# In[14]:


data=data.dropna(how='any')
data


# In[15]:


data_1 = data[data['product'] == 'credit_card']
data_2 = data[data['product'] == 'debt_collection']
data_3 = data[data['product'] == 'retail_banking']
data_4 = data[data['product'] == 'credit_reporting']
data_5 = data[data['product'] == 'mortgages_and_loans']


# In[16]:


dataset = pd.concat([data_1, data_2, data_3, data_4, data_5])


# In[17]:


import pandas as pd
from sklearn.utils import resample

upsampled_data = []
for product_class in dataset['product'].unique():
    class_data = dataset[dataset['product'] == product_class]
    upsampled_class_data = resample(class_data, replace=True, n_samples=32482)
    upsampled_data.append(upsampled_class_data)

upsampled_df = pd.concat(upsampled_data)


upsampled_df.reset_index(drop=True, inplace=True)


# In[18]:


upsampled_df.drop_duplicates()


# In[19]:


X=dataset['narrative']
y=dataset['product']


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state =26105111, stratify = y) 


# In[ ]:


import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# Download NLTK resources (if not already downloaded)
nltk.download('vader_lexicon')


# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()


# Function to perform sentiment analysis
def analyze_sentiment(text):
    sentiment_score = sia.polarity_scores(text)
    # Score interpretation: positive if compound score > 0, negative if < 0, neutral otherwise
    if sentiment_score['compound'] > 0.2:
        return 'Positive'
    elif sentiment_score['compound'] < -0.2:
        return 'Negative'
    else:
        return 'Neutral'


# Load your dataset (assuming 'data' is already loaded)


# Perform sentiment analysis on each narrative
sentiments = []
for index, row in data.iterrows():
    sentiment = analyze_sentiment(row['narrative'])
    sentiments.append(sentiment)


# Add sentiment labels to the dataset or store separately for further analysis
data['sentiment'] = sentiments


# Display the DataFrame with sentiment labels
print(data.head())


# Count the total number of positive and negative rows
positive_count = (data['sentiment'] == 'Positive').sum()
negative_count = (data['sentiment'] == 'Negative').sum()
total_count = len(data)


# Calculate accuracy
accuracy = ((positive_count + negative_count) / total_count) * 100


# Print the results
print("Total Positive Rows:", positive_count)
print("Total Negative Rows:", negative_count)
print("Total Rows:", total_count)
print("Accuracy:", accuracy, "%")


import pandas as pd
import matplotlib.pyplot as plt


# Assuming y_test is your target variable
# Calculate the counts of each unique value in y_test
value_counts = dataset['product'].value_counts()


# Calculate the percentage of each value
percentages = (value_counts / len(y_test)) * 100


# Plot the bar chart
plt.figure(figsize=(10, 6))
percentages.plot(kind='bar', color='skyblue')
plt.title('Distribution of complaints')
plt.xlabel('Categories')
plt.ylabel('Percentage')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()


import pandas as pd
import matplotlib.pyplot as plt


# Assuming y_test is your target variable
# Calculate the counts of each unique value in y_test
value_counts = dataset['product'].value_counts()


# Calculate the percentage of each value
percentages = (value_counts / len(y_test)) * 100


# Plot the pie chart
plt.figure(figsize=(8, 8))
plt.pie(percentages, labels=percentages.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of complaints')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.show()


# In[ ]:




