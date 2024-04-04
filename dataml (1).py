#!/usr/bin/env python
# coding: utf-8

# **TEAM TARA**

# IMPORTING ALL NEEDED FUNCTIONS AND LIBRARIES

# In[788]:




import re
import numpy as np
import pandas as pd
import seaborn as sns
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


# In[789]:


data=pd.read_csv("complaints.csv")   #complaints.csv: customer complaints dataset 
data


# In[790]:


data.columns


# DATA PREPROCESSING

# In[791]:


data=data.dropna(how='any')
data


# In[792]:


data.isna().sum()


# In[793]:


data.shape


# In[ ]:


stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're','s', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']


# In[794]:


data_1 = data[data['product'] == 'credit_card']
data_2 = data[data['product'] == 'debt_collection']
data_3 = data[data['product'] == 'retail_banking']
data_4 = data[data['product'] == 'credit_reporting']
data_5 = data[data['product'] == 'mortgages_and_loans']


# In[ ]:


data_1


# In[ ]:


data_2


# In[ ]:


data_3


# In[ ]:


data_4


# In[ ]:


data_5


# In[795]:


dataset = pd.concat([data_1, data_2, data_3, data_4, data_5])


# In[796]:


import pandas as pd
from sklearn.utils import resample

upsampled_data = []
for product_class in df['product'].unique():
    class_data = df[df['product'] == product_class]
    upsampled_class_data = resample(class_data, replace=True, n_samples=32482)
    upsampled_data.append(upsampled_class_data)

upsampled_df = pd.concat(upsampled_data)


upsampled_df.reset_index(drop=True, inplace=True)


# In[797]:


upsampled_df.drop_duplicates()


# In[798]:


import pandas as pd
value_counts =upsampled_df['product'].value_counts()


# Calculate the percentage of each value
percentages = (value_counts / len(upsampled_df)) * 100

print(percentages)


# In[799]:


upsampled_df.shape


# In[800]:


upsampled_df['narrative']=upsampled_df['narrative'].str.lower()


# In[801]:


dataset=upsampled_df


# In[802]:


STOPWORDS = set(stopwordlist)   #REMOVING STOPWORDS FROM THE DATASET
def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])
dataset['narrative'] = dataset['narrative'].apply(lambda text: cleaning_stopwords(text))
dataset['narrative'].head(10)


# In[803]:


import string                         #REMOVING PUNCTUATIONS FROM THE DATASET
english_punctuations = string.punctuation
punctuations_list = english_punctuations
def cleaning_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)
dataset['narrative']= dataset['narrative'].apply(lambda x: cleaning_punctuations(x))
dataset['narrative'].head(20)


# In[804]:


def cleaning_repeating_char(text):               #REMOVING REPETITIVE CHARS FROM THE DATASET
    return re.sub(r'(.)1+', r'1', text)
dataset['narrative'] = dataset['narrative'].apply(lambda x: cleaning_repeating_char(x))
dataset['narrative'].head(45)


# In[805]:


dataset.head()
dataset.drop(dataset.columns[0], axis=1, inplace=True)


# In[806]:


is_present = dataset.apply(lambda col: col.str.contains('name', case=False, na=False)).any().any().sum()


# In[807]:


import re

def remove_repetitive_words(text):
    words = re.findall(r'\b\w+\b', text)
    unique_words = set(words)
    cleaned_text = ' '.join(unique_words)
    return cleaned_text

dataset['narrative'] = dataset['narrative'].apply(remove_repetitive_words)


# In[808]:


from textblob import Word
dataset['narrative'] = dataset['narrative'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


# In[809]:


lemmatizer = WordNetLemmatizer()         #LEMMATIZATION 

def lemmatize_sentence(sentence):
    tokens = word_tokenize(sentence)  # Tokenizing the sentence into words
    lemmatized_sentence = ' '.join([lemmatizer.lemmatize(word) for word in tokens])  # Lemmatize each word
    return lemmatized_sentence


dataset['narrative'] = dataset['narrative'].apply(lemmatize_sentence)
dataset.head()


# In[810]:


dataset['narrative'] = dataset['narrative'].apply(lambda x: x.split())
dataset.head()


# In[811]:


from sklearn.preprocessing import LabelEncoder   #LABEL ENCODING
label_encoder = LabelEncoder()

dataset['product'] = label_encoder.fit_transform(dataset['product'])

# Retrieve the mapping between original categories and encoded labels
category_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

print("Category Mapping:")
for category, label in category_mapping.items():
    print(f"{category}: {label}")


# In[812]:


X=dataset['narrative']
y=dataset['product']


# In[813]:


import pandas as pd
value_counts =dataset['product'].value_counts()


# Calculate the percentage of each value
percentages = (value_counts / len(dataset)) * 100


# Print the percentages
print(percentages)


# In[814]:


from sklearn.feature_extraction.text import TfidfVectorizer      #VECTORIZATION

# Ensure X_train contains only strings
X = [' '.join(tokens) for tokens in X]

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features = 10000)
X = vectorizer.fit_transform(X)

print('No. of feature words:', len(vectorizer.get_feature_names_out()))


# SPLITTING DATASET INTO TESTING AND TRAINING DATA

# In[815]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state =26105111, stratify = y) 


# In[816]:


X_train.shape


# APPLYING ML MODELS

# In[817]:


mnb = MultinomialNB()    #MULTINOMIALNB
mnb.fit(X_train, y_train)


# In[818]:


y_pred = mnb.predict(X_test)
accuracy = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average="weighted")

print("Accuracy:", accuracy)
print("F1 Score:", f1)


# In[819]:


from sklearn.ensemble import RandomForestClassifier  #RANDOMFOREST

classifier_rf = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=5,
                                       n_estimators=100, oob_score=True)

classifier_rf.fit(X_train, y_train)


# In[820]:


y_pred = classifier_rf.predict(X_test)
accuracy = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average="weighted")

print("Accuracy:", accuracy)
print("F1 Score:", f1)


# In[822]:


classifier_rf.oob_score_


# In[ ]:


rf = RandomForestClassifier(random_state=42, n_jobs=-1)

params = {
    'max_depth': [2,3,5,10,20],
    'min_samples_leaf': [5,10,20,50,100,200],
    'n_estimators': [10,25,30,50,100,200]
}

from sklearn.model_selection import GridSearchCV

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rf,
                           param_grid=params,
                           cv = 4,
                           n_jobs=-1, verbose=1, scoring="accuracy")

grid_search.fit(X_train, y_train)


# In[ ]:


import lightgbm as lgb     #LIGHTGBM MODEL
clf = lgb.LGBMClassifier()
clf.fit(X_train, y_train)


# In[ ]:


y_pred=clf.predict(X_test)


# In[ ]:


accuracy_score(y_test, y_pred)


# In[ ]:


import xgboost as xgb     #XGBOOST
from xgboost import XGBClassifier
xgb_cl = xgb.XGBClassifier()
xgb_cl.fit(X_train, y_train)
predxgb = xgb_cl.predict(X_test)
accuracy_score(y_test, predxgb)


# XGBOOST GAVE THE MOST ACCURATE RESULT (88.75%)

# In[784]:


def classify_complaint(narrative):            #VALIDATIONS ON CUSTOMER INPUT
    # Check if the narrative is empty or too short
    if not narrative or len(narrative.strip()) < 10:  # Adjust the minimum length as needed
        print("Invalid complaint: Narrative is empty or too short.")
        return
    
    # Preprocess the input narrative (similar to what you did for the dataset)
    narrative = narrative.lower()
    narrative = cleaning_stopwords(narrative)
    narrative = cleaning_punctuations(narrative)
    narrative = cleaning_repeating_char(narrative)
    narrative = remove_repetitive_words(narrative)
    narrative = ' '.join([Word(word).lemmatize() for word in narrative.split()])
    narrative = lemmatize_sentence(narrative)
    
    # Vectorize the preprocessed narrative
    narrative_vector = vectorizer.transform([narrative])
    
    # Predict the class using the XGBoost classifier
    predicted_class = xgb_cl.predict(narrative_vector)[0]
    
    # Map the predicted class label to the original category
    category_mapping_inverse = {v: k for k, v in category_mapping.items()}
    predicted_category = category_mapping_inverse.get(predicted_class, "Unknown")
    
    print("Predicted category:", predicted_category)

narrative = input("Enter your complaint narrative: ")
classify_complaint(narrative)


# In[786]:


from langdetect import detect

def classify_complaint(narrative):
    # Check if the narrative is empty or too short
    if not narrative or len(narrative.strip()) < 10:  # Adjust the minimum length as needed
        print("Invalid complaint: Narrative is empty or too short.")
        return
    
    # Detect the language of the narrative
    try:
        language = detect(narrative)
    except:
        print("Error: Unable to detect language.")
        return
    
    # Check if the language is English
    if language != 'en':
        print("Invalid complaint: The complaint is not in English.")
        return
    
    # Preprocess the input narrative (similar to what you did for the dataset)
    narrative = narrative.lower()
    narrative = cleaning_stopwords(narrative)
    narrative = cleaning_punctuations(narrative)
    narrative = cleaning_repeating_char(narrative)
    narrative = remove_repetitive_words(narrative)
    narrative = ' '.join([Word(word).lemmatize() for word in narrative.split()])
    narrative = lemmatize_sentence(narrative)
    
    # Vectorize the preprocessed narrative
    narrative_vector = vectorizer.transform([narrative])
    
    # Predict the class using the XGBoost classifier
    predicted_class = xgb_cl.predict(narrative_vector)[0]
    
    # Map the predicted class label to the original category
    category_mapping_inverse = {v: k for k, v in category_mapping.items()}
    predicted_category = category_mapping_inverse.get(predicted_class, "Unknown")
    
    print("Predicted category:", predicted_category)


narrative = input("Enter your complaint narrative: ")
classify_complaint(narrative)


# SENTIMENT ANALYSIS AND DATA VISUALISATION

# In[ ]:



from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
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


# Performing sentiment analysis on each narrative
sentiments = []
for index, row in data.iterrows():
    sentiment = analyze_sentiment(row['narrative'])
    sentiments.append(sentiment)

data['sentiment'] = sentiments

print(data.head())


# Counting the total number of positive and negative rows
positive_count = (data['sentiment'] == 'Positive').sum()
negative_count = (data['sentiment'] == 'Negative').sum()
total_count = len(data)


# Calculating the accuracy
accuracy = ((positive_count + negative_count) / total_count) * 100


# Printing the results
print("Total Positive Rows:", positive_count)
print("Total Negative Rows:", negative_count)
print("Total Rows:", total_count)
print("Accuracy:", accuracy, "%")


#BAR GRAPH

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

#PIE CHART:

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




