# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 06:06:21 2024

@author: amifa
"""

import pandas as pd

# Step 1: Load the dataset
dataset_path = "D:/COMP262-NLP/Assignment/Assignment 1/COVID19_data.csv"
df = pd.read_csv(dataset_path)

# Step 2: Drop the "user" column
df.drop(columns=['user'], inplace=True)


# Step 3: Data preprocessing
df['text'] = df['text'].str.replace(r'^\d+\s+|\s+\d+\s+|\s+\d+$', '')

# Step 4: Data exploration
print("Summary statistics of the dataset:")
print(df.describe())

# Step v: Add a column for tweet length
df['tweet_len'] = df['text'].apply(len)

# Print the dataframe to see the updated structure
print(df.head(5))

# Step vi: Load the positive and negative words lexicons into two dataframes
positive_lexicon_path = "D:/COMP262-NLP/Assignment/Assignment 1/positive lexicon.xlsx"
negative_lexicon_path = "D:/COMP262-NLP/Assignment/Assignment 1/negative lexicon.xlsx"

positive_lexicon_df = pd.read_excel(positive_lexicon_path, header=None)
negative_lexicon_df = pd.read_excel(negative_lexicon_path, header=None)

# Step vii: Iterate through all of the words in each tweet and hit against the list of lexicons
# Normalize the number of positive and negative hits by the number of words in each tweet
def calculate_sentiment_percentage(text, positive_lexicon_df, negative_lexicon_df):
    positive_count = 0
    negative_count = 0
    
    # Split text into words
    words = text.split()
    
    # Iterate through words in the text
    for word in words:
        # Check if word is in positive lexicon
        if word in positive_lexicon_df.values:
            positive_count += 1
        # Check if word is in negative lexicon
        elif word in negative_lexicon_df.values:
            negative_count += 1
    
    # Normalize sentiment scores by dividing by total number of words in the text
    total_words = len(words)
    if total_words > 0:
        positive_percentage = (positive_count / total_words) * 100
        negative_percentage = (negative_count / total_words) * 100
    else:
        positive_percentage = 0
        negative_percentage = 0
    
    return positive_percentage, negative_percentage

# Step viii: viii.	Add two columns to the datafrme for negative and positive
# Apply the function to each text in the dataframe
df['positive_percentage'], df['negative_percentage'] = zip(*df['text'].apply(
    lambda x: calculate_sentiment_percentage(x, positive_lexicon_df, negative_lexicon_df)))

# Step ix: Tag each tweet with a sentiment score
def tag_sentiment(row):
    if row['positive_percentage'] == 0 and row['negative_percentage'] == 0:
        return 'neutral'
    elif row['positive_percentage'] > row['negative_percentage']:
        return 'positive'
    elif row['negative_percentage'] > row['positive_percentage']:
        return 'negative'
    else:
        # In case of tie, consider it neutral
        return 'neutral'

# Apply the function to each row in the dataframe to create the predicted_sentiment_score column
df['predicted_sentiment_score'] = df.apply(tag_sentiment, axis=1)


# Step x: Compare original sentiments to predicted sentiments and calculate Accuracy and F1 score
from sklearn.metrics import accuracy_score, f1_score
accuracy = accuracy_score(df['sentiment'], df['predicted_sentiment_score'])
f1 = f1_score(df['sentiment'], df['predicted_sentiment_score'], average='weighted')

# Print the Accuracy and F1 score
print("Accuracy:", accuracy)
print("F1 Score:", f1)
