# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 05:47:56 2024

@author: amifa
"""

import pandas as pd

# Load the data
faiyaz_df = pd.read_csv("D:\COMP262-NLP\Assignment\Assignment 1\COVID19_mini.csv")  # Update the path
print(faiyaz_df.head())

import re
# Drop the 'user' column
faiyaz_df.drop(columns=['user'], inplace=True)


def clean_text(text):
    # Example: remove URLs and special characters
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)  # Remove @ and # symbols
    # Add more cleaning steps as necessary
    return text

faiyaz_df['text'] = faiyaz_df['text'].apply(clean_text)

faiyaz_df['text'] = faiyaz_df['text'].str.lower()


import nlpaug.augmenter.word as naw
import random

# Initialize the Word2Vec augmenter
augmenter = naw.WordEmbsAug(model_type='word2vec', action="substitute")

def augment_text(text):
    # Tokenization and stop words removal can be integrated here if necessary
    words = text.split()
    # Randomly select three words to augment
    selected_words = random.sample(words, min(3, len(words)))
    for word in selected_words:
        # Get synonyms and replace in the text
        augmented_words = augmenter.augment(word)
        text = text.replace(word, augmented_words, 1)
    return text

# Apply augmentation and duplicate the dataset
augmented_texts = faiyaz_df['text'].apply(augment_text)
augmented_df = faiyaz_df.copy()
augmented_df['text'] = augmented_texts

# Combine original and augmented dataframes
faiyaz_df_after_word_augmenter = pd.concat([faiyaz_df, augmented_df], ignore_index=True)

# Export the augmented dataset
faiyaz_df_after_word_augmenter.to_csv('faiyaz_df_after_random_insertion.txt', index=False, sep='\t')
