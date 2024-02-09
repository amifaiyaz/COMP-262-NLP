#!/usr/bin/env python
# coding: utf-8

# ## Code 1 to Code 9

# In[16]:


import pandas as pd
#import stop_words
import re
#get_ipython().system('python -m nltk.downloader stopwords')
from nltk.tokenize import sent_tokenize
#nltk.download('stopwords')


# In[17]:


t1 = pd.read_csv("C:/courses_centennial/COMP 262/nlp_book_apress-master/Chapter 3/Reviews.csv")


# In[18]:


t1.shape


# In[19]:


t1.head(212)


# In[20]:


print(t1.info())


# In[21]:


# Let us look at the score, we will use the score to understand and measure the accuracy of the lexicon-based approach.
t1["Score"].value_counts()


# ;Citing Lexicons
# ;
# 1. Minqing Hu and Bing Liu. "Mining and Summarizing Customer Reviews." 
# ;       Proceedings of the ACM SIGKDD International Conference on Knowledge  ;       Discovery and Data Mining (KDD-2004), Aug 22-25, 2004, Seattle, 
# ;       Washington, USA, 
# 2. Bing Liu, Minqing Hu and Junsheng Cheng. "Opinion Observer: Analyzing 
# ;       and Comparing Opinions on the Web." Proceedings of the 14th 
# ;       International World Wide Web conference (WWW-2005), May 10-14, 
# ;       2005, Chiba, Japan.

# In[22]:


pos1 = pd.read_csv("C:/courses_centennial/COMP 262/nlp_book_apress-master/Chapter 3/positive-words.txt",sep="\t",encoding='latin1',header=None)
neg1 = pd.read_csv("C:/courses_centennial/COMP 262/nlp_book_apress-master/Chapter 3/negative-words.txt",sep="\t",encoding='latin1',header=None)

pos1.columns = ["words"]
neg1.columns = ["words"]

pos_set = set(list(pos1["words"]))
neg_set = set(list(neg1["words"]))
print (len(pos_set))
print (len(neg_set))
print ( list(pos_set)[5])
print ( list(neg_set)[5])


# In[23]:


# In our “t1” dataset we have 2 columns of text - “Text” and “Summary”. 
# We would now combine them and process them into a single column. 
# It is this column that would go through the lexicon mining


# In[24]:


t1["full_txt"] = t1["Summary"] + " " + t1["Text"]
t1["full_txt"] = t1["full_txt"].str.lower()
t1["sent_len"] = t1["full_txt"].str.count(" ") + 1


# In[25]:


t1.head(5)


# In[26]:


#Exclude empty rows


# In[27]:


t2 = t1[t1.sent_len>=1]
len(t1),len(t2)

t2.info()


# In[28]:


##for meausring accuracy Set all to neutral then set an score =4 and above to  positve and any score less than =2 to negative
#t2["score_bkt"]="neu"
t2.loc[:,"score_bkt"]="neu"
t2.loc[t2.Score>=4,"score_bkt"] = "pos"
t2.loc[t2.Score<=2,"score_bkt"] = "neg"
t2.head()


# In[29]:


#take a sample of 10%
t3 = t2.sample(frac=0.1)
len(t2),len(t3)


# In[30]:


print(t2["score_bkt"].value_counts())
print(t2["score_bkt"].count())
print(t3["score_bkt"].count())

# To iterate through all words in a sentence corpus and hit against the list of lexicons. 
# Since these are longer sentences we would want to normalize the number of positive and negative hits by number of words in the sentences. 
# After this a simple comparison between the positive, negative and neutral scores is done and the sentences are tagged based on whichever scores are higher

# In[31]:


final_tag_list = []
pos_percent_list = []
neg_percent_list = []
pos_set_list = []
neg_set_list = []

for i,row in t3.iterrows():
    
    full_txt_set = set(row["full_txt"].split())
    sent_len = len(full_txt_set)
    
    pos_set1 = (full_txt_set) & (pos_set)
    neg_set1 = (full_txt_set) & (neg_set)
    
    com_pos = len(pos_set1)
    com_neg = len(neg_set1)
    
    if(com_pos>0):
        pos_percent = com_pos/sent_len
    else:
        pos_percent = 0

    
    if(com_neg>0):
        neg_percent = com_neg/sent_len
    else:
        neg_percent =0
        
    if(pos_percent>0)|(neg_percent>0):
        if(pos_percent>neg_percent):
            final_tag = "pos"
        else:
            final_tag = "neg"
    else:
        final_tag="neu"
    
    final_tag_list.append(final_tag)
    pos_percent_list.append(pos_percent)
    neg_percent_list.append(neg_percent)
    pos_set_list.append(pos_set1)
    neg_set_list.append(neg_set1)


# In[32]:


t3.head(2)


# In[33]:


t3["final_tags"] = final_tag_list
t3["pos_percent"] = pos_percent_list
t3["neg_percent"] = neg_percent_list

t3["pos_set"] = pos_set_list
t3["neg_set"] = neg_set_list


# In[34]:


t3.info()


# In[35]:


from sklearn.metrics import accuracy_score
print (accuracy_score(t3["score_bkt"],t3["final_tags"]))


# In[36]:


from sklearn.metrics import f1_score
f1_score(t3["score_bkt"],t3["final_tags"], average='macro')  


# In[37]:


rows_name = t3["score_bkt"].unique()

from sklearn.metrics import confusion_matrix
cmat = pd.DataFrame(confusion_matrix(t3["score_bkt"],t3["final_tags"], labels=rows_name, sample_weight=None))
cmat.columns = rows_name 
cmat["act"] = rows_name
cmat


# In[38]:


##we first investigate the errors by comparing the predicted sentiment and the actual tags


# In[39]:


pd.options.display.max_colwidth=1000
t3.loc[t3.score_bkt!=t3.final_tags,["Summary","full_txt","final_tags","score_bkt","pos_percent","neg_percent","pos_set","neg_set"]]


# In[40]:


### Accuracy without neutral to understand performance better


# In[41]:


t4 = t3.loc[(t3.score_bkt!="neu") & (t3.final_tags!="neu")].reset_index()
print (accuracy_score(t4["score_bkt"],t4["final_tags"]))
print (f1_score(t4["score_bkt"],t4["final_tags"],average='macro'))


# In[ ]:




