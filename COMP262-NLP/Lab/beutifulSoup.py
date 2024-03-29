# -*- coding: utf-8 -*-
"""
Created on Mon May 17 11:31:39 2021

@author: NLP 
"""

from bs4 import BeautifulSoup
from urllib.request import urlopen
myurl = "https://stackoverflow.com/questions/415511/ \
  how-to-get-the-current-time-in-python"
myurl = myurl.replace(" ","")
html = urlopen(myurl).read()

soupified = BeautifulSoup(html, "html.parser")
question = soupified.find("div", {"class": "question"})
questiontext = question.find("div", {"class": "s-prose js-post-body"})
print("Question: \n", questiontext.get_text().strip())


answer = soupified.find("div", {"class": "answer"}) 
answertext = answer.find("div", {"class": "s-prose js-post-body"})
print("Best answer: \n", answertext.get_text().strip())



