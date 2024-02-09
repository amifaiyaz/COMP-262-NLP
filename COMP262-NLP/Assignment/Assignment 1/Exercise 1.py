# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 04:50:51 2024

@author: amifa
"""

from bs4 import BeautifulSoup
import requests

myurl = "https://www.centennialcollege.ca/programs-courses/full-time/artificial-intelligence"
html = requests.get(myurl).text
soupified = BeautifulSoup(html, "html.parser")

# Using the CSS selectors to extract the desired information
title = soupified.find("div", {"class": "your-title-class-here"}).get_text().strip()
vocational_outcomes = soupified.find("div", {"class": "your-vocational-outcomes-class-here"}).get_text().strip()
overview = soupified.find_all("div", {"class": "your-overview-class-here"}, limit=2)


# Printing the extracted information
print("Title:", title)
print("Vocational Outcome:", vocational_outcomes)
print("Overview:")
for paragraph in overview:
    print(paragraph.get_text().strip())
