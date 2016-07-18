from __future__ import division
import json
import pandas 
import matplotlib as plt
import pprint
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import operator 
from collections import Counter, defaultdict
import string
from nltk import bigrams
import sys
import math
import vincent
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
from datetime import datetime
from flask import Flask,render_template
from bokeh.plotting import figure, output_file, show
from bokeh.resources import CDN
from bokeh.embed import file_html,components
from bokeh.charts import Bar, TimeSeries, output_file, show
##from bokeh.charts import TimeSeries, show, output_file

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
	
	
regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]

tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)


def tokenize(s):
    return tokens_re.findall(s)
	
def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

tweets_path='C:/Users/USER/Portfolio/twitter_data6.txt'

punctuation = list(string.punctuation)
stop = stopwords.words('english')+punctuation+['RT','via']

com = defaultdict(lambda : defaultdict(int))
tweets_file = open(tweets_path,'r')
dates_Nice=[]
dates_Baton=[]
dates_Turkey=[]
count_terms_all=Counter()
count_all=Counter()
count_hash=Counter()
count_only=Counter()
count_bigrams=Counter()
##search_word=sys.argv[1]
search_word = 'Nice'
count_search = Counter()
count_stop_single=Counter()
count_single=Counter()

for line in tweets_file:
    try:
	   tweet = json.loads(line)
	   
	   terms_all = [term for term in preprocess(tweet['text'])] ## all terms : yes(hashtags,stopwords)
	   terms_all_single = set(terms_all)   ## getting unique terms from all terms
	   terms_hash = [term for term in preprocess(tweet['text']) if term.startswith('#')]  ## only hash tags 
	   if '#Nice' in terms_hash:  ## tracking #Nice hash tag
	                      dates_Nice.append(tweet['created_at'])
	   if '#BatonRouge' in terms_hash:  ## tracking #Nice hash tag
	                      dates_Baton.append(tweet['created_at'])
	   '''				  
       if '#Turkey' in terms_hash:     
	                      dates_Trump.append(tweet['created_at'])
       '''						  
	   if '#Turkey' in terms_hash:     
	                      dates_Turkey.append(tweet['created_at'])   ## fetching the created_at field for the time of that hash tag				  
	   terms_only = [term for term in preprocess(tweet['text']) if term not in stop and not term.startswith(('#','@'))] ##terms only : no (hashtags,stopwords) 
	   terms_stop = [term for term in preprocess(tweet['text']) if term not in stop] ## terms all : yes(hashtags), no(Stopwords)
	   terms_stop_single = set(terms_stop)
	   terms_bigram = bigrams(terms_stop)  ## jointly occurening terms : 2 terms occurening together : from terms stop : yes(hashtags), no(stopwords)
	   count_terms_all.update(terms_all) ## counter for all terms : yes(hashtags,stopwords)
	   count_all.update(terms_stop) ## counter for all terms : yes(hashtags), no(stopwords)
	   count_hash.update(terms_hash)  ## counter for hash tags
	   count_only.update(terms_only)  ## counter for terms only : no(hashtags,stopwords)
	   count_bigrams.update(terms_bigram)   ## counter for bigrams
	   count_stop_single.update(terms_stop_single)
	   count_single.update(terms_all_single)
	   ## Co-occurrence matrix com: com[w1][w2] contains no. of times the term w1 has been seen in the same tweet as the term w2.
	   for i in range(len(terms_only)-1):            
            for j in range(i+1, len(terms_only)):
               w1, w2 = sorted([terms_only[i], terms_only[j]])                
               if w1 != w2:
                  com[w1][w2] += 1
	   if search_word in terms_only:
            count_search.update(terms_only)       
    except:
       continue
	   
com_max = []
# For each term, find out the most common co-occurring terms
for t1 in com:
    t1_max_terms = sorted(com[t1].items(), key=operator.itemgetter(1), reverse=True)[:5]
    for t2, t2_count in t1_max_terms:
        com_max.append(((t1, t2), t2_count))
		
# Getting the most frequent co-occurrences
terms_max = sorted(com_max, key=operator.itemgetter(1), reverse=True)
print "most common co-occurent :", (terms_max[:10])

print "most common words: ",count_all.most_common(5)
print "most common words only:", count_only.most_common(5)
print "most common hash tags:" , count_hash.most_common(30)
print "most common bigrams:", count_bigrams.most_common(5)  ### most common biagrams
print "most common terms:", count_terms_all.most_common(20)

## Getting the most common co-occurent with term given as a command line argument $ Python analysis.py James
print("Co-occurrence for %s:" % search_word)
print(count_search.most_common(20))

## Visualization on the frequency of hashtags

hash_freq = count_hash.most_common(10)
labels, freq = zip(*hash_freq)
print "labels", labels
print "freq", freq
data = {'data':freq, 'x': labels}
print "data:", data
df = pandas.DataFrame()
df['hashtags']=labels
df['counts']=freq
print df.head()
p = Bar(df, 'hashtags', values='counts', title="Most common Hashtags",xlabel='Hashtags',ylabel='Frequency',legend=False)
##p = Bar(df, label='hashtags', values='counts', agg='count', title="Most common Hashtags")
output_file("bar_chart.html")
show(p)


## Data Visualization of time series data
 	   
## Creating a list of ones to count the hashtags
nice_ones = [1] * len(dates_Nice)
baton_ones = [1] * len(dates_Baton)
turkey_ones = [1] * len(dates_Turkey)
## index of a series
nice_idx = pandas.DatetimeIndex(dates_Nice)
baton_idx = pandas.DatetimeIndex(dates_Baton)
turkey_idx = pandas.DatetimeIndex(dates_Turkey)
## Actual series
Nice = pandas.Series(nice_ones,index=nice_idx)
Baton = pandas.Series(baton_ones,index=baton_idx)
Turkey = pandas.Series(turkey_ones,index=turkey_idx)
##print "Nice:",Nice

## Bucketing per minute
per_minute_nice=Nice.resample('1Min',how = 'sum').fillna(0)
per_minute_baton=Baton.resample('1Min',how = 'sum').fillna(0)
per_minute_turkey=Turkey.resample('1Min',how = 'sum').fillna(0)

## All hastags together
Hashtags = dict(NiceAttack=per_minute_nice,BatonRouge=per_minute_baton,TurkeyCoup=per_minute_turkey)
## Putting everything in a dataframe
##All_Hashtags = pandas.DataFrame(data=Hashtags, index=per_minute_nice.index)
All_Hashtags = pandas.DataFrame(data=Hashtags)
###print All_Hashtags.head()

# Bucketing the dataframe again
All_Hashtags = All_Hashtags.resample('1Min', how='sum').fillna(0)   
print All_Hashtags.head()
print All_Hashtags.index

output_file("timeseries_chart.html")

t = TimeSeries(All_Hashtags, legend=True,
               title="Tweets", ylabel='Frequency')
show(t)

############################################################################################
