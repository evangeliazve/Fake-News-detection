#!/usr/bin/env python
# coding: utf-8

#Master's Degree (2nd Year) - Statistics and Econometrics
#Evangelia Grigoria ZVE
#Master Thesis Chapter 3: Fake News Prediction Algorithm
#This code accompanies the 3rd Chapter of my manuscript.
#Written in Python 3

##Loading Packages and functions
get_ipython().system(' pip install python-louvain')
get_ipython().system(' pip install gensim')
get_ipython().system(' pip install vaderSentiment')
get_ipython().system(' pip install lexical-diversity')
get_ipython().system(' pip install tag-fixer')
get_ipython().system(' pip install yellowbrick')

import numpy as np
import re  
import nltk  
from sklearn.datasets import load_files  
from collections import Counter
import community
import pandas as pd
import nltk
import os.path
import sys
import itertools
from itertools import chain
from nltk.collocations import *
from nltk.text import Text
import networkx as nx
from scipy import stats
from nltk import pos_tag
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from collections import Counter
from gensim.models import Word2Vec
from nltk.parse.generate import generate, demo_grammar
from nltk.sentiment import SentimentAnalyzer, SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from lexical_diversity import lex_div as ld
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from nltk.stem import PorterStemmer
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from scipy.stats import chi2_contingency
from scipy.stats import chi2


import matplotlib
from matplotlib import pyplot as plt
# import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


##Loading Datasets
#loading the training and test sets separately
#Importing trainning set
docs_to_train = load_files('D:/Master 2 FOAD/Memoire/data_competition/news', categories='training')
#importing test set
docs_to_test = load_files('D:/Master 2 FOAD/Memoire/data_competition/news', categories='test')

#importing file which indicates whether the news in the training set is fake (1) or real (0)
file_training = 'D:/Master 2 FOAD/Memoire/data_competition/labels_training.txt'
labels_training = pd.read_csv(file_training, dtype={'doc': str,'class': int})

#importing file which indicates whether the news in the test set is fake (1) or real (0)
file_test = 'D:/Master 2 FOAD/Memoire/data_competition/labels_test.txt'
labels_test = pd.read_csv(file_test, dtype={'doc': str,'class': int})

#importing news-user and user-user files
file_news_user = pd.read_csv('D:/Master 2 FOAD/Memoire/data_competition/newsUser.txt', delim_whitespace=True, header=None, names=['User ID', 'News ID','Nb of spreads'])
file_user_user = pd.read_csv('D:/Master 2 FOAD/Memoire/data_competition/UserUser.txt', delim_whitespace=True, header=None, names=['source', 'target'])

##Some descriptive statistics on the loaded dataset
#Number of news in the training set:
print("%d train documents" % len(docs_to_train.data))
# Number of news in the test set:
print("%d test documents" % len(docs_to_test.data))

#plotting the number of documents in training and test set
names = ['Nb. of Train Documents', 'Nb. of Test Documents']
values = [len(docs_to_train.data), len(docs_to_test.data)]

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.bar(names, values)

#pie chart with percentage of training and test.
fig1, ax1 = plt.subplots()
ax1.pie(values, labels=names, autopct='%1.1f%%')
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()

#number of unique users that shared an article
file_news_user["User ID"].unique
file_user_user["target"].unique

##Data Preparation
#changing data type of 'doc' variable of the 'labels_training' file to string.
#fill in the target values for the documents/news to train
for i in range(1, len(docs_to_train.target)):
      if str(labels_training['doc'][i]) in docs_to_train.filenames[i]:
         docs_to_train.target[i] = labels_training['class'][i]

#Transform Bunch of sklearn objects to pandas dataframes
#TRAINING
data_train = pd.DataFrame(data=np.c_[docs_to_train['data']], columns=['News'])
data_train['News ID']=docs_to_train['filenames']

#values cleaning for the variable News ID
data_train['News ID']=data_train['News ID'].str[56:]
data_train['News ID']=data_train['News ID'].str.replace('.txt','')

#transforming variables 'News ID' and 'doc' to string in order to be able to join the tables
data_train['News ID']=data_train['News ID'].astype(str)
labels_training['doc']=labels_training['doc'].astype(str)

#join data_train and labels_training tables on News ID
data_train=pd.merge(data_train, labels_training, how='left', left_on='News ID', right_on='doc')
data_train=data_train.drop('doc',axis=1)
data_train=data_train.rename(index=str,columns={'class': 'Target'})

#TEST
#proceeding the same way as for the training set but I don't include the Target variables.
data_test = pd.DataFrame(data=np.c_[docs_to_test['data']], columns=['News'])
data_test['News ID']=docs_to_test['filenames']

#values cleaning for the variables News ID
data_test['News ID']=data_test['News ID'].str[52:]
data_test['News ID']=data_test['News ID'].str.replace('.txt','')

#Transforming variables 'News ID' and 'doc' to string in order to be able to join the tables
data_test['News ID']=data_test['News ID'].astype(str)
labels_test['doc']=labels_test['doc'].astype(str)

#Join data_test and labels_test tables on News ID
data_test=pd.merge(data_test, labels_test, how='left', left_on='News ID', right_on='doc')
data_test=data_test.drop('doc',axis=1)
data_test=data_test.rename(index=str,columns={'class': 'Target'})


### RELATED CHAPTER IN MANUSCRIPT : "3.4.1 Dataset description"
#True and Fake News in the dataset
Target_nb=data_train.groupby('Target').count()[['News ID']]
Target_nb
#Ploting the above results
Target_nb.plot(kind='bar')
plt.show()


### RELATED CHAPTER IN MANUSCRIPT : "3.2 Social Context Features Extraction - Network Analysis"
##3.2.1 Community Detection
#Hypothesis : Fake news spreaders are part of different network communities compared to true news spreaders
#Related Feature : Number of communities per news
#Related Feature: Most fréquent community per news

#Creating a GRAPH using NetworkX package
G=nx.from_pandas_edgelist(file_user_user, source='source', target='target')

#Louvain heuristices algorithm implementation
#I create new Node attribute that is called " Community ID " and indicates for each node to which community is assigned.
communities = community.best_partition(G)

#number of communitites
max(communities.values())

#attributing community "-1" for Users that are part of communities of less than 4 users. They considered as Users out of communities.
for k, v in communities.items():
    if v > 3:
        communities[k] = -1
        
    else:
        communities[k] = v

#number of communitites after  the above changes.
max(communities.values())

#Adding Nodes attribus to graphs
nx.set_node_attributes(G, communities, 'Community ID')

##Some descriptive analysis
#Number of users per community
#Community "0"
sum(value == 0 for value in communities.values())
#Community "1"
sum(value == 1 for value in communities.values())
#Community "2"
sum(value == 2 for value in communities.values())
#Community "3"
sum(value == 3 for value in communities.values())

#Number of Edges and Nodes in the Graph
print('# of edges: {}'.format(G.number_of_edges()))
print('# of nodes: {}'.format(G.number_of_nodes()))
      
#plot number of users per community
names = ['Community 0', 'Community 1', 'Community 2', 'Community 3','people out of communitites']
values = [4537, 3447, 2530, 12442, 909]

#pie chart of the uders per community
fig1, ax1 = plt.subplots()
ax1.pie(values, labels=names, autopct='%1.1f%%')
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


#Plotting Network graph
import networkx as nx
#drawing
size = float(len(set(communities.values())))
pos = nx.spring_layout(G)

count = 0.


for com in set(communities.values()) :
         count = count + 1.
         list_nodes = [nodes for nodes in communities.keys()
                                     if communities[nodes] == com]

nx.draw_networkx_nodes(G, pos, node_size=1,node_color=list(communities.values()))     
nx.draw_networkx_edges(G, pos, width=0.4)
plt.show()

#Plotting again withou the Community "-1"
for k, v in list(communities.items()):
    if v == -1:
        del communities[k]

#creating subgraph without community "-1"
list_nodes_sub= list(communities.keys())
H = G.subgraph(list_nodes_sub)

nx.draw_networkx_nodes(H, pos, node_size=1,node_color=list(communities.values()))     
nx.draw_networkx_edges(H, pos, width=0.1)
plt.show()

#plotting the grph again
nx.draw(G, pos, node_color=list(communities.values()), node_size=5);
plt.show()

#Creating dataframes with the users and the ID of the louvain community in which they were assigned,
CommunityIDs=nx.get_node_attributes(G,'Community ID')
CommunityIDs=pd.DataFrame.from_dict(CommunityIDs, orient='index')
CommunityIDs.columns = ['Community ID']
CommunityIDs['User ID']=CommunityIDs.index
CommunityIDs = CommunityIDs.reset_index()
CommunityIDs = CommunityIDs.drop(columns='index')
CommunityIDs['Community ID']=CommunityIDs['Community ID'].astype(str)


#Join news-user dataframe with the above dataframe.
result = pd.merge(CommunityIDs,
                 file_news_user,
                 on='User ID', 
                 how='right')


#RELATED CHAPTER IN THE MANUSCRIPT : 3.2.2 Network Centrality (User Influence)
#Hypothesis : fake news spreaders have a higher centrality score compared to true news spreaders
#Features: Average Eigenvector Centrality score per News

# Calculating Eigenvector centrality for each node in the Graph
EC=nx.eigenvector_centrality(G)

#Creating dataframes with the users and their eigenvector centrality scores
EC=pd.DataFrame.from_dict(EC, orient='index')
EC.columns = ['Eigenvector Centrality']
EC['User ID']=EC.index
EC = EC.reset_index()
EC = EC.drop(columns='index')

#Join news-user dataframe (with Community ID included) and the above dataframe in order to add centrality scores.
result = pd.merge(result,
                 EC,
                 on='User ID', 
                 how='left')


#Hypothesis : fake news spreaders (nodes) have significantly diffrent Page Rank score compared to true news spreaders.
#Features : Average PageRank score per News using NetworkX
page_rank = nx.pagerank(G)

#Creating dataframes with the users and the pagerank score
page_rank=pd.DataFrame.from_dict(page_rank, orient='index')
page_rank.columns = ['page_rank']
page_rank['User ID']=page_rank.index
page_rank = page_rank.reset_index()
page_rank = page_rank.drop(columns='index')

#Join news-user dataframe (with Community ID included) and the above dataframe in order to add centrality scores.
result = pd.merge(result,
                 page_rank,
                 on='User ID', 
                 how='left')


#RELATED CHAPTER IN THE MANUSCRIPT : 3.2.3 News Propagation - General measures
#List of  Hypothesis

# 1. More users spread fake news than true news (More-spreader hypothesis 1)
# 2. Spreaders engage more strongly with fake news than with true news (Stronger-Engagement hypothesis 2)
# 3. A Fake News is spreaded more time that the True News (Stronger-Engagement hypothesis 2)

# List of features:

# Number of unique users involved in spreading each fake or true news (related to Hypothesis 1)
# Individual Engagements (average spreading frequencies of all users who have participated in the news propagation) (related to Hypothesis 2)
# Group Engagements (total number of times that the news story has been spread)(related to Hypothesis 3)

## Aggregating Data by News (features creation in the main/final dataset)
# I aggregate the above results table with News ID as primary key.
# I create the dataframe 'result_agg' which contains the aggregated values mentionned above in order to create the features.
get_most_common = lambda values: max(Counter(values).items(), key = lambda x: x[1])[0]

result['Community ID']=result['Community ID'].astype(int)

result_agg=result.groupby('News ID', as_index=False).agg({'Community ID':get_most_common,# find the sum of the durations for each group
                                     'User ID':  lambda x: x.nunique(), # Nb of unique users per news
                                     'Nb of spreads': 'sum', #total number of spreads per news
                                     'Eigenvector Centrality' : 'mean', #average of Eigenvector Centrality
                                     'page_rank' : 'mean' #average Page Rank
                                     }) 

#converting News ID variable of the data_train table to float in order to be able to join on a common key
data_train['News ID']=data_train['News ID'].astype(float)

#join result_agg and data_train on News ID
data_train1=pd.merge(data_train, result_agg, on='News ID', sort=False)

#Average number of spreads per user per news.
result_agg_avgspreads=result.groupby('News ID', as_index=False).agg({'Nb of spreads': {'Avg Nb of spreads': "mean"}})  
result_agg_avgspreads.columns=['News ID', 'Avg Nb of spreads']

#Number of communities per news
result_agg_nbcommunities=result.groupby('News ID', as_index=False).agg({'Community ID': {'Nb of communities per News': lambda x: x.nunique()}})    
result_agg_nbcommunities.columns=['News ID', 'Nb of communities per News']

#merging previous tables to include average spread information & number of communities per news
#trainning set
data_train1=pd.merge(data_train1,result_agg_avgspreads, on='News ID', sort=False)
data_train1=pd.merge(data_train1,result_agg_nbcommunities, on='News ID', sort=False)
#test set
data_test['News ID']=data_test['News ID'].astype(float)
data_test1=pd.merge(data_test, result_agg, on='News ID', sort=False)
#merging
data_test1=pd.merge(data_test1,result_agg_avgspreads, on='News ID', sort=False)
data_test1=pd.merge(data_test1,result_agg_nbcommunities, on='News ID', sort=False)


#RELATED CHAPTER IN MANUSCRIPT: 3.2.3 News Propagation - General measures
#Group Engagements (total number of times that the news story has been spread)(related to Hypothesis 3)
#Create a boxplot
data_train1.boxplot('Nb of spreads', by='Target', figsize=(12, 8))
#plotting results
plt.bar(data_train1["Target"].astype(str), data_train1["Nb of spreads"], alpha=0.5)
plt.ylabel('Number of Spreads')
plt.title('Target')

plt.show()

#Independent t-test using scipy.stats
from scipy import stats
stats.ttest_ind(data_train1["Nb of spreads"][data_train1["Target"]==0], data_train1["Nb of spreads"][data_train1["Target"]==1])
#Point-biserial correlation test
stats.pointbiserialr(np.array(data_train1["Nb of spreads"]), np.array(data_train1["Target"]))
#Create a boxplot
data_train1.boxplot('Avg Nb of spreads', by='Target', figsize=(12, 8))

#Individual Engagements (average spreading frequencies of all users who have participated in the news propagation)(related to Hypothesis 2)
plt.bar(data_train1["Target"].astype(str), data_train1["Avg Nb of spreads"], alpha=0.5)
plt.ylabel("Avg Nb of spreads")
plt.title("Target")

plt.show()

#Point-biserial correlation test
stats.pointbiserialr(np.array(data_train1["Avg Nb of spreads"]), np.array(data_train1["Target"]))
#Independent t-test using scipy.stats
stats.ttest_ind(data_train1["Avg Nb of spreads"][data_train1["Target"]==0], data_train1["Avg Nb of spreads"][data_train1["Target"]==1])

#Number of unique users involved in spreading each fake or true news (related to Hypothesis 1)
#Create a barplot
plt.bar(data_train1["Target"].astype(str), data_train1["User ID"], alpha=0.5)
plt.ylabel("User ID")
plt.title("Target")

plt.show()
#Point-biserial correlation test
stats.pointbiserialr(np.array(data_train1["User ID"]), np.array(data_train1["Target"]))
#Independent t-test using scipy.stats
stats.ttest_ind(data_train1["User ID"][data_train1["Target"]==0], data_train1["User ID"][data_train1["Target"]==1])

#RELATED TO CHAPTER IN THE MANUSCRIPT: 3.2.1 Community Detection
##Descriptive analysis
# Most frequent community per News
data_train1['Community ID'] = data_train1['Community ID'].astype('category')
data_train1['Target'] = data_train1['Target'].astype('category')

#Contigency table
cont_table=pd.crosstab(data_train1['Community ID'], data_train1['Target'])
#print table
print(cont_table)
#Chi-2 Statistical Test
stat, p, dof, expected = chi2_contingency(cont_table)
print('dof=%d' % dof)
# interpret test-statistic
prob = 0.95
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')
# interpret p-value
alpha = 1.0 - prob
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')

#plotting most frequent community per News
pd.crosstab(data_train1['Community ID'], data_train1['Target']).plot.bar()
#plotting most frequent Target value per Community ID
pd.crosstab(data_train1['Target'], data_train1['Community ID']).plot.bar()


#Number of different communities occurred per type of News Article
#Transform variable to categorical
data_train1['Nb of communities per News'] = data_train1['Nb of communities per News'].astype('category')
#Contigency table
cont_table=pd.crosstab(data_train1['Nb of communities per News'], data_train1['Target'])
#print table
print(cont_table)
#plotting the type of News article per Community
cont_table.plot.bar()
#plotting the number of communities per type of News Article
pd.crosstab(data_train1['Target'], data_train1['Nb of communities per News']).plot.bar()

#Chi-2 Statistical Test
stat, p, dof, expected = chi2_contingency(cont_table)
print('dof=%d' % dof)
# interpret test-statistic
prob = 0.95
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')
# interpret p-value
alpha = 1.0 - prob
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')


##DESCRIPTIVE ANALYSIS FOR COMMUNITY DETECTION
#RELATED TO CHAPTER IN THE MANUSCRIPT: 3.2.2 Network Centrality and User Importance
    
#Analyse descriptive - Centrality & User Influence
#Creating a boxplot for Eigenvector Centrality per True and Fake News
data_train1.boxplot('Eigenvector Centrality', by='Target', figsize=(12, 8))

#Independent t-test using scipy.stats
stats.ttest_ind(data_train1["Eigenvector Centrality"][data_train1["Target"]==0], data_train1["Eigenvector Centrality"][data_train1["Target"]==1])
stats.pointbiserialr(np.array(data_train1["Eigenvector Centrality"]), np.array(data_train1["Target"]))

#Page Rank
#Create a boxplot with Page Rank Values per True and Fake News
data_train1.boxplot('page_rank', by='Target', figsize=(12, 8))

#Independent t-test using scipy.stats
stats.ttest_ind(data_train1["page_rank"][data_train1["Target"]==0], data_train1["page_rank"][data_train1["Target"]==1])
#Point-biserial correlation test
stats.pointbiserialr(np.array(data_train1["page_rank"]), np.array(data_train1["Target"]))

#Independent t-test using scipy.stats
from scipy import stats
stats.ttest_ind(data_train1["User ID"][data_train1["Target"]==0], data_train1["page_rank"][data_train1["Target"]==1])


#RELATED CHAPTER IN MANUSCRIPT: 3.4.1 Dataset description
#data shape
data_train1.shape

# extracting the number of examples of each class
Fake_len = data_train1[data_train1['Target'] == 1].shape[0]
True_len = data_train1[data_train1['Target'] == 0].shape[0]
print(Fake_len)
print(True_len)

#barplot of the number of examples of each class
plt.bar(10,Fake_len,3, label="Fake News")
plt.bar(15,True_len,3, label="True News")
plt.legend()
plt.ylabel('Number of news')
plt.title('Fake vs True News')
plt.show()

#RELATED TO CHAPTER IN MANUSCRIPT: 3.3.2 Data Preparation

# Preparing both Training and Test sets. I will apply the below mentionned techniques of data cleaning and pre-processing.
# 
# __Removing useless punctuation__
# 
# __Lowercase words__
# 
# __Tokenization__
# 
# __POS Tagging preparation__
# 
# __Stop words removal__
# 
# __Stemming__
# 

##Removing useless Punctuation

#defining the punctuations to remove
punctuation1='"#$%&\'()*+,-./:;<=>@[\\]^_`{|}~'
table_news = str.maketrans(dict.fromkeys(punctuation1))

#create function that removes punctuation from a string
def remove_punctuation_news(text):
      return str(text).translate(table_news);

#apply punctuation removal to the training set
data_train1['News'] = data_train1['News'].apply(remove_punctuation_news)
#data_train1['News'][0]

#apply punctuation removal to the test set
data_test1['News'] = data_test1['News'].apply(remove_punctuation_news)
#data_test1['News'][0]


##Word Tokenization
#train set
data_train1['News_tokenized']=1
for i, row in enumerate(data_train1['News']):
    data_train1['News_tokenized'][i] = nltk.word_tokenize(data_train1['News'][i].decode('utf-8'))
print(data_train1['News_tokenized'][0])

#test set
data_test1['News_tokenized']=1
for i, row in enumerate(data_test1['News']):
    data_test1['News_tokenized'][i] = nltk.word_tokenize(data_test1['News'][i].decode('utf-8'))


##Sentence tokenization
trainer = PunktTrainer()
tokenizer = PunktSentenceTokenizer(trainer.get_params())
 
#training set 
data_train1['News_sent_tokenized']=data_train1['News'].astype(str).apply(tokenizer.tokenize)
data_train1['News_sent_tokenized'][0]
 
#test set
data_test1['News_sent_tokenized']=data_test1['News'].astype(str).apply(tokenizer.tokenize)
data_test1['News_sent_tokenized'][0]


##POS Tagging - data preparation
#train set
data_train1['News_POStag']=1
data_train1['News_tokenized']=data_train1['News_tokenized'].astype(str)
data_train1['News_POStag'] = data_train1['News_tokenized'].str.split().map(pos_tag)
data_train1['News_POStag'].head()

#test set
data_test1['News_POStag']=1
data_test1['News_tokenized']=data_train1['News_tokenized'].astype(str)
data_test1['News_POStag'] = data_train1['News_tokenized'].str.split().map(pos_tag)
data_test1['News_POStag'].head()


##Removing english Stopwords
#extracting the stopwords from nltk library
sw = stopwords.words('english')
#displaying the stopwords
np.array(sw)
print("Number of stopwords: ", len(sw))

#train set
data_train1['News_tokenized']=1
for i, row in enumerate(data_train1['News']):
    data_train1['News_tokenized'][i] = nltk.word_tokenize(data_train1['News'][i].decode('utf-8'))
    
data_train1['News_tokenized'] = data_train1['News_tokenized'].apply(lambda x: [item for item in x if item not in sw])
data_train1['News_tokenized'][0]

#test set
data_test1['News_tokenized']=1
for i, row in enumerate(data_test1['News']):
    data_test1['News_tokenized'][i] = nltk.word_tokenize(data_test1['News'][i].decode('utf-8'))

data_test1['News_tokenized'] = data_test1['News_tokenized'].apply(lambda x: [item for item in x if item not in sw])
data_test1['News_tokenized'][0]

#train set for sentences
data_train1['News_sent_tokenized'] = data_train1['News_sent_tokenized'].apply(lambda x: [item for item in x if item not in sw])
data_train1['News_sent_tokenized'][0]

#test set for sentences
data_test1['News_sent_tokenized'] = data_test1['News_sent_tokenized'].apply(lambda x: [item for item in x if item not in sw])
data_test1['News_sent_tokenized'][0]


##Stemming
stemmer=PorterStemmer()
#train set
data_train1['News_tokenized_stemmed'] = data_train1['News_tokenized'].apply(lambda x: [stemmer.stem(y) for y in x])
data_train1['News_tokenized_stemmed'][0]

#test set
data_test1['News_tokenized_stemmed'] = data_test1['News_tokenized'].apply(lambda x: [stemmer.stem(y) for y in x])
data_test1['News_tokenized_stemmed'][0]

#train set for sentences
data_train1['News_tokenized_sent_stemmed'] = data_train1['News_sent_tokenized'].apply(lambda x: [stemmer.stem(y) for y in x])
data_train1['News_tokenized_sent_stemmed'][6]

#test set for sentences
data_test1['News_tokenized_sent_stemmed'] = data_test1['News_sent_tokenized'].apply(lambda x: [stemmer.stem(y) for y in x])
data_test1['News_tokenized_sent_stemmed'][5]


##RELATED TO CHAPTER IN MANUSCRIPT: 3.3.3 Textual Features Engineering
#Style-based Features

#3.3.3.1 Lexical Features - General

# Lexical features include character-level  and  word-level  features.

# List of Hypothesis

# 1. The number of words varies significantly between Fake and True News
# 2. The number of unique words varies significantly between Fake and True News
# 3. The appearance of words in a text varies significantly between Fake and True News
# 4. The appearance of combination of words in a text varies significantly between Fake and True News
# 5. The importance of words in a text varies significantly between Fake and True News

# List of features

# 1. Number of words per News (related to hypothesis 1)
# 2. Number of unique words per News (related to hypothesis 2)

# - Bag of words approches :
# 3. CountVectorizer features extraction including unigrams and bigrams(related to hypothesis 3,4)
# 4. TfidfVectorizer  features extraction (related to hypothesis 5)

#Extracting Number of Words per News
# - Train :
data_train1['count_word']=0
data_train1['count_word']=data_train1['News_tokenized_stemmed'].apply(len)

# - Test :
data_test1['count_word']=0
data_test1['count_word']=data_test1['News_tokenized_stemmed'].apply(len)


#Extracting Number of Unique Words per news
# - Train : 
data_train1['count_word_unique']=data_train1['News_tokenized_stemmed'].apply(set)
data_train1['count_word_unique']=data_train1['count_word_unique'].apply(len)

# - Test :
data_test1['count_word_unique']=data_test1['News_tokenized_stemmed'].apply(set)
data_test1['count_word_unique']=data_test1['count_word_unique'].apply(len)


#Bag of word approaches

#CountVectorizer (including Unigrams and Bigrams)

# CountVectorizer offers a easy way of both tokenizing a collection of texts and building a vocabulary of known words, but also encoding new data using that vocabulary. 
# 
# I use it as follows: 
# 
# - I create a CountVectorizer class instance. 
# - To learn a vocabulary from one or more articles, I call the fit() function. 
# - I call the transform() function to encode each as a vector on one or more documents. 
# 

# I will first evaluate the optimal parametres with GridSearch for Random Forest and Naves Bayes classifiers. Once I have the optimal number of n-grams and make a decision on whether to use TfidfTransformer features, then I will try different models with the chosen number n-grams and the most efficient features extraction technique.

# Identifying optimal model parameters

# My objective in this section is to evaluate the need to use both CountVectorizer and TfidfTransformer as well as selecting which ngrams I will take into account for my analysis. Finally, in the next section, I select the top 100 features of CountVectorizer and the top 100 features of TfidfTransformer. The models that I will be using and compare during our analysis are Naive Bayes and Support Vector Machines.

# #### Naive Bayes

# Here, we are creating a list of parameters for which we would like to do performance tuning. 
# All the parameters name start with the classifier name (remember the arbitrary name we gave). 
# E.g. vect__ngram_range; here we are telling to use unigram and bigrams and choose the one which is optimal.

text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB())])

text_clf = text_clf.fit(data_train1["News"], data_train1["Target"])
predicted = text_clf.predict(data_train1["News"])
np.mean(predicted == data_train1["Target"])

parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False)}

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(data_train1["News"].astype(str), data_train1["Target"])

# To see the best mean score and the params, run the following code
print(gs_clf.best_score_)
print(gs_clf.best_params_)

# Random Forest classifier

text_clf_rf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-rf', RandomForestClassifier())])

text_clf_rf = text_clf_rf.fit(data_train1["News"], data_train1["Target"])
predicted_rf = text_clf_rf.predict(data_train1["News"])
np.mean(predicted_rf == data_train1["Target"])

parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False)}

gs_clf_rf = GridSearchCV(text_clf_rf, parameters, n_jobs=-1)
gs_clf_rf = gs_clf_rf.fit(data_train1["News"].astype(str), data_train1["Target"])

# To see the best mean score and the params, run the following code
print(gs_clf_rf.best_score_)
print(gs_clf_rf.best_params_)

#Model fitting :
#Naive Bayes
word_vectorizer_nb = CountVectorizer(ngram_range=(1,1), analyzer='word',lowercase=False, stop_words=None, max_features=50)
sparse_matrix_nb = word_vectorizer_nb.fit_transform(data_train1['News'].astype(str))
#Random Forest
word_vectorizer_rf = CountVectorizer(ngram_range=(1,2), analyzer='word',lowercase=False, stop_words=None, max_features=50)
sparse_matrix_rf = word_vectorizer_rf.fit_transform(data_train1['News'].astype(str))


#Creating a Dataframe with the results
#Naive Bayes
X_train_countvectorizer_123grams_nb=pd.DataFrame(sparse_matrix_nb.toarray(), columns=word_vectorizer_nb.get_feature_names()).add_suffix('_count_vect')
#Random Forest
X_train_countvectorizer_123grams_rf=pd.DataFrame(sparse_matrix_rf.toarray(), columns=word_vectorizer_rf.get_feature_names()).add_suffix('_count_vect')

#Creating Dataframes with the selected features
#Naive Bayes
#word_vectorizer_nb = CountVectorizer(ngram_range=(1,1), lowercase=False,vocabulary=dict_nb,stop_words='english')
word_vectorizer_nb = CountVectorizer(ngram_range=(1,1), lowercase=False,stop_words='english',max_features=50)
sparse_matrix_nb = word_vectorizer_nb.fit_transform(data_train1['News'].astype(str))
X_train_countvectorizer_123grams_nb=pd.DataFrame(sparse_matrix_nb.toarray(), columns=word_vectorizer_nb.get_feature_names()).add_suffix('_count_vect')

#Random Forest
#word_vectorizer_rf = CountVectorizer(ngram_range=(1,2), lowercase=False,vocabulary=dict_rf,stop_words='english')
word_vectorizer_rf = CountVectorizer(ngram_range=(1,2), lowercase=False,stop_words='english',max_features=50)
sparse_matrix_rf = word_vectorizer_rf.fit_transform(data_train1['News'].astype(str))
X_train_countvectorizer_123grams_rf=pd.DataFrame(sparse_matrix_rf.toarray(), columns=word_vectorizer_rf.get_feature_names()).add_suffix('_count_vect')


# Concatenate with main dataset - Training set 
# Main Dataset = the one that includes also the social context features
#Naives Bayes
res_train_nb = pd.concat([data_train1, X_train_countvectorizer_123grams_nb], axis=1)
print(res_train_nb.head())

#Random Forest
res_train_rf = pd.concat([data_train1, X_train_countvectorizer_123grams_rf], axis=1)
print(res_train_rf.head())


# - Concatenate with main dataset - Test set :
#Naive Bayes
#word_vectorizer_test_nb = CountVectorizer(ngram_range=(1,1), analyzer='word',vocabulary=dict_nb, stop_words=None)
sparse_matrix_test_nb = word_vectorizer_nb.transform(data_test1['News'])

#Random Forest
#word_vectorizer_test_svm = CountVectorizer(ngram_range=(1,2), analyzer='word',vocabulary=dict_svm, stop_words=None)
sparse_matrix_test_rf = word_vectorizer_rf.transform(data_test1['News'])

#Naive Bayes
X_test_countvectorizer_123grams_nb=pd.DataFrame(sparse_matrix_test_nb.toarray(), columns=word_vectorizer_nb.get_feature_names()).add_suffix('_count_vect')

#Random Forest
X_test_countvectorizer_123grams_rf=pd.DataFrame(sparse_matrix_test_rf.toarray(), columns=word_vectorizer_rf.get_feature_names()).add_suffix('_count_vect')

#Naive Bayes
res_test_nb = pd.concat([data_test1, X_test_countvectorizer_123grams_nb], axis=1)
print(res_test_nb.head())

#Random Forest
res_test_rf = pd.concat([data_test1, X_test_countvectorizer_123grams_rf], axis=1)
print(res_test_rf.head())


# TFIDF (term frequency–inverse document frequency)

# With the TfidfVectorizer() function I first tokenize, then count tokens, then transform the raw counts to TF/IDF values.

# - Training set :
# 
# I select only the 50 most important words ("max_features" parameter)

# Naive Bayes:
v = TfidfVectorizer(analyzer='word', stop_words='english',max_features=50)
x = v.fit_transform(data_train1['News'])
#Convert the tokenized data into a dataframe:
X_train_tfidf_df = pd.DataFrame(x.toarray(), columns=v.get_feature_names())
print(X_train_tfidf_df)


# Random Forest:
v_rf = TfidfVectorizer(analyzer='word', stop_words='english',max_features=50, ngram_range=(1,2))
x_rf = v_rf.fit_transform(data_train1['News'])
#Convert the tokenized data into a dataframe:
X_train_tfidf_df_rf = pd.DataFrame(x_rf.toarray(), columns=v_rf.get_feature_names())
print(X_train_tfidf_df_rf)


# Concatenate the tokenization dataframe to the orignal one:
#Naive Bayes
res_train1_nb = pd.concat([res_train_nb, X_train_tfidf_df], axis=1)
print(res_train1_nb.head())

#Random Forest (I don't add tf-idf metrics to my dataframe)
res_train1_rf = pd.concat([res_train_rf, X_train_tfidf_df_rf], axis=1)
print(res_train1_rf.head())


# - Test set :
x1 = v.transform(data_test1['News'])
#Convert the tokenized data into a dataframe:
X_test_tfidf_df = pd.DataFrame(x1.toarray(), columns=v.get_feature_names())
print(X_test_tfidf_df)

#Naive Bayes
res_test1_nb = pd.concat([res_test_nb, X_test_tfidf_df], axis=1)
print(res_test1_nb.head())

#Random Forest
res_test1_rf = pd.concat([res_test_rf, X_test_tfidf_df], axis=1)


#3.3.3.2 Syntactic Features

# #### List of hypothesis

# 1. The number of sentences per News varies significantly between Fake and True News
# 2. The average lenght of sentence per news varies significantly between Fake and True News
# 3. The frequencies of the different grammatical categories in a News varies significantly between Fake & True News
# 4. The presence of "?" and "!" punctuations varies significantly between Fake and True News. More punctions likely shows that there is a higher propability that a News is Fake.
# 5. The lexical diversity varies significantly between Fake and True News. Less lexical diversity likely means that there is a higher propability that a News is Fake.

# #### List of features

# 1. Number of sentences per News (related to hypothesis 1)
# 2. Average length of sentence per News (related to hypothesis 2)
# 3. POS Tagging Features extraction (related to hypothesis 3)
# 4. Punctuations counts (related to hypothesis 4)
# 5. Lexical Diversity level using Text-Type Ratio (TTR)

# #### Number of sentences per News

# Counting the number of senetences per News by using the already tokenized text to sentences

# - Training set :
data_train1['count_sentence']=0
data_train1['count_sentence']=data_train1['News_sent_tokenized'].apply(len)
data_train1['count_sentence'][5]

# - Test set :

data_test1['count_sentence']=0
data_test1['count_sentence']=data_test1['News_sent_tokenized'].apply(len)
data_test1['count_sentence'][5]


# ####Average length of sentence per News

# - Training set

average_sentence_length = lambda x :sum([len(sentence) for sentence in x])/len(x)
data_train1['average_sentence_length']=data_train1['News_sent_tokenized'].apply(average_sentence_length)

print(data_train1['average_sentence_length'])


# - Test set

average_sentence_length = lambda x :sum([len(sentence) for sentence in x])/len(x)
data_test1['average_sentence_length']=data_test1['News_sent_tokenized'].apply(average_sentence_length)


# ####POS Tagging Features extraction

# - Training set

import itertools
from itertools import chain

tokens, tags = zip(*chain(*data_train1['News_POStag'].tolist()))

possible_tags = sorted(set(tags))

possible_tags_counter = Counter({p:0 for p in possible_tags})
possible_tags_counter


data_train1['News_POStag'].apply(lambda x: Counter(list(zip(*x))[1]))
data_train1['pos_counts'] = data_train1['News_POStag'].apply(lambda x: Counter(list(zip(*x))[1]))
data_train1['pos_counts']

def add_pos_with_zero_counts(counter, keys_to_add):
    for k in keys_to_add:
        counter[k] = counter.get(k, 0)
    return counter

# Detailed steps.
data_train1['pos_counts'] = data_train1['News_POStag'].apply(lambda x: Counter(list(zip(*x))[1]))
data_train1['pos_counts_with_zero'] = data_train1['pos_counts'].apply(lambda x: add_pos_with_zero_counts(x, possible_tags))
data_train1['sent_vector'] = data_train1['pos_counts_with_zero'].apply(lambda x: [count for tag, count in sorted(x.most_common())])

# All in one.
data_train1['sent_vector'] = data_train1['News_POStag'].apply(lambda x:
    [count for tag, count in sorted(
        add_pos_with_zero_counts(
            Counter(list(zip(*x))[1]), 
                    possible_tags).most_common()
         )
    ]
)

data_train1_t = pd.DataFrame(data_train1['sent_vector'].tolist())
data_train1_t.columns = possible_tags

data_train1=pd.merge(data_train1,data_train1_t, left_index=True, right_index=True)
data_train1=data_train1.drop("''",axis=1)


#  - Test set :

tokens, tags = zip(*chain(*data_test1['News_POStag'].tolist()))

possible_tags = sorted(set(tags))

possible_tags_counter = Counter({p:0 for p in possible_tags})
possible_tags_counter


data_test1['News_POStag'].apply(lambda x: Counter(list(zip(*x))[1]))
data_test1['pos_counts'] = data_test1['News_POStag'].apply(lambda x: Counter(list(zip(*x))[1]))
data_test1['pos_counts']

def add_pos_with_zero_counts(counter, keys_to_add):
    for k in keys_to_add:
        counter[k] = counter.get(k, 0)
    return counter

#Detailed steps.
data_test1['pos_counts'] = data_test1['News_POStag'].apply(lambda x: Counter(list(zip(*x))[1]))
data_test1['pos_counts_with_zero'] = data_test1['pos_counts'].apply(lambda x: add_pos_with_zero_counts(x, possible_tags))
data_test1['sent_vector'] = data_test1['pos_counts_with_zero'].apply(lambda x: [count for tag, count in sorted(x.most_common())])

#All in one.
data_test1['sent_vector'] = data_test1['News_POStag'].apply(lambda x:
    [count for tag, count in sorted(
        add_pos_with_zero_counts(
            Counter(list(zip(*x))[1]), 
                    possible_tags).most_common()
         )
    ]
)

data_test1_t = pd.DataFrame(data_test1['sent_vector'].tolist())
data_test1_t.columns = possible_tags

data_test1=pd.merge(data_test1,data_test1_t, left_index=True, right_index=True)
data_test1=data_test1.drop("''",axis=1)


# #### Number of punctuations in each News

# I only take into account "?" and "!" as punctuations as I consider these ones as the most important.

# - Training set

data_train1['News']=data_train1['News'].astype(str)
data_train1['punctuation_counts']=0  
punctuation='!?'
for i in range(len(data_train1['News'])):      
         for f in data_train1['News'][i].astype(str):
            if f in punctuation:
                 data_train1['punctuation_counts'][i]=data_train1['punctuation_counts'][i]+1


# - Test set
data_test1['News']=data_test1['News'].astype(str)
data_test1['punctuation_counts']=0  
punctuation='!?'
for i in range(len(data_test1['News'])):      
         for f in data_test1['News'][i].astype(str):
            if f in punctuation:
                 data_test1['punctuation_counts'][i]=data_test1['punctuation_counts'][i]+1


# ####Lexical Diversity

# - Training set
data_train1['lexical_diversity_ttr']=data_train1['News_tokenized_stemmed'].apply(ld.ttr)
data_train1['lexical_diversity_ttr']


# - Testset :
data_test1['lexical_diversity_ttr']=data_test1['News_tokenized_stemmed'].apply(ld.ttr)
data_test1['lexical_diversity_ttr']

#3.3.3.3 Sentiment analysis

# Here I will use the polarity method by calculating the polarity_scores using VADER SentimentIntensityAnalyzer.
# 
# I classify a text as positive when the rate is higher than 0.05, as negative when the rate is less than - 0.05 and neutral when it's in between.
# 
# Bacause I must have only numeric features to train my model afterwords, I will define the values as follows :
# 
# - Positive = 3
# - Neutral = 2
# - Negative = 1

def sentiment_scores(sentence): 
  
    # Create a SentimentIntensityAnalyzer object. 
    sid_obj = SentimentIntensityAnalyzer() 
  
    # polarity_scores method of SentimentIntensityAnalyzer 
    # oject gives a sentiment dictionary. 
    # which contains pos, neg, neu, and compound scores. 
    sentiment_dict = sid_obj.polarity_scores(sentence) 
     
    # decide sentiment as positive, negative and neutral 
    if sentiment_dict['compound'] >= 0.05 : 
        return("3") 
  
    elif sentiment_dict['compound'] <= - 0.05 : 
        return("1") 
  
    else : 
        return("2")    


#Changing the feature type to integer
data_train1['sentiment_scores']=data_train1['News'].astype(str).apply(sentiment_scores)
data_train1['sentiment_scores']=data_train1['sentiment_scores'].astype(int)

data_test1['sentiment_scores']=data_test1['News'].astype(str).apply(sentiment_scores)
data_test1['sentiment_scores']=data_test1['sentiment_scores'].astype(int)

###################################################################
# ## Dataset preparation & cleaning before training models

# ###Dataset combining both textual & social context data

# ##### Deleting useless features in our final dataset

#train set
train_set=[]
train_set=data_train1.drop(["News ID","pos_counts","News","pos_counts_with_zero","News_tokenized","News_sent_tokenized","sent_vector","News_tokenized_sent_stemmed","News_POStag","News_tokenized_stemmed"],axis=1)

#test set
test_set=[]
test_set=data_test1.drop(["News ID","pos_counts","News","News_tokenized","pos_counts_with_zero","News_sent_tokenized","sent_vector","News_tokenized_sent_stemmed","News_POStag","News_tokenized_stemmed"], axis=1)


# ##### All features must be in Float or Integer format before traning the model.
test_set.dtypes
train_set.dtypes


# ##### Merging final dataset with CountVectorizer & TFIDF vectorizer features

# - Training set

#Naive Bayes
res_training_nb = pd.concat([train_set,X_train_countvectorizer_123grams_nb], axis=1)
res_training_nb = pd.concat([res_training_nb,X_train_tfidf_df], axis=1)
res_training_nb.head(10)

#Random Forest
res_training_rf = pd.concat([train_set,X_train_countvectorizer_123grams_rf], axis=1)
res_training_rf = pd.concat([res_training_rf,X_train_tfidf_df], axis=1)
res_training_rf.head(10)

# - Test set

#Naive Bayes
res_testing_nb = pd.concat([test_set,X_test_countvectorizer_123grams_nb], axis=1)
res_testing_nb = pd.concat([res_testing_nb,X_test_tfidf_df], axis=1)
res_testing_nb.head(10)

#Random Forest
res_testing_rf = pd.concat([test_set,X_test_countvectorizer_123grams_rf], axis=1)
res_testing_rf = pd.concat([res_testing_rf,X_test_tfidf_df], axis=1)
res_testing_rf.head(10)

##############################################################################

#RELATED TO CHAPTER IN MANUSCRIPT 3.3.3 Textual Features Engineering
# ### Descriptive analysis of the features
res_training_nb.dtypes

# - ## Word count
#Barplot for word_count
plt.bar(res_training_nb["Target"].astype(str), res_training_nb["count_word"], alpha=0.5)
plt.ylabel('Number of Words in News')
plt.title('Target')

plt.show()

#Create a boxplot for count_word
res_training_nb.boxplot('count_word', by='Target', figsize=(12, 8))

#Independent t-test using scipy.stats : count_word
from scipy import stats
stats.ttest_ind(res_training_nb["count_word"][res_training_nb["Target"]==0], res_training_nb["count_word"][res_training_nb["Target"]==1])

# - ### Unique Words per News

#Barplot
plt.bar(res_training_nb["Target"].astype(str), res_training_nb["count_word_unique"], alpha=0.5)
plt.ylabel('Nb of unique words')
plt.title('Target')

plt.show()

#Create a boxplot
res_training_nb.boxplot('count_word_unique', by='Target', figsize=(12, 8))

#Independent t-test using scipy.stats
from scipy import stats
stats.ttest_ind(res_training_nb["count_word_unique"][res_training_nb["Target"]==0], res_training_nb["count_word_unique"][res_training_nb["Target"]==1])


# - ### CountVectorizer

import yellowbrick
from yellowbrick.text import FreqDistVisualizer
from yellowbrick.datasets import load_hobbies

###NAIVE BAYES CLASSIFIER
#ALL NEWS - PLOTTING count vectors
features   = word_vectorizer_nb.get_feature_names()

visualizer = FreqDistVisualizer(features=features, orient='v')
visualizer.fit(sparse_matrix_nb)
visualizer.poof()

#TRUE NEWS
# Create a dict to map target labels to documents of that category
from collections import defaultdict
response = defaultdict(list)
for text, label in zip(data_train1['News'].astype(str), data_train1['Target'].astype(str)):
    response[label].append(text)

features = word_vectorizer_nb.get_feature_names()
#Plotting top 50 features
visualizer = FreqDistVisualizer(features=features, orient='v')
visualizer.fit(word_vectorizer_nb.fit_transform(text for text in response['0']))
visualizer.poof()

#FAKE NEWS
# Create a dict to map target labels to documents of that category
response = defaultdict(list)
for text, label in zip(data_train1['News'].astype(str), data_train1['Target'].astype(str)):
    response[label].append(text)

features   = word_vectorizer_nb.get_feature_names()
#Plotting top 50 features
visualizer = FreqDistVisualizer(features=features, orient='v')
visualizer.fit(word_vectorizer_nb.fit_transform(text for text in response['1']))
visualizer.poof()

###RANDOM FOREST CLASSIFIER
# ALL NEWS - Plotting Count vectors
features   = word_vectorizer_rf.get_feature_names()
#Plotting top 50 features
visualizer = FreqDistVisualizer(features=features, orient='v')
visualizer.fit(sparse_matrix_rf)
visualizer.poof()

#FAKE NEWS
# Create a dict to map target labels to documents of that category
response = defaultdict(list)
for text, label in zip(data_train1['News'].astype(str), data_train1['Target'].astype(str)):
    response[label].append(text)

features = word_vectorizer_rf.get_feature_names()
#Plotting top 50 features
visualizer = FreqDistVisualizer(features=features, orient='v')
visualizer.fit(word_vectorizer_rf.fit_transform(text for text in response['0']))
visualizer.poof()

#TRUE NEWS
# Create a dict to map target labels to documents of that category
response = defaultdict(list)
for text, label in zip(data_train1['News'].astype(str), data_train1['Target'].astype(str)):
    response[label].append(text)

features = word_vectorizer_rf.get_feature_names()
#Plotting top 50 features
visualizer = FreqDistVisualizer(features=features, orient='v')
visualizer.fit(word_vectorizer_rf.fit_transform(text for text in response['1']))
visualizer.poof()


# - ### TFIDF
# Similar to Count Vectorizer

# Naive Bayes
#ALL NEWS
features   = v.get_feature_names()

visualizer = FreqDistVisualizer(features=features, orient='v')
visualizer.fit(x)
visualizer.poof()

# TRUE NEWS
# Create a dict to map target labels to documents of that category
response = defaultdict(list)
for text, label in zip(data_train1['News'].astype(str), data_train1['Target'].astype(str)):
    response[label].append(text)

features = v.get_feature_names()

visualizer = FreqDistVisualizer(features=features, orient='v')
visualizer.fit(v.fit_transform(text for text in response['0']))
visualizer.poof()

# FAKE NEWS
# Create a dict to map target labels to documents of that category
response = defaultdict(list)
for text, label in zip(data_train1['News'].astype(str), data_train1['Target'].astype(str)):
    response[label].append(text)

features = v.get_feature_names()

visualizer = FreqDistVisualizer(features=features, orient='v')
visualizer.fit(v.fit_transform(text for text in response['1']))
visualizer.poof()

# Random forest classifier
# ALL NEWS
features   = v_rf.get_feature_names()

visualizer = FreqDistVisualizer(features=features, orient='v')
visualizer.fit(x)
visualizer.poof()

# TRUE NEWS
# Create a dict to map target labels to documents of that category
response = defaultdict(list)
for text, label in zip(data_train1['News'].astype(str), data_train1['Target'].astype(str)):
    response[label].append(text)

features = v_rf.get_feature_names()

visualizer = FreqDistVisualizer(features=features, orient='v')
visualizer.fit(v_rf.fit_transform(text for text in response['0']))
visualizer.poof()

#FAKE NEWS
# Create a dict to map target labels to documents of that category
response = defaultdict(list)
for text, label in zip(data_train1['News'].astype(str), data_train1['Target'].astype(str)):
    response[label].append(text)

features = v_rf.get_feature_names()

visualizer = FreqDistVisualizer(features=features, orient='v')
visualizer.fit(v_rf.fit_transform(text for text in response['1']))
visualizer.poof()


#3.3.3.2 Syntactic Analysis

# 1)	The number of sentences per news varies significantly between Fake and True News

res_training_nb.dtypes

#Barplot
plt.bar(res_training_nb["Target"].astype(str), res_training_nb["count_sentence"], alpha=0.5)
plt.ylabel('Number of sentences per News')
plt.title('Target')

plt.show()
#Boxplot
res_training_nb.boxplot('count_sentence', by='Target', figsize=(12, 8))

#Student's test
stats.ttest_ind(res_training_nb["count_sentence"][res_training_nb["Target"]==0], res_training_nb["count_sentence"][res_training_nb["Target"]==1])


# 2)	The average length of a sentence per news varies significantly between Fake and True News

#Barplot
plt.bar(res_training_nb["Target"].astype(str), res_training_nb["average_sentence_length"], alpha=0.5)
plt.ylabel('average_sentence_length')
plt.title('Target')

plt.show()

#Boxplot
res_training_nb.boxplot('average_sentence_length', by='Target', figsize=(12, 8))

#Student's test
stats.ttest_ind(res_training_nb["average_sentence_length"][res_training_nb["Target"]==0], res_training_nb["average_sentence_length"][res_training_nb["Target"]==1])


# 3)	The frequencies of the different grammatical categories in a News varies significantly between Fake & True News

#Barplot - CD
plt.bar(res_training_nb["Target"].astype(str), res_training_nb["CD"], alpha=0.5)
plt.ylabel('Cardinal Digit')
plt.title('Target')

plt.show()

#Student's test - CD
stats.ttest_ind(res_training_nb["CD"][res_training_nb["Target"]==0], res_training_nb["CD"][res_training_nb["Target"]==1])

#Barplot - JJ
plt.bar(res_training_nb["Target"].astype(str), res_training_nb["JJ"], alpha=0.5)
plt.ylabel('adjective')
plt.title('Target')

plt.show()

#Student's test - JJ
stats.ttest_ind(res_training_nb["JJ"][res_training_nb["Target"]==0], res_training_nb["JJ"][res_training_nb["Target"]==1])

#Barplot - NN
plt.bar(res_training_nb["Target"].astype(str), res_training_nb["NN"], alpha=0.5)
plt.ylabel('noun singular')
plt.title('Target')

plt.show()

#Student's test - NN
stats.ttest_ind(res_training_nb["NN"][res_training_nb["Target"]==0], res_training_nb["NN"][res_training_nb["Target"]==1])

#Barplot - NNS
plt.bar(res_training_nb["Target"].astype(str), res_training_nb["NNS"], alpha=0.5)
plt.ylabel('noun plural')
plt.title('Target')

plt.show()

#Student's test - NNS
stats.ttest_ind(res_training_nb["NNS"][res_training_nb["Target"]==0], res_training_nb["NNS"][res_training_nb["Target"]==1])

#Barplot - POS
plt.bar(res_training_nb["Target"].astype(str), res_training_nb["POS"], alpha=0.5)
plt.ylabel('possessive ending parent’s')
plt.title('Target')

plt.show()

#Student's test - POS
stats.ttest_ind(res_training_nb["POS"][res_training_nb["Target"]==0], res_training_nb["POS"][res_training_nb["Target"]==1])


# 4)	The presence of "?" and "!" punctuations varies significantly between Fake and True News. More punctuations likely show that there is a higher probability that a News is Fake.

#Barplot
plt.bar(res_training_nb["Target"].astype(str), res_training_nb["punctuation_counts"], alpha=0.5)
plt.ylabel('punctuation_counts')
plt.title('Target')

plt.show()

#Boxplot
res_training_nb.boxplot('punctuation_counts', by='Target', figsize=(12, 8))

#Student's test
stats.ttest_ind(res_training_nb["punctuation_counts"][res_training_nb["Target"]==0], res_training_nb["punctuation_counts"][res_training_nb["Target"]==1])


# 5)	Lexical diversity varies significantly between Fake and True News. Less lexical diversity likely means that there is a higher probability that a News is Fake.

#Barplot
plt.bar(res_training_nb["Target"].astype(str), res_training_nb["lexical_diversity_ttr"], alpha=0.5)
plt.ylabel('lexical_diversity_ttr')
plt.title('Target')

plt.show()

#Boxplot
res_training_nb.boxplot('lexical_diversity_ttr', by='Target', figsize=(12, 8))

#Student's test
stats.ttest_ind(res_training_nb["lexical_diversity_ttr"][res_training_nb["Target"]==0], res_training_nb["lexical_diversity_ttr"][res_training_nb["Target"]==1])

#3.3.3.2 Sentiment analysis

#Contigency table - Sentiment score pet True and Fake News
cont_table=pd.crosstab(res_training_nb['sentiment_scores'].astype("category"), res_training_nb['Target'])

print(cont_table)

stat, p, dof, expected = chi2_contingency(cont_table)
print('dof=%d' % dof)

from scipy.stats import chi2
# interpret test-statistic
prob = 0.95
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')
# interpret p-value
alpha = 1.0 - prob
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')

#barplot - target value by sentiment scores
pd.crosstab(res_training_nb['sentiment_scores'], res_training_nb['Target']).plot.bar()

#bar plot - sentiment scores by target value
pd.crosstab(data_train1['Target'], data_train1['sentiment_scores']).plot.bar()

###################################################################
# Data Preparation Tasks

# ### Dataset with only textual data
#train set
train_set_text=[]
train_set_text=train_set.drop(["page_rank","Community ID","User ID","Nb of spreads","Eigenvector Centrality","Avg Nb of spreads","Nb of communities per News"],axis=1)

#test set
test_set_text=[]
test_set_text=test_set.drop(["page_rank","Community ID","User ID","Nb of spreads","Eigenvector Centrality","Avg Nb of spreads","Nb of communities per News"], axis=1)


# ##### Merging final dataset with CountVectorizer & TFIDF vectorizer features(only for Naive Bayes)

# - Training set
#Naive Bayes
res_training_nb_text = pd.concat([train_set_text,X_train_countvectorizer_123grams_nb], axis=1)
res_training_nb_text = pd.concat([res_training_nb_text,X_train_tfidf_df], axis=1)
res_training_nb_text.head(10)

#Random Forest
res_training_rf_text = pd.concat([train_set_text,X_train_countvectorizer_123grams_rf], axis=1)
res_training_rf_text = pd.concat([res_training_rf_text,X_train_tfidf_df], axis=1)
res_training_rf_text.head(10)

# - Test test
#Naive Bayes
res_testing_nb_text = pd.concat([test_set_text,X_test_countvectorizer_123grams_nb], axis=1)
res_testing_nb_text = pd.concat([res_testing_nb_text,X_test_tfidf_df], axis=1)
res_testing_nb_text.head(10)

#Random Forest
res_testing_rf_text = pd.concat([test_set_text,X_test_countvectorizer_123grams_rf], axis=1)
res_testing_rf_text = pd.concat([res_testing_rf_text,X_test_tfidf_df], axis=1)
res_testing_rf_text.head(10)

# ### Dataset with only social data
#train set
train_set_social=[]
train_set_social=train_set[["Target","page_rank","Community ID","User ID","Nb of spreads","Eigenvector Centrality","Avg Nb of spreads","Nb of communities per News"]]

#test set
test_set_social=[]
test_set_social=test_set[["Target","page_rank","Community ID","User ID","Nb of spreads","Eigenvector Centrality","Avg Nb of spreads","Nb of communities per News"]]


# ## Final Features - Data preparation tasks

# Selection of features is a technique where you automatically select those features in your data that most contribute to the prediction variable you are interested in. (reference : https://www.kaggle.com/jepsds/feature-selection-using-selectkbest?utm_campaign=News&utm_medium=Community&utm_source=DataCamp.com)
# 
# Model accuracy will be reduced by having too many irrelevant features in your data. 3 The advantages of selecting the feature before modeling your data are: 
# - Lowers overfitting: less redundant data means less opportunity to make noise-based decisions.
# - Improves accuracy: Modeling accuracy improves with less misleading information.
# - Less data implies that algorithms train more quickly.
# 
# Here I put in place the Recursive feature elimination with cross-validation (RFECV)technique with the help of RFECV() function of the sklearn package that basically realize feature ranking with recursive feature elimination and cross-validated selection of the best number of features.(references : https://www.scikit-yb.org/en/latest/api/features/rfecv.html)



#RELATED TO CHAPTER IN MANUSCRIPT :  3.4.2.2 Features Selection Methodology

#Selecting final features using Naive Bayes classifier

#### - Both textual & Social context features

clf = MultinomialNB()

res_training_nb.shape

res_training_nb["Community ID"]=res_training_nb["Community ID"].replace(-1, 4)

from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

# classifications
rfecv = RFECV(estimator=clf,cv=5,scoring='accuracy')
rfecv.fit(res_training_nb.drop("Target",axis=1), res_training_nb['Target'].astype(int))
X_rfecv=rfecv.transform(res_training_nb.drop("Target",axis=1))

print("Optimal number of features : %d" % rfecv.n_features_)
print('Optimal number of features :', rfecv.n_features_)
print('Best features :', res_training_nb.drop("Target",axis=1).columns[rfecv.support_])
print('Original features :', res_training_nb.drop("Target",axis=1).columns)
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score \n of number of selected features")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
rfecv.grid_scores_

#train
#creating a list with the selected features
best_nb=list(res_training_nb.drop("Target",axis=1).columns[rfecv.support_])

#Creating a new dataset only with the selected final features
res_training_new_nb = res_training_nb[[*best_nb,"Target"]]

#test
#creating a list with the selected features
res_testing_new_nb = res_testing_nb[[*best_nb,"Target"]]


#### - Only textual features

rfecv_text = RFECV(estimator=clf,cv=5,scoring='accuracy')
rfecv_text.fit(res_training_nb_text.drop("Target",axis=1), res_training_nb_text['Target'].astype(int))
X_rfecv=rfecv_text.transform(res_training_nb_text.drop("Target",axis=1))

print("Optimal number of features : %d" % rfecv_text.n_features_)
print('Optimal number of features :', rfecv_text.n_features_)
print('Best features :', res_training_nb_text.drop("Target",axis=1).columns[rfecv_text.support_])
print('Original features :', res_training_nb_text.drop("Target",axis=1).columns)
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score \n of number of selected features")
plt.plot(range(1, len(rfecv_text.grid_scores_) + 1), rfecv_text.grid_scores_)
plt.show()
rfecv_text.grid_scores_

#train
#creating a list with the selected features
best_nb_text=list(res_training_nb_text.drop("Target",axis=1).columns[rfecv_text.support_])

#Creating a new dataset only with the selected final features
res_training_new_nb_text = res_training_nb_text[[*best_nb_text,"Target"]]

#test
#creating a list with the selected features
res_testing_new_nb_text = res_testing_nb_text[[*best_nb_text,"Target"]]


#### - Only Social Context data

train_set_social["Community ID"]=train_set_social["Community ID"].replace(-1, 4)
rfecv_social = RFECV(estimator=clf,cv=5,scoring='accuracy')
rfecv_social.fit(train_set_social.drop("Target",axis=1), train_set_social["Target"].astype(int))
X_rfecv=rfecv_social.transform(train_set_social.drop("Target",axis=1))

print("Optimal number of features : %d" % rfecv_social.n_features_)
print('Optimal number of features :', rfecv_social.n_features_)
print('Best features :', train_set_social.drop("Target",axis=1).columns[rfecv_social.support_])
print('Original features :', train_set_social.drop("Target",axis=1).columns)
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score \n of number of selected features")
plt.plot(range(1, len(rfecv_social.grid_scores_) + 1), rfecv_social.grid_scores_)
plt.show()
rfecv_social.grid_scores_

#train
#creating a list with the selected features
best_nb_social=list((train_set_social.drop("Target",axis=1).columns[rfecv_social.support_]))

#Creating a new dataset only with the selected final features
train_set_social_new_nb = train_set_social[[*best_nb_social,"Target"]]

#test
#creating a list with the selected features
res_testing_new_nb_social = test_set_social[[*best_nb_social,"Target"]]


##### RELATED TO CHAPTER IN MANUSCRIPT : 3.4.2.2 Test scenarios
#3.10 Modeling

# Machine Learning algorithms

#### Naive Bayes (NB) classifier

# #### Training Naive Bayes (NB) classifier on training data

# - ##### Both textual & Social context features : 

clf_new = MultinomialNB().fit(res_training_new_nb.drop("Target",axis=1), res_training_new_nb['Target'].astype(int))

# - ##### Only textual data

clf_new_text = MultinomialNB().fit(res_training_new_nb_text.drop("Target",axis=1), res_training_new_nb_text['Target'].astype(int))

# - ##### Social context data

clf_new_soc=MultinomialNB().fit(train_set_social.drop("Target",axis=1), train_set_social['Target'].astype(int))


# #### Prediction using both training and test set

# - ##### Both textual & Social context features : 

predicted_nb = clf_new.predict(res_training_new_nb.drop("Target",axis=1))

#print results - train
print(predicted_nb)

predicted_nb_test = clf_new.predict(res_testing_new_nb.drop("Target",axis=1))

#print results - test
print(predicted_nb_test)

#####Learning Curve function
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

#Plotting Learning Curve
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
plot_learning_curve(clf_new, "Learning Curves (Naive Bayes) - Text & Social features", res_training_new_nb.drop("Target",axis=1), res_training_new_nb['Target'].astype(int), cv=cv, n_jobs=4)

plt.show()


# - ##### Only textual data

predicted_nb_text = clf_new_text.predict(res_training_new_nb_text.drop("Target",axis=1))

print(predicted_nb_text)

predicted_nb_text_test = clf_new_text.predict(res_testing_new_nb_text.drop("Target",axis=1))

print(predicted_nb_text_test)

plot_learning_curve(clf_new, "Learning Curves (Naive Bayes) - Text features", res_training_new_nb_text.drop("Target",axis=1), res_training_new_nb_text['Target'].astype(int), cv=cv, n_jobs=4)

plt.show()


# - ##### Social features

predicted_nb_social = clf_new_soc.predict(train_set_social.drop("Target",axis=1))

print(predicted_nb_social)

predicted_nb_social_test = clf_new_soc.predict(test_set_social.drop("Target",axis=1))

print(predicted_nb_social_test)

plot_learning_curve(clf_new, "Learning Curves (Naive Bayes) - Social features", train_set_social.drop("Target",axis=1), train_set_social['Target'].astype(int), cv=cv, n_jobs=4)

plt.show()


# #### Performance

# - ##### Both textual & Social context features : 

# I calculate the performance/accuracy on the training data.
np.mean(predicted_nb == res_training_new_nb["Target"].astype(int))

# I calculate the performance/accuracy on the test data.
np.mean(predicted_nb_test == res_testing_new_nb["Target"].astype(int))

# - Confusion Matrix
print(confusion_matrix(predicted_nb, np.ravel(res_training_new_nb['Target'].astype(int))))
print(confusion_matrix(predicted_nb_test, np.ravel(res_testing_new_nb['Target'].astype(int))))

# - Evaluation metrics table
target_names = ['True News', 'Fake News']
print(classification_report(res_testing_new_nb['Target'], predicted_nb_test,target_names=target_names))

# - ROC AUC
auc_roc = metrics.roc_auc_score(predicted_nb, np.ravel(res_training_new_nb['Target'].astype(int)))
print(auc_roc)

auc_roc_test = metrics.roc_auc_score(predicted_nb_test, np.ravel(res_testing_new_nb['Target'].astype(int)))
print(auc_roc_test)

# - ##### Only textual data :

#evaluation metrics
target_names = ['True News', 'Fake News']
print(classification_report(res_testing_new_nb_text['Target'], predicted_nb_text_test,target_names=target_names))

np.mean(predicted_nb_text == res_training_new_nb_text["Target"].astype(int))
np.mean(predicted_nb_text_test == res_testing_new_nb_text["Target"].astype(int))

#auc score
auc_roc = metrics.roc_auc_score(predicted_nb_text, np.ravel(res_training_new_nb_text['Target'].astype(int)))
print(auc_roc)

auc_roc_test = metrics.roc_auc_score(predicted_nb_text_test, np.ravel(res_testing_new_nb_text['Target'].astype(int)))
print(auc_roc_test)


#ROC CURVE 
fpr, tpr, thresholds = metrics.roc_curve(predicted_nb_text_test, np.ravel(res_testing_new_nb_text['Target'].astype(int)))

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()    

plot_roc_curve(fpr, tpr)


# - ##### Social features

#Evaluation metrics
target_names = ['True News', 'Fake News']
print(classification_report(test_set_social['Target'], predicted_nb_social_test,target_names=target_names))


#### Random Forest classifier

# #### Training Random Forest and calculating its performance

# - ##### Both textual & Social context features : 

#fitting model
clf_rf_new=RandomForestClassifier(n_estimators = 100, random_state=50,n_jobs=-1).fit(res_training_rf.drop('Target',axis=1),res_training_rf['Target'].astype(int))

#RELATED CHAPTER IN MANUSCRIPT: 3.4.2.2 Features Selection
#Selecting features
sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
sel.fit(res_training_rf.drop('Target',axis=1),res_training_rf['Target'].astype(int))
sel.get_support()
selected_feat= res_training_rf.drop('Target',axis=1).columns[(sel.get_support())]
len(selected_feat)

print(selected_feat)

#Learning curve
plot_learning_curve(RandomForestClassifier(n_estimators = 100), "Learning Curves (Random Forest) - Social & Text features", res_training_rf.drop("Target",axis=1), res_training_rf['Target'].astype(int), cv=cv, n_jobs=4)

plt.show()

# - ##### Only textual data

#fitting model
clf_rf_new_text=RandomForestClassifier(n_estimators=100,random_state=50,n_jobs=-1).fit(res_training_rf_text.drop('Target',axis=1),res_training_rf_text['Target'].astype(int))

#RELATED CHAPTER IN MANUSCRIPT: 3.4.2.2 Features Selection
#Selecting features
sel_text = SelectFromModel(RandomForestClassifier(n_estimators = 100))
sel_text.fit(res_training_rf_text.drop('Target',axis=1),res_training_rf_text['Target'].astype(int))
sel_text.get_support()
selected_feat_text= res_training_rf_text.drop('Target',axis=1).columns[(sel_text.get_support())]
len(selected_feat_text)

print(selected_feat_text)

#Learning curve
plot_learning_curve(RandomForestClassifier(n_estimators = 100), "Learning Curves (Random Forest) - Text features", res_training_rf_text.drop("Target",axis=1), res_training_rf_text['Target'].astype(int), cv=cv, n_jobs=4)

plt.show()


# - ##### Only social data

#fitting model
cf_rf_soc=RandomForestClassifier(n_estimators = 100).fit(train_set_social.drop('Target',axis=1),train_set_social['Target'].astype(int))

#RELATED CHAPTER IN MANUSCRIPT: 3.4.2.2 Features Selection
#Selecting features
sel_soc = SelectFromModel(RandomForestClassifier(n_estimators = 100))
sel_soc.fit(train_set_social.drop('Target',axis=1),train_set_social['Target'].astype(int))
sel_soc.get_support()
selected_feat_soc= train_set_social.drop('Target',axis=1).columns[(sel_soc.get_support())]
len(selected_feat_soc)

print(selected_feat_soc)

#Learning curve
plot_learning_curve(RandomForestClassifier(n_estimators = 100), "Learning Curves (Random Forest) - Social features", train_set_social.drop("Target",axis=1), train_set_social['Target'].astype(int), cv=cv, n_jobs=4)

plt.show()


# #### Prediction using both training and test set - Random forest classifier

# - ##### Both textual & Social context features : 

predicted_rf = clf_rf_new.predict(res_training_rf.drop("Target",axis=1))

print(predicted_rf)

predicted_rf_test = clf_rf_new.predict(res_testing_rf.drop("Target",axis=1))

print(predicted_rf_test)

#evaluation metrics
target_names = ['True News', 'Fake News']
print(classification_report(res_test_rf["Target"], predicted_rf_test,target_names=target_names))

np.mean(predicted_rf == res_training_rf["Target"].astype(int))
np.mean(predicted_rf_test == res_test_rf["Target"].astype(int))

# - Confusion Matrix
print(confusion_matrix(predicted_rf, np.ravel(res_training_rf['Target'].astype(int))))
print(confusion_matrix(predicted_rf_test, np.ravel(res_testing_rf['Target'].astype(int))))

# - AUC score
auc_roc_rf = metrics.roc_auc_score(predicted_rf, np.ravel(res_training_rf['Target'].astype(int)))
print(auc_roc_rf)

auc_roc_rf_test = metrics.roc_auc_score(predicted_rf_test, np.ravel(res_testing_rf['Target'].astype(int)))
print(auc_roc_rf_test)

# - ROC Curve
fpr, tpr, thresholds = roc_curve(predicted_rf_test, np.ravel(res_testing_rf['Target'].astype(int)))

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    
plot_roc_curve(fpr, tpr)

# - ##### Only textual data

predicted_rf_text = clf_rf_new_text.predict(res_training_rf_text.drop("Target",axis=1))

print(predicted_rf_text)

predicted_rf_test_text = clf_rf_new_text.predict(res_testing_rf_text.drop("Target",axis=1))

print(predicted_rf_test_text)

#evaluation metrics
target_names = ['True News', 'Fake News']
print(classification_report(res_testing_rf_text["Target"], predicted_rf_test_text,target_names=target_names))

np.mean(predicted_rf_text == res_training_rf_text["Target"].astype(int))
np.mean(predicted_rf_test_text == res_testing_rf_text["Target"].astype(int))

# - AUC score
auc_roc = metrics.roc_auc_score(predicted_rf_text, np.ravel(res_training_rf_text['Target'].astype(int)))
auc_roc

auc_roc_test = metrics.roc_auc_score(predicted_rf_test_text, np.ravel(res_testing_rf_text['Target'].astype(int)))
auc_roc_test

# - ROC curve
fpr, tpr, thresholds = roc_curve(predicted_rf_test_text, np.ravel(res_testing_rf_text['Target'].astype(int)))
plot_roc_curve(fpr, tpr)

# - ##### Social context features :

predicted_rf_test_social = cf_rf_soc.predict(test_set_social.drop("Target",axis=1))

print(predicted_rf_test_social)

#evaluation metrics
target_names = ['True News', 'Fake News']
print(classification_report(test_set_social["Target"], predicted_rf_test_social,target_names=target_names))
