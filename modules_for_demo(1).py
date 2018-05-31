#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 

@author: Bottini Gianfausto @Pi school
"""

import pandas as pd
import gensim
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import glob
from gensim import models
import string
import numpy as np
import json
from stop_words import get_stop_words
import re
from sklearn import preprocessing
from sklearn.cluster import DBSCAN
from scipy.stats import mode
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
#plotly.tools.set_credentials_file(username='faffids', api_key='fk1d3sbNzj81AECxuWlO')
import math
import plotly.offline as offline
offline.init_notebook_mode(connected=True)

pd.set_option('display.max_colwidth', 100)
#matrix is the vector space created during the previous script
#"matrix_user_vector_facounnier_200" is the name of the csv file loaded into pandas dataframe
matrix = pd.read_csv('matrix_user_vector_facounnier_200_more_people.csv',sep='\t',index_col=0)
#it is the word2vec dictionary , the .bin file can be downloaded (as for the following file) or trained with gensim
file_where_dictionary_is_stored = "/home/ubuntu/mynotebooks/frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin"
#french dictionary as KeyedVectors (check gensim library)
fr_dictionary = gensim.models.KeyedVectors.load_word2vec_format(file_where_dictionary_is_stored,binary=True)

dimensions = len(fr_dictionary['ok'])
dimensions_string_1 = str(dimensions-1)


#function input: the matrix vector space of users 
#function output: the same matrix adding a last row , the marketing topic plus its vector


def addwords(matrix,word):
    newmatrix = matrix.copy()
    newmatrix.loc[word] = fr_dictionary[word]
    
    return newmatrix


#function input: the marketing campaign topic in a single french word
#function output: a column sorted by the similarity of users to that topic

def results(topic_fr):
    txt_analysis = addwords(matrix,topic_fr )
    txt_similarity = pd.DataFrame(cosine_similarity(txt_analysis) , columns= txt_analysis.index , index = txt_analysis.index)[topic_fr]
    return txt_similarity.sort_values(ascending=False).iloc[1:]



#function input: 1 column dataframe with ids as index and topic closeness measure as unique feature
#function output: a dataframe completed with number other kind of features taking back jsons informations (total likes , likes per day , followers)


def finaldataframe(df_similarity):
    likes = []
    followers = []
    delta = []
    for index, row in df_similarity.iterrows():
        
        index = str(index)
        #change path with respect to where user jsons are stored
        file = "/home/ubuntu/mynotebooks/dati_utenti/dati_utenti/"+index+".json"
        
        try:
            data = json.load(open(file))
            
            nfollowers = len(data[index]["followers"])
            followers.append(nfollowers)
            delta_max = data[index]["items"]["0"][1]
            delta_min = np.inf
            total_likes = 0
            for post in data[index]["items"]:
                nlikes = data[index]["items"][post][2]
                
                try:
                    total_likes += nlikes
                except:
                    total_likes += 0
                timestamp = data[index]["items"][post][1]
                
                try:
                    if timestamp <= delta_min:
                        delta_min = timestamp
                except:
                    pass
            likes.append(total_likes)
            try:  
                delta.append(delta_max-delta_min)
            except:
                delta.append(None)
        except:
            likes.append(None)
            followers.append(None)
            delta.append(None)

    
    df_similarity['likes'] = likes
    df_similarity['followers'] = followers
    df_similarity['delta'] = delta
    df_similarity['deltadays'] = df_similarity['delta']/86400.0
    df_similarity['likesperday'] = df_similarity['likes'] / df_similarity['deltadays']
    
    return df_similarity



#function input: the previous dataframe having ids as index and closeness measure , likes , followers , likes per day as features and 4 kind of quantiles 
#q1 : topic closeness
#q2 : likes
#q3 : followers
#q4 : likes per day
#function output: skimming process , rows having a value less than the fixed quantile over the entire list are deleted , even if just one of the 4 feature does not pass the exam

#obviously quantiles are not fixed

def select (df_,q1 , q2 , q3 ,q4):
    
    a_ = df_[['final_measure','likes','followers','likesperday']]
    lower_quantile1, upper_quantile1 = a_.final_measure.quantile([q1, .99])
    lower_quantile2, upper_quantile2 = a_.likes.quantile([q2, .99])
    lower_quantile3, upper_quantile3 = a_.followers.quantile([q3, .99])
    lower_quantile4, upper_quantile4 = a_.likesperday.quantile([q4, .99])

    a_ = a_.loc[(a_.final_measure > lower_quantile1)& (a_.likes > lower_quantile2) & (a_.followers > lower_quantile3) & (a_.likesperday > lower_quantile4)]
    
    return a_


#function input: vector space matrix (it is needed to have the users 200 dimensions vectors) , the dataframe of selected users 
#function output: a dataframe with standard features plus the X and Y coming from the projection from 200 dimensions to 2

def pcaxy(dfpca,df_plot):
    pca = PCA(n_components=2)
    pca.fit(dfpca)
    PCA(copy=True, n_components=2, whiten=False)
    X = pd.DataFrame(pca.transform(dfpca) ,index = dfpca.index , columns = ['x','y'])
    users = df_plot.index
    pd.concat([df_plot,X.loc[users]], axis=1)
    

    return pd.concat([df_plot,X.loc[users]], axis=1)



#function input: dataframe with features like : URL , topic closeness , lokes , followers , likes per day , x , y
#function output: plotly interactive map

def finalplot(df_plot):

    hover_text = []
    bubble_size = []
    
    for index, row in df_plot.iterrows():

        hover_text.append(('instagram URL: {url}<br>'+
                           'instagram ID: {id}<br>'+
                          'final measure: {measure}<br>'+
                          'likes: {likes}<br>'+
                          'followers: {flw}<br>'+
                          'likesperday: {lpd}<br>').format(url = row['url'],
                                                id=index,
                                                measure=row['final_measure'],
                                                likes=row['likes'],
                                                flw=row['followers'],
                                                lpd=row['likesperday']
                                                ))
        bubble_size.append(math.sqrt(row['followers']*row['likes']*row['likesperday'])/10000)

    df_plot['text'] = hover_text
    df_plot['size'] = bubble_size

    data = go.Scatter(
        x = df_plot['x'],
        y = df_plot['y'],
        text = df_plot['text'],
        mode = 'markers',
        marker =  {

                 "color" :  df_plot['final_measure'],
                "size" : df_plot['size'],
                "showscale" :  True
        }



    )
    

    data = ([data])
    fig = go.Figure(data=data)

    
    offline.iplot(fig)
    return

    
def make_clickable(val):
    return '<a href="{}">{}</a>'.format(val,val)



#function input: any dataframe having IDs as index
#function output: a dataframe with users pictaram website (pictarame is instagram with ID instead of screennames)
def addurl(dataframe):
    dataframe['url'] = f'http://instagram.com/web/friendships/'+dataframe.index.astype(str) + '/follow/'
    return dataframe


import seaborn as sns
cm = sns.light_palette("green", as_cmap=True)

def beautifuldf(df):
    df = df.style.background_gradient(cmap=cm)\
        .format({'url': make_clickable})\
        .set_properties(**{'text-align': 'left'})
    return df

#function input: marketing campaign topic in a single word
#function output: final map plus final dataframe of candidates
#you can increase or decrease quantiles with respect to the search purposes
def launch_campaign_about(word):
    M = addwords(matrix , word)
    
    df_similarity = pd.DataFrame(results(topic_fr=word))
    a = finaldataframe(df_similarity)
    a.columns.values[0] = 'final_measure'
    q = select(a , 0.9, 0.3 , 0.3 ,0.3 ).sort_values('final_measure' , ascending=False).head(100).round({'final_measure': 2, 'likesperday': 0})
    finalplot(addurl(pcaxy(M.loc[:, '1':dimensions_string_1],q)))
    return beautifuldf(addurl(q).head(10))