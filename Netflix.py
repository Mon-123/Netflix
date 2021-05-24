#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


os.chdir('C:\\Users\\Monali\\OneDrive\\Desktop\\Case Study-DS\\Netflix')


# In[3]:


os.getcwd()


# Import Dataset

# In[4]:


Data = pd.read_csv("netflix_titles.csv")


# In[5]:


Data.head()


# In[6]:


Data.shape


# In[7]:


Data.info()


# In[9]:


Data.nunique()


# #Handling Null values

# In[10]:


Data.isnull().sum()


# In[11]:


Data.isnull().sum().sum()


# In[16]:


sns.heatmap(Data.isnull(), cbar=False)
plt.title('Null Values Heatmap')
plt.show()


# # Above in the heatmap and table, 
# we can see that there are quite a few null values
# in the dataset. There are a total of 3,036 
# null values across the entire dataset with 1,969 missing points under 
# 'director', 570 under 'cast', 476 under 'country', 11 under 'date_added',
# and 10 under 'rating'. We will have to handle all null data points before we 
# can dive into EDA and modeling.

# In[19]:


Data['director'].fillna('No_Director',inplace = True)
Data['cast'].fillna('cast',inplace = True)
Data['country'].fillna('no_country',inplace = True)


# In[20]:


Data.head()


# In[24]:


Data.dropna(subset = ['date_added','rating'],inplace=True)


# In[25]:


Data.isnull().sum()


# In[27]:


Data.isnull().any()


# For null values, the easiest way to get rid of them would be to delete the rows with the missing data. However, this wouldn't be beneficial to our EDA since there is loss of information. Since 'director', 'cast', and 'country' contain the majority of null values, I will choose to treat each missing value as unavailable. The other two labels 'date_added' and 'rating' contains an insignificant portion of the data so I will drop them from the dataset. After, we can see that there are no more null values in the dataset.

# # Splitting the dataset
# Since the dataset can either contain movies or shows, it'd be nice to have datasets for both so we can take a deep dive into just Netflix movies or Netflix TV shows so we will create two new datasets. One for movies and the other one for shows.

# In[36]:


Data_Movies = Data[Data['type']=='Movie'].copy()


# In[33]:


Data_Movies.head()


# In[30]:


Data_Movies.shape


# In[31]:


Data_Movies.info()


# In[34]:


Data_TVShow = Data[Data['type']=='TV Show'].copy()


# In[35]:


Data_TVShow


# In[38]:


Data_Movies


# # DATA PREPARATION
# In the duration column, there appears to be a discrepancy between movies and shows. Movies are based on the duration of the movie and shows are based on the number of seasons. To make EDA easier, I will convert the values in these columns into integers for both the movies and shows datasets.

# In[47]:


Data_Movies.duration = Data_Movies.duration.str.replace(' min','').astype(int)
Data_TVShow.rename(columns={'duration':'seasons'}, inplace=True)
Data_TVShow.replace({'seasons':{'1 Season':'1 Seasons'}}, inplace=True)
Data_TVShow.seasons = Data_TVShows.seasons.str.replace('seasons','').astype(int)


# In[48]:


plt.figure(figsize=(7,5))
g = sns.countplot(Data.type, palette="pastel");
plt.title("Count of Movies and TV Shows")
plt.xlabel("Type (Movie/TV Show)")
plt.ylabel("Total Count")
plt.show()


# In[51]:


data:image/png;base64,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plt.figure(figsize=(12,6))
plt.title("% of Netflix Titles that are either Movies or TV Shows")
g = plt.pie(Data.type.value_counts(), explode=(0.025,0.025), labels=Data.type.value_counts().index, colors=['skyblue','navajowhite'],autopct='%1.1f%%', startangle=180);
plt.legend()
plt.show()


# So there are roughly 4,000+ movies and almost 2,000 shows with movies being the majority. This makes sense since shows are always an ongoing thing and have episodes. If we were to do a headcount of TV show episodes vs. movies, I am sure that TV shows would come out as the majority. However, in terms of title, there are far more movie titles (68.5%) than TV show titles (31.5%).

# In[53]:


order =  ['G', 'TV-Y', 'TV-G', 'PG', 'TV-Y7', 'TV-Y7-FV', 'TV-PG', 'PG-13', 'TV-14', 'R', 'NC-17', 'TV-MA']
plt.figure(figsize=(15,7))
g = sns.countplot(Data.rating, hue=Data.type, order=order, palette="pastel");
plt.title("Ratings for Movies & TV Shows")
plt.xlabel("Rating")
plt.ylabel("Total Count")
plt.show()


# Countries with most content available

# In[56]:


Filtered_Countries = Data.set_index('title').country.str.split(', ',expand = True).stack().reset_index(level = 1,drop = True);
Filtered_Countries = Filtered_Countries[Filtered_Countries != 'Country Unavailable']

plt.figure(figsize=(7,9))
g = sns.countplot(y = Filtered_Countries, order=Filtered_Countries.value_counts().index[:20])
plt.title('Top 20 Countries on Netflix')
plt.xlabel('Titles')
plt.ylabel('Country')
plt.show()


# Questions and Answers

# In[59]:


filtered_directors = Data[Data.director != 'No Director'].set_index('title').director.str.split(', ', expand=True).stack().reset_index(level=1, drop=True)
sns.countplot(y = filtered_directors, order=filtered_directors.value_counts().index[:10], palette='mako')
plt.show()


# Top ten actors cast

# In[64]:


filtered_cast = Data[Data.cast != 'No Cast'].set_index('title').cast.str.split(', ', expand=True).stack().reset_index(level=1, drop=True)
sns.countplot(y = filtered_cast, order=filtered_cast.value_counts().index[:10], palette='rocket')
plt.show()


# In[ ]:




