---
layout: page-fullwidth
show_meta: true
title: "Kaggle Renthop Challenge"
teaser: "Prediction popularity of rental listings.Exploratory Data Analysis, Feature Engineering using Geo-spatial Data, followed by build an Classification model using XGBoost"
date: "2017-04-23"
tags:
  - Data Analysis
  - Machine Learning
  - Kaggle
  - Geocoding
  - Exploratory Data Analysis.
category:
  - work
header: no
permalink: "/work/kaggle-challenge-renthop.html"
---

## Problem Description:
In this competition, you will predict how popular an apartment rental listing is based on the listing content like text description, photos, number of bedrooms, price, etc. The data comes from renthop.com, an apartment listing website. These apartments are located in New York City. The target variable, __interest_level__, is defined by the number of inquiries a listing has in the duration that the listing was live on the site. 

## Data fields

bathrooms: number of bathrooms
bedrooms: number of bathrooms
building_id
created
description
display_address
features: a list of features about this apartment
latitude
listing_id
longitude
manager_id
photos: a list of photo links. You are welcome to download the pictures yourselves from renthop's site, but they are the same as imgs.zip. 
price: in USD
street_address
interest_level: this is the target variable. It has 3 categories: 'high', 'medium', 'low'

# Our Approach
## Exploratory Data Analysis
### Importing python libraries and Loading Data
```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
% matplotlib inline
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
sns.set(font_scale=1)

import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools


from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
init_notebook_mode(connected=True)

train_df = pd.read_json("../input/two-sigma-connect-rental-listing-inquiries/train.json")
test_df = pd.read_json("../input/two-sigma-connect-rental-listing-inquiries/test.json")
```
### The target variable
```python
sns.countplot(train_df.interest_level, order=['low', 'medium', 'high']);
plt.xlabel('Interest Level');
plt.ylabel('Number of occurrences');
```
![target_variable](https://github.com/akshaykumarvikram/Kaggle-Renthop-challenge/blob/master/renthop/image1.png)

### Bathrooms and Bedrooms
```python
fig = plt.figure(figsize=(12,12))
### Number of occurrences
sns.countplot(train_df.bathrooms, ax = plt.subplot(221));
plt.xlabel('Number of Bathrooms');
plt.ylabel('Number of occurrences');
### Average number of Bathrooms per Interest Level
sns.barplot(x='interest_level', y='bathrooms', data=train_df, order=['low', 'medium', 'high'],
            ax = plt.subplot(222));
plt.xlabel('Interest Level');
plt.ylabel('Average Number of Bathrooms');
### Average interest for every number of bathrooms
sns.pointplot(x="bathrooms", y="interest", data=train_df, ax = plt.subplot(212));
plt.xlabel('Number of Bathrooms');
plt.ylabel('Average Interest');
```
![image2](https://github.com/akshaykumarvikram/Kaggle-Renthop-challenge/blob/master/renthop/image2.png)

```python
### Bedrooms graphs
fig = plt.figure(figsize=(12,12))
### Number of occurrences
sns.countplot(train_df.bedrooms, ax = plt.subplot(221));
plt.xlabel('Number of Bedrooms');
plt.ylabel('Number of occurrences');
### Average number of Bedrooms per Interest Level
sns.barplot(x='interest_level', y='bedrooms', data=train_df, order=['low', 'medium', 'high'],
            ax = plt.subplot(222));
plt.xlabel('Interest Level');
plt.ylabel('Average Number of Bedrooms');
### Average interest for every number of bedrooms
sns.pointplot(x="bedrooms", y="interest", data=train_df, ax = plt.subplot(212));
plt.xlabel('Number of Bedrooms');
plt.ylabel('Average Interest');
```
![image5](https://github.com/akshaykumarvikram/Kaggle-Renthop-challenge/blob/master/renthop/image5.png)

### Interest levels on different days of the week
```python
### Iterest per Day of Week
fig = plt.figure(figsize=(12,6))
ax = sns.countplot(x="day_of_week", hue="interest_level",
                   hue_order=['low', 'medium', 'high'], data=train_df,
                   order=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']);
plt.xlabel('Day of Week');
plt.ylabel('Number of occurrences');

### Adding percents over bars
height = [p.get_height() for p in ax.patches]
ncol = int(len(height)/3)
total = [height[i] + height[i + ncol] + height[i + 2*ncol] for i in range(ncol)] * 3
for i, p in enumerate(ax.patches):    
    ax.text(p.get_x()+p.get_width()/2,
            height[i] + 50,
            '{:1.0%}'.format(height[i]/total[i]),
            ha="center") 
```
![image8](https://github.com/akshaykumarvikram/Kaggle-Renthop-challenge/blob/master/renthop/image8.png)

### Exploring the Price
```python
fig = plt.figure(figsize=(12,12))
sns.distplot(train_data.price[train_data.price<=train_data.price.quantile(0.99)], ax=plt.subplot(211));
plt.xlabel('Price');
plt.ylabel('Density');
### Average Price per Interest Level
sns.barplot(x="interest_level", y="price", order=['low', 'medium', 'high'],
            data=train_data, ax=plt.subplot(223));
plt.xlabel('Interest Level');
plt.ylabel('Price');
### Violinplot of price for every Interest Level
sns.violinplot(x="interest_level", y="price", order=['low', 'medium', 'high'],
               data=train_data[train_data.price<=train_data.price.quantile(0.99)],
               ax=plt.subplot(224));
plt.xlabel('Interest Level');
plt.ylabel('Price');
```
![image9](https://github.com/akshaykumarvikram/Kaggle-Renthop-challenge/blob/master/renthop/image9.png)

### Word Clouds 
### Features
```python
from wordcloud import WordCloud

text = ''
text_da = ''
text_desc = ''
for ind, row in train_df.iterrows():
    for feature in row['features']:
        text = " ".join([text, "_".join(feature.strip().split(" "))])
    text_da = " ".join([text_da,"_".join(row['display_address'].strip().split(" "))])
    #text_desc = " ".join([text_desc, row['description']])
text = text.strip()
text_da = text_da.strip()
text_desc = text_desc.strip()

plt.figure(figsize=(12,6))
wordcloud = WordCloud(background_color='white', width=600, height=300, max_font_size=50, max_words=40).generate(text)
wordcloud.recolor(random_state=0)
plt.imshow(wordcloud)
plt.title("Wordcloud for features", fontsize=30)
plt.axis("off")
plt.show()
```
![image12](https://github.com/akshaykumarvikram/Kaggle-Renthop-challenge/blob/master/renthop/image12.png)
```python
# wordcloud for display address
plt.figure(figsize=(12,6))
wordcloud = WordCloud(background_color='white', width=600, height=300, max_font_size=50, max_words=40).generate(text_da)
wordcloud.recolor(random_state=0)
plt.imshow(wordcloud)
plt.title("Wordcloud for Display Address", fontsize=30)
plt.axis("off")
plt.show()
```
![image13](https://github.com/akshaykumarvikram/Kaggle-Renthop-challenge/blob/master/renthop/image13.png)

## Exploring the geographic location of all the listings
#### (NOTE: We have used R for mapping the locations of all the listings)
### Loading necessary Libraries
```{r}
library(tigris)
library(dplyr)
library(leaflet)
library(sp)
library(ggmap)
library(maptools)
library(broom)
library(httr)
library(rgdal)
loading the li
```
### Importing New york city neighborhood data

```{r}
r <- GET('http://data.beta.nyc//dataset/0ff93d2d-90ba-457c-9f7e-39e47bf2ac5f/resource/35dd04fb-81b3-479b-a074-a27a37888ce7/download/d085e2f8d0b54d4590b1e7d1f35594c1pediacitiesnycneighborhoods.geojson')
nyc_neighborhoods <- readOGR(content(r,'text'), 'OGRGeoJSON', verbose = F)
nyc_neighborhoods_df <- tidy(nyc_neighborhoods)
```
### Plotting it neighborhood data
```{r}
nyc_neighborhoods_df <- tidy(nyc_neighborhoods)
nyc_map <- get_map(location = c(lon = -74.00, lat = 40.71), maptype = "terrain", zoom = 11)
suppressMessages(ggmap(nyc_map)) + 
  geom_polygon(data=nyc_neighborhoods_df, aes(x=long, y=lat, group=group), color="blue", fill=NA)
```
![image14](https://github.com/akshaykumarvikram/Kaggle-Renthop-challenge/blob/master/renthop/image14.png)

### Finding Neighborhoods of all the locations
```{r}
lats <- train$latitude
lngs <- train$longitude
points <- data.frame(lat=as.numeric(lats), lng=as.numeric(lngs))
points_spdf <- points
coordinates(points_spdf) <- ~lng + lat
proj4string(points_spdf) <- proj4string(nyc_neighborhoods)
matches <- over(points_spdf, nyc_neighborhoods)
points <- cbind(points, matches)
```
### Plotting the distirbution of the listings
```{r}
points <- train[c('lat','lng','neighborhood','boroughCode','borough','X.id')]
points_by_neighborhood <- points %>%
  group_by(neighborhood) %>%
  summarize(num_points=n())

map_data <- geo_join(nyc_neighborhoods, points_by_neighborhood, "neighborhood", "neighborhood")
pal <- colorNumeric(palette = "RdBu", domain = range(map_data@data$num_points, na.rm=T))

plot_data <- tidy(nyc_neighborhoods, region="neighborhood") %>%
  left_join(., points_by_neighborhood, by=c("id"="neighborhood")) %>%
  filter(!is.na(num_points))
  nyc_map <- get_map(location = c(lon = -74.00, lat = 40.71), maptype = "terrain", zoom = 10)
  
ggmap(nyc_map) + 
  geom_polygon(data=plot_data, aes(x=long, y=lat, group=group, fill=num_points),colour='black', alpha=0.75)
```
![image15](https://github.com/akshaykumarvikram/Kaggle-Renthop-challenge/blob/master/renthop/image15.png)

### Exploting transportation options for each listing
#### Plotting the subway data of NYC


```{r}
library(geosphere)
subway <- GET('https://data.cityofnewyork.us/api/views/kk4q-3rt2/rows.csv?accessType=DOWNLOAD')
subway_data <- read.csv(subway)
train$latitude <- as.numeric(train$latitude)
train$longitude <- as.numeric(train$longitude)
test$latitude <- as.numeric(test$latitude)
test$longitude <- as.numeric(test$longitude)
subway_data <- read.csv('subway.csv')
nyc_map <- get_map(location = c(lon = -74.00, lat = 40.71), maptype = "terrain", zoom = 11)
ggmap(nyc_map) +
  geom_point(data = subway_data, aes(x = longitude, y = latitude, fill = "red", alpha = 1), size = 2,shape = 21) +
  guides(fill=FALSE, alpha=FALSE, size=FALSE)
```
![image16](https://github.com/akshaykumarvikram/Kaggle-Renthop-challenge/blob/master/renthop/image16.PNG)

### Creating a database of average rent in each neighborhood
```{r}
grp_cols <- c('neighborhood','bedrooms')

# Convert character vector to list of symbols
dots <- lapply(grp_cols, as.symbol)

# Perform frequency counts
area_database <- all_data %>%
    group_by_(.dots=dots) %>%
    summarise(price = median(price),n=n())
write.csv(area_database,file = 'area_database.csv')    
```

## Feature Engineering
#### (NOTE: Back to python)
### Importing the necessary libraries
```python
import pandas as pd
import os
import sys
import operator
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
import xgboost as xgb
import random
from sklearn import model_selection, preprocessing, ensemble
from sklearn.preprocessing import Imputer
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import ExtraTreesClassifier
```
### Converting 'created' column into a datatime object
```python
train_df["created"] = pd.to_datetime(train_df["created"])
test_df["created"] = pd.to_datetime(test_df["created"])
```
### Extracting Additional features from datetime object
```python
train_df["created_year"] = train_df["created"].dt.year
test_df["created_year"] = test_df["created"].dt.year

train_df["created_month"] = train_df["created"].dt.month
test_df["created_month"] = test_df["created"].dt.month

train_df["created_day"] = train_df["created"].dt.day
test_df["created_day"] = test_df["created"].dt.day
```
### Calculating the average price for similar house in the neighborhood (similar number of bedrooms)
```python
area_database = pd.read_csv('area_database.csv')
def get_neigborhood_avg(row):
    return float(area_database.loc[(area_database.neighborhood==row['neighborhood']) & (area_database.bedrooms==row['bedrooms'])].price)
train_df['neighborhood_avg'] = train_df.apply(get_neigborhood_avg, axis=1)
test_df['neighborhood_avg'] = test_df.apply(get_neigborhood_avg, axis=1)
```
### calculating the price difference between the listing and market rate
```python
train_df['price_difference'] = train_df['price'] - train_df['neighborhood_avg']
test_df['price_difference'] = test_df['price'] - test_df['neighborhood_avg']
train_df['relative_price'] = train_df['price_difference']/train_df['neighborhood_avg']
test_df['relative_price'] = test_df['price_difference']/test_df['neighborhood_avg']
```
### few additional features
```python
# count of photos #
train_df["num_photos"] = train_df["photos"].apply(len)
test_df["num_photos"] = test_df["photos"].apply(len)

# count of "features" #
train_df["num_features"] = train_df["features"].apply(len)
test_df["num_features"] = test_df["features"].apply(len)

# count of words present in description column #
train_df["num_description_words"] = train_df["description"].apply(lambda x: len(x.split(" ")))
test_df["num_description_words"] = test_df["description"].apply(lambda x: len(x.split(" ")))
```
### Label encoding the categorical features
```python
categorical = ["display_address", "manager_id", "building_id", "street_address",'neighborhood']
for f in categorical:
        if train_df[f].dtype=='object':
            #print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_df[f].values) + list(test_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))
            test_df[f] = lbl.transform(list(test_df[f].values))
```
### Dealing with 'Features' column
#### Features column has a list representing the features of the listing, so we combine all the strings to-gether and apply a count vectorizer on top of it.
```python
train_df['features'].fillna("",inplace=True)
test_df['features'].fillna("",inplace=True)
train_df['features'] = train_df["features"].apply(lambda x: " ".join(["_".join(x.split(" "))]))
test_df['features'] = test_df["features"].apply(lambda x: " ".join(["_".join(x.split(" "))]))
tfidf = CountVectorizer(stop_words='english', max_features=200)
tfidf.fit(list(train_df['features']) + list(test_df['features']))
tr_sparse = tfidf.transform(train_df["features"])
te_sparse = tfidf.transform(test_df["features"])
```
### Dealing with the missing values
```python
fill_NaN = Imputer(missing_values=np.nan, strategy='mean', axis=1)
train_imputed = pd.DataFrame(fill_NaN.fit_transform(train_df[features_to_use]))
train_imputed.columns = train_df[features_to_use].columns
train_imputed.index = train_df.index

test_imputed = pd.DataFrame(fill_NaN.fit_transform(test_df[features_to_use]))
test_imputed.columns = test_df[features_to_use].columns
test_imputed.index = test_df.index
```
### Building the final dataset by stacking densely and sparsely populated features into one dataset
```python
train_X = sparse.hstack([train_imputed, tr_sparse]).tocsr()
test_X = sparse.hstack([test_imputed, te_sparse]).tocsr()
```
### converting the target variable
```python
target_num_map = {'high':0, 'medium':1, 'low':2}
train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))
print(train_X.shape, test_X.shape)
```
## Machine Learning
### Building an XGBOOST model
```python
def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=1000):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.1
    param['max_depth'] = 6
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model
```
### Cross validation and training the model
```python
cv_scores = []
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2016)
for dev_index, val_index in kf.split(range(train_X.shape[0])):
        dev_X, val_X = train_X[dev_index,:], train_X[val_index,:]
        dev_y, val_y = train_y[dev_index], train_y[val_index]
        preds, model = runXGB(dev_X, dev_y, val_X, val_y)
        cv_scores.append(log_loss(val_y, preds))
        print(cv_scores)
        break
```
Model stopped after 854 iterations with train-mlogloss:0.37921 test-mlogloss:0.522394
### Predicting on the test set
```python
preds, model = runXGB(train_X, train_y, test_X, num_rounds=400)
out_df = pd.DataFrame(preds)
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = test_df.listing_id.values
out_df.to_csv("predictions.csv", index=False)
```
##### Note: XBG was inspired from a notebook on kaggle by SRK.
Project Link: [Click here to view full notebook on github](https://github.com/akshaykumarvikram/Kaggle-Renthop-challenge)
