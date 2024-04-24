#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[170]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
#colors = ['purple']

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')


# In[171]:


data = pd.read_csv(r"C:\Users\Hp\Desktop\projet-data science\codsoft data science\IMDb Movies India.csv", encoding="latin-1") 
data


# # Understanding of data

# In[172]:


data.head()


# In[173]:


data.info()


# # Data Cleaning

# In[174]:


# Vérification des valeurs null
data.isna().sum()


# In[175]:


# Localisation des lignes avec des valeurs manquantes dans les colonnes de 1 à 9
nulls = data[data.iloc[:, 1:9].isna().all(axis=1)]
nulls.head()


# In[176]:


#Vérifier s'il y a des fautes de frappe

for column in data.select_dtypes(include = "object"):
    print(f"Name of Column: {column}")
    print(data[column].unique())
    print('\n', '-'*60, '\n')


# In[177]:


# Gestion des valeurs nulles
data.dropna(subset=['Name', 'Year', 'Duration', 'Rating', 'Votes', 'Director', 'Actor 1', 'Actor 2', 'Actor 3'], inplace=True)
#Extraire uniquement la partie textuelle de la colonne Nom
data['Name'] = data['Name'].str.extract('([A-Za-z\s\'\-]+)')
# Remplacer les crochets de la colonne des années comme observé ci-dessus
data['Year'] = data['Year'].str.replace(r'[()]', '', regex=True).astype(int)
data.head()


# In[178]:


# Convert 'Duration' to numeric and replacing the min, while keeping only numerical part

data['Duration'] = pd.to_numeric(data['Duration'].str.replace(r' min', '', regex=True), errors='coerce')


# In[179]:


# Splitting the genre by , to keep only unique genres and replacing the null values with mode

data['Genre'] = data['Genre'].str.split(',')

data = data.explode('Genre')

data['Genre'].fillna(data['Genre'].mode()[0], inplace=True)
data.head()


# In[180]:


# Convert 'Votes' to numeric and replace the , to keep only numerical part
data['Votes'] = pd.to_numeric(data['Votes'].str.replace(',', ''), errors='coerce')
data.head()


# In[181]:


#checking duplicate values by Name and Year

duplicate = data.groupby(['Name', 'Year']).filter(lambda x: len(x) > 1)
duplicate.head(5)


# In[182]:


# Dropping the duplicated values by Name
data = data.drop_duplicates(subset=['Name'], keep=False)
data.head()


# # Exploratory Data Analysis

# In[183]:


data.describe()


# In[184]:


data.describe(include = 'O')


# In[185]:


# Find the row with the highest number of votes

max_votes_row = data[data['Votes'] == data['Votes'].max()]
# Get the name of the movie with the highest votes

movie_highest_votes = max_votes_row['Name'].values[0]
# Find the number of votes for the movie with the highest votes

votes_highest_votes = max_votes_row['Votes'].values[0]

print("Movie with the highest votes:", movie_highest_votes)

print("Number of votes for the movie with the highest votes:", votes_highest_votes)



# In[186]:


# Find the row with the lowest number of votes

min_votes_row = data[data['Votes'] == data['Votes'].min()]



# Get the name of the movie with the lowest votes

movie_lowest_votes = min_votes_row['Name'].values[0]



# Find the number of votes for the movie with the lowest votes

votes_lowest_votes = min_votes_row['Votes'].values[0]



print("Movie with the highest votes:", movie_lowest_votes)

print("Number of votes for the movie with the highest votes:", votes_lowest_votes)


# In[187]:


# Find the row with the highest rating
max_rating_row = data[data['Rating'] == data['Rating'].max()]
movie_highest_rating = max_rating_row['Name'].values[0]
votes_highest_rating = max_rating_row['Votes'].values[0]

print("Movie with the highest rating:", movie_highest_rating)
print("Number of votes for the movie with the highest rating:", votes_highest_rating)
print('\n', '='*100, '\n')

# Find the row with the lowest rating
min_rating_row = data[data['Rating'] == data['Rating'].min()]
movie_lowest_rating = min_rating_row['Name'].values[0]
votes_lowest_rating = min_rating_row['Votes'].values[0]

print("Movie with the highest rating:", movie_lowest_rating)
print("Number of votes for the movie with the highest rating:", votes_lowest_rating)


# In[188]:


# Group the dataset by the 'Director' column and count the number of movies each director has directed
director_counts = data['Director'].value_counts()

# Find the director with the highest number of movies directed
most_prolific_director = director_counts.idxmax()
num_movies_directed = director_counts.max()

print("Director with the most movies directed:", most_prolific_director)
print("Number of movies directed by", most_prolific_director, ":", num_movies_directed)
print('\n', '='*100, '\n')

# Group the dataset by the 'Director' column and count the number of movies each director has directed
director_counts = data['Director'].value_counts()

# Find the director with the lowest number of movies directed
least_prolific_director = director_counts.idxmin()
num_movies_directed = director_counts.min()

print("Director with the most movies directed:", least_prolific_director)
print("Number of movies directed by", most_prolific_director, ":", num_movies_directed)


# In[243]:


ig_year = px.histogram(data, x = 'Year', histnorm='probability density', nbins = 30, color_discrete_sequence = colors)
fig_year.update_traces(selector=dict(type='histogram'))
fig_year.update_layout(title='Distribution of Year', title_x=0.5, title_pad=dict(t=20), title_font=dict(size=20), xaxis_title='Year', yaxis_title='Probability Density', xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), bargap=0.02, plot_bgcolor = 'white')
fig_year.show()


# In[222]:


import seaborn as sns
import matplotlib.pyplot as plt

color = 'brown'  

sns.distplot(data['Duration'], color=color)  

plt.title('Distribution of Duration')

# Display the plot
plt.show()


# In[212]:


colors = ['orange']
sns.distplot(data['Rating'])
plt.title('Distribution of rating')
plt.show()


# In[241]:


colors = ['#00CC96']
fig_votes = px.box(data, x = 'Votes', color_discrete_sequence = colors)
fig_votes.update_layout(title='Distribution of Votes', title_x=0.5, title_pad=dict(t=20), title_font=dict(size=20), xaxis_title='Votes', yaxis_title='Probability Density', xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), plot_bgcolor = 'white')
fig_votes.show()



# In[230]:


year_avg_rating = data.groupby('Year')['Rating'].mean().reset_index()

top_5_years = year_avg_rating.nlargest(10, 'Rating')
fig = px.bar(top_5_years, x='Year', y='Rating', title='Top 10 Years by Average Rating', color = "Rating", color_continuous_scale = "darkmint")
fig.update_xaxes(type='category')  
fig.update_layout(xaxis_title='Year', yaxis_title='Average Rating', plot_bgcolor = 'white')
fig.show()


# In[235]:


# Group data by Year and calculate the average rating

average_rating_by_year = data.groupby('Year')['Rating'].mean().reset_index()

# Create the line plot with Plotly Express

fig = px.line(average_rating_by_year, x='Year', y='Rating', color_discrete_sequence=['purple'])

fig.update_layout(title='Are there any trends in ratings across year?', title_x=0.5, title_pad=dict(t=20), title_font=dict(size=20), xaxis_title='Year', yaxis_title='Rating', xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), plot_bgcolor = 'white')


# In[236]:


# Group data by Year and calculate the average rating
average_rating_by_year = data.groupby('Year')['Votes'].mean().reset_index()

# Create the line plot with Plotly Express
fig = px.line(average_rating_by_year, x='Year', y='Votes', color_discrete_sequence=['purple'])
fig.update_layout(title='Are there any trends in votes across year?', title_x=0.5, title_pad=dict(t=20), title_font=dict(size=20), xaxis_title='Year', yaxis_title='Votes', xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), plot_bgcolor = 'white')
fig.show()


# In[238]:


# Group data by Year and calculate the average rating
average_rating_by_year = data.groupby(['Year', 'Genre'])['Rating'].mean().reset_index()

# Get the top 3 genres
top_3_genres = data['Genre'].value_counts().head(3).index

# Filter the data to include only the top 3 genres
average_rating_by_year = average_rating_by_year[average_rating_by_year['Genre'].isin(top_3_genres)]

# Create the line plot with Plotly Express
fig = px.line(average_rating_by_year, x='Year', y='Rating', color = "Genre", color_discrete_sequence=['purple', '#0B1F26', '#00CC96'])
# Customize the layout
fig.update_layout(title='Average Rating by Year for Top 3 Genres', xaxis_title='Year', yaxis_title='Average Rating', xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), plot_bgcolor='white')

# Show the plot
fig.show()


# In[244]:


fig_dur_rat = px.scatter(data, x = 'Duration', y = 'Rating', trendline='ols', color = "Rating", color_continuous_scale = "darkmint")
fig_dur_rat.update_layout(title='Does length of movie have any impact on rating?', title_x=0.5, title_pad=dict(t=20), title_font=dict(size=20), xaxis_title='Duration of Movie in Minutes', yaxis_title='Rating of a movie', xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), plot_bgcolor = 'white')
fig_dur_rat.show()


# In[245]:


fig_dur_votes = px.scatter(data, x = 'Duration', y = 'Votes', trendline='ols', color = "Votes", color_continuous_scale = "darkmint")
fig_dur_votes.update_layout(title='Does length of movie have any impact on Votes?', title_x=0.5, title_pad=dict(t=20), title_font=dict(size=20), xaxis_title='Duration of Movie in Minutes', yaxis_title='Votes of a movie', xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), plot_bgcolor = 'white')
fig_dur_votes.show()


# In[246]:


fig_rat_votes = px.scatter(data, x = 'Rating', y = 'Votes', trendline='ols', color = "Votes", color_continuous_scale = "darkmint")
fig_rat_votes.update_layout(title='Does Ratings of movie have any impact on Votes?', title_x=0.5, title_pad=dict(t=20), title_font=dict(size=20), xaxis_title='Ratings of Movies', yaxis_title='Votes of movies', xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), plot_bgcolor = 'white')
fig_rat_votes.show()


# # Data Preprocessing 

# In[247]:


# Dropping non essential columns
data.drop('Name', axis = 1, inplace = True)
data.head()


# In[248]:


#Feature Engineering
# Grouping the columns with their average rating and then creating a new feature

genre_mean_rating = data.groupby('Genre')['Rating'].transform('mean')
data['Genre_mean_rating'] = genre_mean_rating

director_mean_rating = data.groupby('Director')['Rating'].transform('mean')
data['Director_encoded'] = director_mean_rating

actor1_mean_rating = data.groupby('Actor 1')['Rating'].transform('mean')
data['Actor1_encoded'] = actor1_mean_rating

actor2_mean_rating = data.groupby('Actor 2')['Rating'].transform('mean')
data['Actor2_encoded'] = actor2_mean_rating

actor3_mean_rating = data.groupby('Actor 3')['Rating'].transform('mean')
data['Actor3_encoded'] = actor3_mean_rating


# In[249]:


#Splitting into training and testing
# Keeping the predictor and target variable

X = data[[ 'Year', 'Votes', 'Duration', 'Genre_mean_rating','Director_encoded','Actor1_encoded', 'Actor2_encoded', 'Actor3_encoded']]
y = data['Rating']


# In[250]:


# Splitting the dataset into training and testing parts

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# # Model Building

# In[251]:


# Building 2 machine learning models and training them

lr = LinearRegression()
lr.fit(X_train,y_train)
lr_pred = lr.predict(X_test)

rf = RandomForestRegressor()
rf.fit(X_train,y_train)
rf_pred = rf.predict(X_test)


# # Model Evaluation

# In[252]:


# Evaluating the performance of trained algos

print('The performance evaluation of Logistic Regression is below: ', '\n')
print('Mean squared error: ',mean_squared_error(y_test, lr_pred))
print('Mean absolute error: ',mean_absolute_error(y_test, lr_pred))
print('R2 score: ',r2_score(y_test, lr_pred))
print('\n', '='*100, '\n')

print('The performance evaluation of Random Forest Regressor is below: ', '\n')
print('Mean squared error: ',mean_squared_error(y_test, rf_pred))
print('Mean absolute error: ',mean_absolute_error(y_test, rf_pred))
print('R2 score: ',r2_score(y_test, rf_pred))


# # Model Testing 

# In[253]:


# Checking a sample of predictor values on whom the model is trained
X.head()


# In[254]:


# Checking the rating according to above predictor variables
y.head()


# In[255]:


# Creating a new dataframe with values close to the 3rd row according to the sample above 

data2 = {'Year': [2016], 'Votes': [58], 'Duration': [121], 'Genre_mean_rating': [4.5], 'Director_encoded': [5.8], 'Actor1_encoded': [5.9], 'Actor2_encoded': [5.9], 'Actor3_encoded': [5.900]}
df = pd.DataFrame(data2)


# In[256]:


# Predict the movie rating
predicted_rating = rf.predict(df)

# Display the predicted rating
print("Predicted Rating:", predicted_rating[0])


#  
