import os
import plotly.express as px
import pandas as pd

# Folder containing the articles
folder = '../data/articles'

# Get list of files and extract the year
article_files = os.listdir(folder)
years = []

for filename in article_files:
    if filename.endswith('.txt'):
        year = filename[:4]  # as the format is 'YYYY-MM-DD_XXXX.txt'
        years.append(year)

# Count articles per year

# create a dataframe for the list of years
df = pd.DataFrame({'year': years})

# count how many times each year appear which is the number of articles
article_counts = df['year'].value_counts().sort_index().reset_index()

# rename year as article count for clarity
article_counts.columns = ['year', 'article_count']

# converting sting 'year' into integer type
article_counts['year'] = article_counts['year'].astype(int)

# Plotting with text on bars
fig = px.bar(
    article_counts,
    x='year',
    y='article_count',
    title='Number of Articles Published per Year',
    labels={'article_count': 'Articles'},
    text='article_count'  # Show count on bars
)


fig.write_html('Faizan-Amir-article-count-per-year.html')
