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

# Create a DataFrame for the list of years
df = pd.DataFrame({'year': years})

# Print basic info 
print("First few rows:\n", df.head(), "\n")
print("Data types:\n", df.dtypes, "\n")
print("Year frequency:\n", df['year'].value_counts().sort_index(), "\n")

# Count how many times each year appears (number of articles)
article_counts = df['year'].value_counts().sort_index().reset_index()
article_counts.columns = ['year', 'article_count']
article_counts['year'] = article_counts['year'].astype(int)

# Bar chart of number of articles per year
fig_bar = px.bar(
    article_counts,
    x='year',
    y='article_count',
    title='Number of Articles Published per Year (Bar Chart)',
    labels={'article_count': 'Articles'},
    text='article_count'
)
fig_bar.write_html('exploration-article-count-bar.html')

# Line chart to see article trend over time
fig_line = px.line(
    article_counts.sort_values('year'),
    x='year',
    y='article_count',
    title='Number of Articles Published per Year (Line Chart)',
    labels={'article_count': 'Articles'},
    markers=True
)
fig_line.write_html('exploration-article-count-line.html')



