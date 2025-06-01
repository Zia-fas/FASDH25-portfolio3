import os
import pandas as pd
import plotly.express as px

# Load the CSV file
df_year = pd.read_csv('../data/dataframes/length/length-year.csv')

# Sort by year to make the line chart 
df_year = df_year.sort_values('year')

# Average words per article per year
fig1 = px.line(
    df_year,
    x='year',
    y='length-mean',
    title='Average Words per Article Over the Years',
    markers=True,
    labels={'length-mean': 'Average Words per Article', 'year': 'Year'}
)

 fig1.write_html("Faizan-Amir-average-words-per-article.html")

# Total words in articles per year
fig2 = px.bar(
    df_year,
    x='year',
    y='length-sum',
    title='Total Words Published in Articles Each Year',
    labels={'length-sum': 'Total Words in Articles', 'year': 'Year'},
    text='length-sum'
)

 fig2.write_html("Faizan-Amir-total-words-published.html")
