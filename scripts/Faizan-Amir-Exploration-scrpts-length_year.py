import os
import pandas as pd
import plotly.express as px

# Load the CSV file
df_year = pd.read_csv('../data/dataframes/length/length-year.csv')


# info about the dataset

print(df_year.head(), "\n")


print(df_year.columns, "\n")


print(df_year.dtypes, "\n")


# Check for outlier years
print(df_year['year'].sort_values().unique(), "\n")

# trend line for average word
# exploring line chart
fig = px.line(
    df_year.sort_values('year'),
     x='year',
     y='length-mean',
    title='Avg Words per Article',
    markers=True
)
fig.write_html("exploration-avg-length-line.html")

# explore bar chart 
figure2 = px.bar(
    df_year.sort_values('year'),
    x='year',
    y='length-mean',
    title='Average Words per Article (Bar Chart)',
    labels={'length-mean': 'Average Words per Article', 'year': 'Year'},
    text='length-mean'  # Show values on top of bars
)

figure2.write_html("exploration-avg-length-bar.html")

# total word in article per year graphs
#exploring bar chart
figure3 = px.bar(
    df_year.sort_values('year'),
    x='year',
    y='length-sum',
    title='Total Words Published per Year',
    labels={'length-sum': 'Total Words', 'year': 'Year'},
    text='length-sum'  # show value on top of bars
)
figure3.write_html("exploration-total-length-bar.html")

# exploring histogram
figure4 = px.histogram(
    df_year,
    x='length-sum',
    nbins=10,
    title='Distribution of Total Words Published per Year',
    labels={'length-sum': 'Total Words per Year'}
)

figure4.write_html("exploration-total-length-histo.html")

# exploring line chart
figure5 = px.line(
    df_year.sort_values('year'),
    x='year',
    y='length-sum',
    title='Total Words Published in Articles Each Year (Line Chart)',
    labels={'length-sum': 'Total Words', 'year': 'Year'},
    markers=True
)

figure5.write_html("exploration-total-length-line.html")
