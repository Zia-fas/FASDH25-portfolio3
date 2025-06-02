import os
import pandas as pd
import plotly.express as px

# Path to article folder
folder = '../data/articles'

# Empty dictionary to store word lengths per year
word_lengths = {}

# Loop through each article file
for filename in os.listdir(folder):
    if filename.endswith('.txt'):
        year = filename[:4]  # Files are saved as 'YYYY-MM-DD-XXXX.txt'
        file_path = os.path.join(folder, filename)

        with open(file_path, 'r', encoding='utf-8') as f:
            words = f.read().split()
            word_count = len(words)

        if year not in word_lengths:
            word_lengths[year] = []

        word_lengths[year].append(word_count)

# Create summary of min and max word lengths per year
summary = []
for year, lengths in word_lengths.items():
    summary.append({
        'year': int(year),
        'min_length': min(lengths),
        'max_length': max(lengths)
    })

# Convert summary to DataFrame
df_extremes = pd.DataFrame(summary).sort_values('year')

# dataframe informations
print("First few rows of extremes DataFrame:\n", df_extremes.head(), "\n")
print("Data types:\n", df_extremes.dtypes, "\n")
print("Year range:\n", df_extremes['year'].min(), "to", df_extremes['year'].max(), "\n")

# check for the line chart for the range per year
fig_line = px.line(
    df_extremes,
    x='year',
    y=['min_length', 'max_length'],
    title='Shortest and Longest Article Lengths Per Year',
    labels={'value': 'Word Count', 'variable': 'Type'},
    markers=True
)
fig_line.write_html('exploration-min-max-lengths-line.html')

# check how bar graph would look like
df_melted = df_extremes.melt(id_vars='year', value_vars=['min_length', 'max_length'],
                              var_name='Type', value_name='Word Count')

fig_bar = px.bar(
    df_melted,
    x='year',
    y='Word Count',
    color='Type',
    barmode='group',
    title='Shortest and Longest Article Lengths Per Year',
    labels={'year': 'Year'}
)
fig_bar.write_html('exploration-min-max-lengths-bar.html')
