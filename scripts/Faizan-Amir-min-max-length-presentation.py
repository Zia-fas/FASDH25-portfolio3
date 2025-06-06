import os
import pandas as pd
import plotly.express as px

# Path to article folder
folder = '../data/articles'

# empty dictionary to store word length by year
word_lengths = {}

# Loop through each article file
for filename in os.listdir(folder):
    if filename.endswith('.txt'): # processing only the files with .txt
        year = filename[:4] # considering the first 4 digits as that is the year because file are saved in format od YYYY-MM-DD-XXXX
        file_path = os.path.join(folder, filename)

        with open(file_path, 'r', encoding='utf-8') as f:
            # split text into words and count them
            words = f.read().split()
            word_count = len(words)

        # If the year is not already a key in the dictionary, start it with an empty list
        if year not in word_lengths:
            word_lengths[year] = []

        word_lengths[year].append(word_count)# Append the word count of the current article to the list for its year


# Create a list of min and max values per year # See AI documentation 1
summary = []

for year, lengths in word_lengths.items():
    summary.append({
        'year': int(year),
        'min_length': min(lengths),
        'max_length': max(lengths)
    })

# Convert to DataFrame # see AI documentation 1
df_extremes = pd.DataFrame(summary).sort_values('year')

# Plotting in line chart
fig = px.line(
    df_extremes,
    x='year',
    y=['min_length', 'max_length'],
    title='Shortest and Longest Article Lengths Per Year',
    labels={'value': 'Word Count', 'variable': 'Type'},
    markers=True
)

fig.write_html('Faizan-Amir-min-max-lengths.html')
