import pandas as pd
import plotly.express as px

# Load the 2-gram data
df = pd.read_csv("../data/dataframes/n-grams/2-gram/2-gram-year.csv")

# Ensure 'year' is treated as integer
df['year'] = df['year'].astype(int)

# Define excluded words (stopwords)
excluded = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up",
    "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
    "should", "now", "people", "said"
])

# Split 2-gram into two words
split_words = df['2-gram'].str.split(' ', n=1, expand=True)
df['w1'] = split_words[0]
df['w2'] = split_words[1]

# Remove rows where both words are stopwords
df = df[~((df['w1'].isin(excluded)) & (df['w2'].isin(excluded)))]

# Group by 2-gram and get total count
top_2grams = (
    df.groupby('2-gram')['count-sum']
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

# Plot bar chart
fig = px.bar(
    top_2grams,
    x='2-gram',
    y='count-sum',
    title='Top 10 Most Frequent 2-Grams (All Years)',
    labels={'2-gram': '2-Gram', 'count-sum': 'Total Count'},
    text='count-sum'
)

fig.update_layout(
    template='plotly_white',
    title_font_size=18,
    margin=dict(l=50, r=50, t=60, b=50)
)

# Save and show the plot
fig.write_html("kamil-ahmad-2gram-bar-chart.html")
fig.show()
