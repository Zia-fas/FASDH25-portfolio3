import pandas as pd
import plotly.express as px

# Load the 3-gram dataset
df = pd.read_csv("../data/dataframes/n-grams/3-gram/3-gram-year.csv")

# Ensure 'year' is integer
df['year'] = df['year'].astype(int)

# Split 3-gram into individual words
df[['w1', 'w2', 'w3']] = df['3-gram'].str.split(' ', n=2, expand=True)

# Define stop words
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

# Filter out 3-grams where all three words are stopwords
df = df[~((df['w1'].isin(excluded)) & (df['w2'].isin(excluded)) & (df['w3'].isin(excluded)))]

# Recombine into clean 3-gram
df['3-gram'] = df[['w1', 'w2', 'w3']].agg(' '.join, axis=1)

# Group by 3-gram and sum counts over all years
top_grams = (
    df.groupby('3-gram')['count-sum']
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

# Plot as bar chart using Plotly
fig = px.bar(
    top_grams,
    x='3-gram',
    y='count-sum',
    title='Top 10 Most Frequent 3-Grams (All Years)',
    labels={'count-sum': 'Total Count', '3-gram': '3-Gram'},
    text='count-sum'
)

fig.update_traces(marker_color='indigo', textposition='outside')
fig.update_layout(
    template="plotly_white",
    xaxis_tickangle=-45,
    title_font_size=18,
    margin=dict(l=50, r=50, t=60, b=100)
)

# Save plot as HTML
fig.write_html("kamil-ahmad-3-gram-bar-chart.html")

# Show the plot
fig.show()

