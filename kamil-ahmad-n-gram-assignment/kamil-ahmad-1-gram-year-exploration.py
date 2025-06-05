import pandas as pd
import plotly.express as px

# Load the dataset
df = pd.read_csv("../data/dataframes/n-grams/1-gram/1-gram-year.csv")

# Ensure 'year' is treated as an integer
df['year'] = df['year'].astype(int)

# Define excluded terms
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

# Filter out excluded terms
df = df[~df['1-gram'].isin(excluded)]

# Group by 1-gram and compute total frequency
top_grams = (
    df.groupby('1-gram')['count-sum']
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

# Plot using Plotly bar chart
fig = px.bar(
    top_grams,
    x='1-gram',
    y='count-sum',
    title='Top 10 Most Frequent 1-Grams (All Years)',
    labels={'1-gram': '1-Gram', 'count-sum': 'Total Count'},
    text='count-sum'
)

fig.update_layout(
    template='plotly_white',
    title_font_size=18,
    margin=dict(l=50, r=50, t=60, b=50)
)

# Save and display the bar chart
fig.write_html("kamil-ahmad-1-gram-year-exploration.html")
fig.show()
