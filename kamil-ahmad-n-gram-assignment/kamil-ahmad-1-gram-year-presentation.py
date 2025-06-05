import pandas as pd
import plotly.express as px

# Manually defined stop words (from https://gist.github.com/sebleier/554280) + added "people" and "said"
excluded_words = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up",
    "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
    "should", "now",
    "people", "said"  # Added words
])

# Load the data
df = pd.read_csv("../data/dataframes/n-grams/1-gram/1-gram-year.csv")

# Print basic info
print(df.head())
print("Columns:", df.columns.tolist())
print("Unique 1-grams:", df['1-gram'].nunique())
print("Years:", sorted(df['year'].unique()))

# Ensure 'year' is treated as integer
df['year'] = df['year'].astype(int)

# Remove stop words from '1-gram' column
df = df[~df['1-gram'].isin(excluded_words)]

# Group by 1-gram and get total count across all years
top_grams = (
    df.groupby('1-gram')['count-sum']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("Top 10 most frequent 1-grams (after stop words removal):")
print(top_grams)

# Select top 10 1-grams for plotting
top_words = top_grams.index[:10]

# Filter data to only include top 1-grams
df_top = df[df['1-gram'].isin(top_words)]

# Plot using Plotly
fig = px.line(
    df_top,
    x="year",
    y="count-sum",
    color="1-gram",
    markers=True,
    title="Frequency of Top 1-Grams Over Time (Stop Words Removed)",
    labels={"count-sum": "Count", "year": "Year", "1-gram": "1-Gram"}
)

fig.update_layout(
    template="plotly_white",
    title_font_size=18,
    legend_title="1-Gram",
    legend=dict(x=0.01, y=0.99),
    margin=dict(l=50, r=50, t=60, b=50)
)

# Save the interactive plot as HTML
fig.write_html("kamil-ahmad-1-gram-presentation-visualization.html")

# Show the plot
fig.show()



