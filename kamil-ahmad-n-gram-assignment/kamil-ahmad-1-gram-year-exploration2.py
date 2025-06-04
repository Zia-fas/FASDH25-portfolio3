
import pandas as pd
import plotly.express as px

# Load the data
df = pd.read_csv("../data/dataframes/n-grams/1-gram/1-gram-year.csv")

# Print basic info
print(df.head())
print("Columns:", df.columns.tolist())
print("Unique 1-grams:", df['1-gram'].nunique())
print("Years:", sorted(df['year'].unique()))

# Ensure 'year' is treated as integer
df['year'] = df['year'].astype(int)

# Group by 1-gram and get total count across all years
top_grams = (
    df.groupby('1-gram')['count-sum']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("Top 10 most frequent 1-grams:")
print(top_grams)

# Select top 5 1-grams for plotting
top_words = top_grams.index[:5]

# Filter data to only include top 1-grams
df_top = df[df['1-gram'].isin(top_words)]

# Plot using Plotly
fig = px.line(
    df_top,
    x="year",
    y="count-sum",
    color="1-gram",
    markers=True,
    title="Frequency of Top 1-Grams Over Time",
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
fig.write_html("kamil-ahmad-1-gram-year-exploration2.html")

# Show the plot
fig.show()

