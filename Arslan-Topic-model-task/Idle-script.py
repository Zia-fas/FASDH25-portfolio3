import pandas as pd
import plotly.express as px

# 1. Load and prepare the data
df = pd.read_csv('../data/dataframes/topic-model/topic-model.csv')

# 2. Clean the data (simple version)
df = df[df["Topic"] != -1]  # Remove unclassified articles
df['date'] = pd.to_datetime(df[['year', 'month', 'day']])  # Create date column
df['month_year'] = df['date'].dt.strftime('%Y-%m')  # Format as "2021-01"

# 3. Create readable topic labels
df['Topic_Label'] = "Topic " + df['Topic'].astype(str)

# 4. Count articles per topic per month
topic_counts = df.groupby(['month_year', 'Topic_Label']).size().reset_index(name='Article Count')

# 5. Create the visualization
fig = px.line(
    topic_counts,
    x='month_year',
    y='Article Count',
    color='Topic_Label',
    title='Topic Trends Over Time',
    labels={'month_year': 'Month', 'Article Count': 'Number of Articles'},
    markers=True  # Adds dots to the line
)

# 6. Improve the layout
fig.update_layout(
    xaxis={'tickangle': 45},
    legend_title_text='Topics',
    hovermode='x unified'
)

# 7. Show the plot
fig.show()

