import plotly.express as px
import pandas as pd

# Firstly, we have to Load data
df = pd.read_csv('../data/dataframes/topic-model/topic-model.csv')

# Then let's see what columns we have
print("Available columns in the DataFrame:")
print(df.columns.tolist())

# now we have to filter data and remove unclassified articles if Topic = -1 exists
if 'Topic' in df.columns:
    df = df[df["Topic"] != -1]
else:
    print("'Topic' column not found - skipping filtering")

# we will Determine which column to use for topics
# Try common column names that might contain topic labels
topic_column = None
for possible_name in ['Topic_Label', 'Topic_Name', 'Topic', 'Label', 'topic_label']:
    if possible_name in df.columns:
        topic_column = possible_name
        break

if topic_column is None:
    raise ValueError("Could not find a topic label column in the DataFrame")

# We have to Count articles per topic
topic_counts = df[topic_column].value_counts().reset_index()
topic_counts.columns = ["Topic", "Count"]  # Using generic column names

# Get top 5 topics
top5_topics = topic_counts.head(5)

# Create bar graph for clearer visualization
fig = px.bar(top5_topics, 
             x="Topic", 
             y="Count", 
             color="Topic",
             title="Top 5 Most Frequent Topics",
             labels={"Topic": "Topic", "Count": "Number of Articles"},
             text="Count")

# for lay out customization 
fig.update_layout(
    showlegend=False,
    xaxis_title="Topic",
    yaxis_title="Number of Articles",
    title_x=0.5,
    height=600
)

# Rotate x-axis labels if needed
fig.update_xaxes(tickangle=45)

# save the figure in repository
fig.write_html("Topic_modeling.html")
