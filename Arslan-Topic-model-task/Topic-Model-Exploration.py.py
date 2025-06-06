import plotly.express as px
import pandas as pd

# Firstly we will Load the data
df = pd.read_csv('../data/dataframes/topic-model/topic-model.csv')

# then let's see what columns we have
print("Available columns in the DataFrame:")
print(df.columns.tolist())

# now we will Filter data and remove unclassified articles if Topic = -1 exists
if 'Topic' in df.columns:
    df = df[df["Topic"] != -1].copy()  # Add .copy() to avoid chained assignment
else:
    print("'Topic' column not found - skipping filtering")

# Get the top 5 most frequent topics
top_5_topics = df['Topic'].value_counts().head(5).index.tolist()

# we have to filter dataframe for only these top 5 topics and create a copy
df_top5 = df[df['Topic'].isin(top_5_topics)].copy()  # Add .copy() here

# Now Prepare keyword data - melt the topic keyword columns
keyword_cols = ['topic_1', 'topic_2', 'topic_3', 'topic_4']
melted_keywords = df_top5.melt(id_vars=['Topic'], 
                              value_vars=keyword_cols, 
                              var_name='keyword_rank', 
                              value_name='keyword')

# Count keyword frequencies per topic
keyword_counts = melted_keywords.groupby(['Topic', 'keyword']).size().reset_index(name='count')

# Get top 3 keywords for each topic
top_keywords = keyword_counts.sort_values(['Topic', 'count'], ascending=[True, False]) \
                            .groupby('Topic').head(3)

# Now, we have to Create labels combining topic number with top keywords
topic_labels = {}
for topic in top_5_topics:
    keywords = top_keywords[top_keywords['Topic'] == topic]['keyword'].tolist()[:3]
    topic_labels[topic] = f"Topic {topic}: " + ", ".join(keywords)

# Apply these labels to our dataframe using .loc
df_top5.loc[:, 'Topic_Label'] = df_top5['Topic'].map(topic_labels)  # Modified this line

# Count articles per labeled topic
topic_counts = df_top5['Topic_Label'].value_counts().reset_index()
topic_counts.columns = ["Topic_Label", "Count"]

# then Create interactive bar chart
fig = px.bar(topic_counts,
             x="Count",
             y="Topic_Label",
             color="Topic_Label",
             orientation='h',
             title="Top 5 Key Topics with Representative Keywords",
             labels={"Topic_Label": "Topic with Top Keywords", "Count": "Number of Articles"},
             text="Count")

# finally, we have to Customize the layout
fig.update_layout(
    showlegend=False,
    yaxis_title="Topic (with top 3 keywords)",
    xaxis_title="Number of Articles",
    title_x=0.5,
    height=500,
    margin=dict(l=150)  # Add left margin for long labels
)

# Save the figure
fig.write_html("topic_modeling_with_keywords.html")

print("Visualization saved as 'topic_modeling_with_keywords.html'")
