import plotly.express as px
import pandas as pd

# # First things first, load the topic model CSV data — this has all the news data we want to visualize.
df = pd.read_csv('../data/dataframes/topic-model/topic-model.csv')

# Now let's create a proper datetime column from separate year, month, day columns.
# If there's any issue (like a missing date), we'll catch it and just assign a fallback date range.
try:
    df['Date'] = pd.to_datetime(df[['year', 'month', 'day']])
except Exception as e:
    print(f"Error creating Date column: {e}")
    df['Date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')[:len(df)]

# Drop rows where the topic model assigned -1 (i.e., probably unclassified or junk topics)
df = df[df["Topic"] != -1]  

# Okay, time to define a big set of stopwords — these are generic words that don’t carry much meaning.
# We’ll use this to filter out rows that are just full of filler words and not useful for classification.
stop_words = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 
    'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 
    'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 
    'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 
    'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
    'very', 's', 't', 'can', 'will', 'just', 'al', 'said', 'don', 'should', 'now',
    'one', 'get', 'got', 'say', 'says', 'made', 'make', 'thing', 'things', 'like', 
    'see', 'still', 'also', 'new', 'news', 'use', 'used', 'using', 'every', 'many', 
    'much', 'back', 'even', 'really', 'another', 'year', 'years'
}
# Now remove rows where *all four* topic keywords are stopwords — i.e., there's nothing useful to work with.
df = df[~df.apply(lambda row: all(word.lower() in stop_words 
                for word in [row['topic_1'], row['topic_2'], 
                            row['topic_3'], row['topic_4']]), axis=1)]

#Now we will categorize each article into one of three themes based on keyword matching.
def classify_topic(row):
    keywords = [str(row['topic_1']).lower(), str(row['topic_2']).lower(), 
                str(row['topic_3']).lower(), str(row['topic_4']).lower()]
     
    security_words = {'hamas', 'missile', 'force', 'military', 'attack', 'war', 'terror', 'defense',
        'hezbollah', 'houthi', 'houthis', 'lebanon', 'lebanese', 'gaza', 'west', 'bank',
        'israeli', 'palestinian', 'border', 'iran', 'iranian', 'syria', 'sea', 'killed'}
    diplomacy_words = {'peace', 'negotiation', 'netanyahu', 'biden', 'un', 'treaty', 'diplomacy', 'resolution',
        'us', 'government', 'engagement'}
    civilian_words = {'child', 'hospital', 'civilian', 'aid', 'refugee', 'death', 'victim', 'humanitarian',
        'patients', 'medical', 'hospitals', 'hostages', 'captives', 'akleh'}
# Check each set — the first one that matches wins. Otherwise, we call it 'Other'.
    if any(word in security_words for word in keywords):
        return 'Security & Conflict'
    if any(word in diplomacy_words for word in keywords):
        return 'Diplomacy & Politics'
    if any(word in civilian_words for word in keywords):
        return 'Civilian Toll, Crisis & Aid'
    return 'Other'

df['Theme'] = df.apply(classify_topic, axis=1)
df = df[df['Theme'] != 'Other']

# Now create monthly period
df['Month'] = df['Date'].dt.to_period('M').astype(str)

# Prepare data for bar chart
theme_counts = df.groupby(['Month', 'Theme']).size().reset_index(name='Count')

# Create stacked bar chart
fig = px.bar(
    theme_counts,
    x='Month',
    y='Count',
    color='Theme',
    title='Media Discourse Themes by Month (Stacked Bar Chart)',
    labels={'Count': 'Number of Articles', 'Month': 'Time Period'},
    color_discrete_map={
        'Security & Conflict': '#EF553B',
        'Diplomacy & Politics': '#636EFA', 
        'Civilian Toll, Crisis & Aid': '#00CC96'
    },
    template='plotly_white'
)

# Improve layout
fig.update_layout(
    barmode='stack',
    hovermode='x unified',
    height=600,
    width=1000,
    xaxis_title='Time Period',
    yaxis_title='Number of Articles',
    margin={'l': 50, 'r': 50, 't': 80, 'b': 50},
    legend_title='Discourse Theme'
)

fig.update_xaxes(tickangle=45, tickvals=theme_counts['Month'].unique()[::3])  # Show every 3rd month
fig.write_html("Topic-Modeling_Presentation_trends.html")

