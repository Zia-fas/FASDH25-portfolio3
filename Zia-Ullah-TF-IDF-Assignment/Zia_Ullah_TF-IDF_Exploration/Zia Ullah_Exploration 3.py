#import libraries
#https://chat.deepseek.com/a/chat/s/42f7726a-3b0e-4774-b391-9a39c05e1a90
#https://www.geeksforgeeks.org/visualizing-tf-idf-scores-a-comprehensive-guide-to-plotting-a-document-tf-idf-2d-graph/
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Set Plotly to display graphs in the default web browser
pio.renderers.default = 'browser'

# # Load a CSV file containing TF-IDF (Term Frequency-Inverse Document Frequency) data
# File contains words/terms with TF-IDF scores above 0.3 and document length of 200
file_path = r'C:\Users\DELL\Downloads\FASDH25-portfolio3\data\dataframes\tfidf\tfidf-over-0.3-len200.csv'
try:
    
    # Attempt to read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    
     # If successful, print confirmation and show available columns
    print("Data loaded successfully. Columns:", df.columns.tolist())
except Exception as e:
    
    # If loading fails, print the error message for debugging
    print("Error loading file:", e)

# Histogram showing counts of articles and similarity score (Exclude 0.1-0.2 range)
if 'df' in locals(): # Only proceed if the DataFrame 'df' exists
    
   # Filter out similarity scores below 0.2 (assumed to be noise/irrelevant)
    hist_df = df[df['similarity'] >= 0.2].copy()

    # Create interactive histogram using Plotly Express
    fig_hist = px.histogram(
        hist_df,
        x='similarity', # Column containing similarity scores (TF-IDF-based)
        title='Article Pair Similarity Distribution (0.2-1.0 Range)',
        labels={'similarity': 'TF-IDF Similarity Score', 'count': 'Number of Pairs'},
        nbins=40,  # More bins for finer granularity
        color_discrete_sequence=['#3366CC'],
        opacity=0.8
    )

    # Improve plot readability and styling
    fig_hist.update_layout(
        bargap=0.1,                  # Small gap between bars for clarity
        yaxis_title='Count of Article Pairs',
        xaxis_title='Similarity Score',
        plot_bgcolor='white',                  # White background for cleaner look
        hovermode='x',                        # Show hover tooltips aligned to x-axis values
        xaxis=dict(range=[0.2, 1.0])  # Explicitly set x-axis range
    )
    
    # Highlight analysis range
    # Vertical rectangle from x=0.3 to x=0.5
    fig_hist.add_vrect(
        x0=0.3, x1=0.5,                    # Semi-transparent green highlight
        fillcolor="green", opacity=0.15,
        line_width=0,               # No border line
        annotation_text="Focus Range (0.3-0.5)",      # Label for the highlighted area
        annotation_position="top left"               # Position of the label
    )
    
    fig_hist.write_html("similarity_histogram.html")
    fig_hist.show()

# heatmap to show the trends according to the similarity score (Monthly view with better labels)
if 'df' in locals():
    # Filter for 2021-2024 and similarity 0.3-0.5
    heatmap_df = df[
        (df['similarity'] >= 0.3) & 
        (df['similarity'] <= 0.5) &
        (df['year-1'].between(2021, 2024)) & 
        (df['year-2'].between(2021, 2024))
    ].copy()
    
    # Create month-year labels with actual month names
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    heatmap_df['period1'] = heatmap_df.apply(
        lambda x: f"{months[x['month-1']-1]} {x['year-1']}", axis=1)
    heatmap_df['period2'] = heatmap_df.apply(
        lambda x: f"{months[x['month-2']-1]} {x['year-2']}", axis=1)
    
    # Aggregate and sort chronologically
    heatmap_data = heatmap_df.groupby(['period1', 'period2'])['similarity'].mean().unstack()
    
    # Sort columns and index chronologically
    month_order = [f"{month} {year}" 
                  for year in range(2021, 2025) 
                  for month in months]
    heatmap_data = heatmap_data.reindex(
        index=[x for x in month_order if x in heatmap_data.index],
        columns=[x for x in month_order if x in heatmap_data.columns]
    )
    
    # Create heatmap with improved settings
    fig_heat = px.imshow(
        heatmap_data,
        color_continuous_scale='Rainbow',  # More colorful scale
        labels=dict(x="Article 2 Date", y="Article 1 Date", color="Similarity"),
        title="Monthly Article Similarity (2021-2024, Scores 0.3-0.5)",
        aspect="auto",
        zmin=0.3,
        zmax=0.5
    )
    
    # Enhanced layout
    fig_heat.update_layout(
        height=900,
        width=900,
        xaxis_tickangle=45,
        coloraxis_colorbar=dict(
            title="Similarity",
            thickness=20,
            tickvals=[0.3, 0.35, 0.4, 0.45, 0.5],
            ticktext=["0.3", "0.35", "0.4", "0.45", "0.5"]
        )
    )
    
    # Add diagonal and annotations
    fig_heat.add_shape(
        type="line",
        x0=-0.5, y0=-0.5,
        x1=len(heatmap_data.columns)-0.5,
        y1=len(heatmap_data.index)-0.5,
        line=dict(color="black", width=2)
    )
    
    # Add year separators
    for year in range(2021, 2025):
        pos = month_order.index(f"Jan {year}") - 0.5
        fig_heat.add_shape(type="line", x0=pos, y0=-0.5, x1=pos, y1=len(heatmap_data.index)-0.5,
                         line=dict(color="white", width=2, dash="dash"))
        fig_heat.add_shape(type="line", x0=-0.5, y0=pos, x1=len(heatmap_data.columns)-0.5, y1=pos,
                         line=dict(color="white", width=2, dash="dash"))
    
    fig_heat.write_html("monthly_similarity_heatmap.html")
    fig_heat.show()
fig.show()


# Filter for low-similarity articles (0.3-0.4 range) and recent years
low_sim_df = df[
    (df['similarity'] >= 0.3) & 
    (df['similarity'] < 0.4) &
    (df['year-1'].between(2021, 2024)) & 
    (df['year-2'].between(2021, 2024))
].copy()

# 1. Temporal Heatmap Analysis
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Create period labels
low_sim_df['period1'] = low_sim_df.apply(
    lambda x: f"{months[x['month-1']-1]} {x['year-1']}", axis=1)
low_sim_df['period2'] = low_sim_df.apply(
    lambda x: f"{months[x['month-2']-1]} {x['year-2']}", axis=1)

# Aggregate data
heatmap_data = low_sim_df.groupby(['period1', 'period2'])['similarity'].mean().unstack()

# Sort chronologically
month_order = [f"{month} {year}" for year in range(2021, 2025) for month in months]
heatmap_data = heatmap_data.reindex(
    index=[x for x in month_order if x in heatmap_data.index],
    columns=[x for x in month_order if x in heatmap_data.columns]
)

# Create heatmap
fig_heat = px.imshow(
    heatmap_data,
    color_continuous_scale='Blues',
    labels=dict(x="Article 2 Date", y="Article 1 Date", color="Similarity"),
    title="Low-Similarity Articles (0.3-0.4) by Publication Month",
    aspect="auto",
    zmin=0.3,
    zmax=0.4
)

# Add temporal reference lines
for year in range(2021, 2025):
    pos = month_order.index(f"Jan {year}") - 0.5
    fig_heat.add_shape(type="line", x0=pos, y0=-0.5, x1=pos, y1=len(heatmap_data.index)-0.5,
                     line=dict(color="gray", width=1, dash="dot"))
    fig_heat.add_shape(type="line", x0=-0.5, y0=pos, x1=len(heatmap_data.columns)-0.5, y1=pos,
                     line=dict(color="gray", width=1, dash="dot"))

# Thematic Analysis (using article titles)
titles = pd.concat([low_sim_df['title-1'], low_sim_df['title-2']]).unique()

# Basic topic modeling
vectorizer = TfidfVectorizer(max_df=0.9, min_df=3, stop_words='english')
dtm = vectorizer.fit_transform(titles)

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(dtm)

# Get top words for each topic
feature_names = vectorizer.get_feature_names_out()
topics = []
for idx, topic in enumerate(lda.components_):
    topics.append(", ".join([feature_names[i] for i in topic.argsort()[-5:]]))

# Add thematic annotation to heatmap
fig_heat.update_layout(
    annotations=[
        dict(
            x=0.5, y=1.1,
            xref="paper", yref="paper",
            text=f"Common Themes in Low-Similarity Articles: {', '.join(topics)}",
            showarrow=False,
            font=dict(size=12)
        )
    ],
    height=800,
    width=800,
    margin=dict(t=150) # Extra space for annotation
)

# Yearly Distribution Analysis
yearly_dist = low_sim_df.groupby(['year-1', 'year-2']).size().unstack()
fig_dist = px.imshow(
    yearly_dist,
    color_continuous_scale='Greens',
    labels=dict(x="Article 2 Year", y="Article 1 Year", color="Count"),
    title="Distribution of Low-Similarity Pairs by Year"
)

# Save and show
fig_heat.write_html("low_similarity_heatmap.html")
fig_dist.write_html("yearly_distribution.html")

fig_heat.show()
fig_dist.show()
