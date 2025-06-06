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

# First, ensure your dataframe 'df' exists and has the required columns
if 'df' in locals() or 'df' in globals():
    # Filter for high-similarity articles (0.4-0.9 range) and recent years
    high_sim_df = df[
        (df['similarity'] >= 0.4) & 
        (df['similarity'] <= 0.9) &
        (df['year-1'].between(2021, 2024)) & 
        (df['year-2'].between(2021, 2024))
    ].copy()

    # Create month-year labels
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Create period columns
    high_sim_df['period1'] = high_sim_df.apply(
        lambda x: f"{months[x['month-1']-1]} {x['year-1']}", axis=1)
    high_sim_df['period2'] = high_sim_df.apply(
        lambda x: f"{months[x['month-2']-1]} {x['year-2']}", axis=1)

    # Create month order for sorting
    month_order = [f"{month} {year}" for year in range(2021, 2025) for month in months]

    # Now create the heatmap_data by grouping
    heatmap_data = high_sim_df.groupby(['period1', 'period2'])['similarity'].mean().unstack()

    # Sort chronologically
    heatmap_data = heatmap_data.reindex(
        index=[x for x in month_order if x in heatmap_data.index],
        columns=[x for x in month_order if x in heatmap_data.columns]
    )

    # Create heatmap with annotations inside the graph
    fig_heat = px.imshow(
        heatmap_data,
        color_continuous_scale='RdYlBu_r',
        labels=dict(x="Second Article Date", y="First Article Date", color="Similarity"),
        title="<b>High-Similarity Article Trends (0.4-0.9)</b><br><span style='font-size:12px'>2021-2024 Monthly Comparison</span>",
        aspect="auto",
        zmin=0.4,
        zmax=0.9
    )

    # Enhanced layout with compact annotations
    fig_heat.update_layout(
        height=900,
        width=900,
        margin=dict(t=100, l=100, r=100, b=100),  # Reduced top margin
        xaxis_tickangle=45,
        coloraxis_colorbar=dict(
            title="<span style='font-size:12px'>Similarity</span>",
            thickness=15,
            tickvals=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            ticktext=["0.4", "0.5", "0.6", "0.7", "0.8", "0.9"],
            tickfont=dict(size=10),
            yanchor="middle"
        ),
        annotations=[
            # Interpretation guide placed in empty graph space
            dict(
                x=0.05,  # Left side
                y=0.95,  # Top of graph area
                xref="paper",
                yref="paper",
                text="<span style='font-size:11px'><b>How to read:</b><br>• Compare dates on X & Y axes<br>• Warmer colors = more similar<br>• Diagonal = time symmetry</span>",
                showarrow=False,
                font=dict(color="black"),
                align="left",
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="rgba(0,0,0,0.3)",
                borderwidth=1,
                borderpad=4
            ),
            # Year markers explanation
            dict(
                x=0.95,  # Right side
                y=0.15,  # Bottom of graph area
                xref="paper",
                yref="paper",
                text="<span style='font-size:10px'>Dashed lines = Year boundaries</span>",
                showarrow=False,
                font=dict(color="black"),
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="rgba(0,0,0,0.3)",
                borderwidth=1
            )
        ]
    )

    # Add diagonal line
    fig_heat.add_shape(
        type="line",
        x0=-0.5, y0=-0.5,
        x1=len(heatmap_data.columns)-0.5,
        y1=len(heatmap_data.index)-0.5,
        line=dict(color="black", width=1.5, dash="dot")
    )

    # Add subtle year separators
    for year in range(2021, 2025):
        pos = month_order.index(f"Jan {year}") - 0.5
        fig_heat.add_shape(
            type="line", 
            x0=pos, y0=-0.5, 
            x1=pos, y1=len(heatmap_data.index)-0.5,
            line=dict(color="rgba(255,255,255,0.7)", width=1.5, dash="dash")
        )
        fig_heat.add_shape(
            type="line", 
            x0=-0.5, y0=pos, 
            x1=len(heatmap_data.columns)-0.5, y1=pos,
            line=dict(color="rgba(255,255,255,0.7)", width=1.5, dash="dash")
        )

    # Save and show
    fig_heat.write_html("compact_high_similarity_heatmap.html")
    fig_heat.show()
else:
    print("Error: DataFrame 'df' not found. Please load your data first.")
