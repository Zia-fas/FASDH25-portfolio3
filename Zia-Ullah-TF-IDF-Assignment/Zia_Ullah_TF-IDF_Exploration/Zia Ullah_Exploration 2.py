import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Path to the folder with all article text files
text_dir = 'C:/Users/DELL/Downloads/FASDH25-portfolio3/data/articles/'

# Step 2: Get all .txt filenames from the folder
all_files = [f for f in os.listdir(text_dir) if f.endswith('.txt')]
print(f"✅ Found {len(all_files)} article files.")

# Step 3: Read content of each file
texts = {}
for fname in all_files:
    full_path = os.path.join(text_dir, fname)
    try:
        with open(full_path, 'r', encoding='utf-8') as file:
            texts[fname] = file.read()
    except Exception as e:
        print(f"❌ Could not read {fname}: {e}")
        texts[fname] = ""

# Step 4: Prepare the list of filenames and their texts
filenames = list(texts.keys())
documents = [texts[f] for f in filenames]

# Step 5: Apply TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

# Step 6: Compute cosine similarity
cos_sim = cosine_similarity(tfidf_matrix)

# Step 7: Create similarity DataFrame
cos_sim_df = pd.DataFrame(cos_sim, index=filenames, columns=filenames)

# Step 8: Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(cos_sim_df, cmap='coolwarm', xticklabels=False, yticklabels=False)
plt.title("Cosine Similarity between Articles (TF-IDF)")
plt.xlabel("Articles")
plt.ylabel("Articles")
plt.tight_layout()
plt.show()

# Optional: Show top similarity scores (excluding self-similarity = 1.0)
print("\n🔍 Top Similarities Between Different Articles:")
similarities = []
for i in range(len(filenames)):
    for j in range(i+1, len(filenames)):
        score = cos_sim[i][j]
        similarities.append((filenames[i], filenames[j], round(score, 3)))

# Sort and print top 5 most similar article pairs
top_similar = sorted(similarities, key=lambda x: x[2], reverse=True)[:5]
for a, b, score in top_similar:
    print(f"{a} <--> {b} → similarity: {score}")
