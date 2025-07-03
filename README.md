# News Article Clustering and Topic Modeling

This project analyzes a dataset of BBC news articles to identify thematic structures and group similar articles using text preprocessing, dimensionality reduction, clustering, and topic modeling.

## Features

- **Data Loading and Exploration**: Loads the BBC news dataset (`bbc-news-data.csv`) and performs initial checks.
- **Text Preprocessing**:
  - Tokenization
  - Lowercasing
  - Punctuation removal
  - Stop word removal (including custom stop words)
  - Lemmatization
  - Pronoun and number removal
  - Removal of words shorter than three characters
- **TF-IDF Vectorization**: Converts processed text into numerical feature vectors using TF-IDF.
- **Dimensionality Reduction**:
  - **PCA**: Reduces TF-IDF matrix dimensionality while retaining 95% variance.
  - **t-SNE**: Reduces to 2D for cluster visualization, preserving local structures.
- **Clustering (K-Means)**:
  - Uses the Elbow Method to estimate optimal cluster numbers.
  - Evaluates clustering with Normalized Mutual Information (NMI) against true labels.
- **Topic Modeling (LDA)**: Applies Latent Dirichlet Allocation to identify latent topics within clusters.
- **Topic Coherence (C_V)**: Determines optimal topic numbers for meaningful, interpretable topics.
- **Keyword Extraction**: Extracts top keywords for each document based on dominant topics.
- **Interactive Visualization**: Provides a Dash application for exploring t-SNE cluster plots with keyword search functionality.
- **Word Clouds**: Generates visual representations of frequent and important keywords per cluster.

## Intuition Behind Key Concepts

- **TF-IDF**: Measures word importance in a document relative to the corpus, down-weighting common words.
- **Curse of Dimensionality**: High-dimensional data complicates clustering; dimensionality reduction (PCA, t-SNE) improves cluster separation.
- **t-SNE**: Preserves local data relationships for effective 2D/3D visualization of high-dimensional data.
- **Topic Coherence (C_V)**: Assesses topic interpretability by measuring semantic similarity of top words.

## Getting Started

### Prerequisites
- Python 3.x
- Jupyter Notebook or Google Colab

### Installation
1. Clone the repository or download the notebook.
2. Install required packages:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn tqdm spacy nltk gensim wordcloud dash plotly
   python -m spacy download en_core_web_sm
   ```

### Usage
1. Place `bbc-news-data.csv` in the same directory as the notebook.
2. Open `Project_3_Fill_in.ipynb` in Jupyter or Colab.
3. Run cells sequentially to preprocess data, perform clustering, topic modeling, and generate visualizations.
4. Explore the interactive t-SNE plot and word clouds for cluster analysis.

## Dataset
- **File**: `bbc-news-data.csv`
- **Description**: Contains BBC news articles categorized by topic.

## Project Structure
- `Project_3_Fill_in.ipynb`: Main notebook with preprocessing, clustering, topic modeling, and visualization code.
- `bbc-news-data.csv`: Input dataset of BBC news articles.
- `processed_df.pkl`: Pickled DataFrame with preprocessed text (generated).
- `clustered_df.pkl`: Pickled DataFrame with clustered data (generated).
- `final_df.pkl`: Pickled DataFrame with keywords and t-SNE coordinates (generated).
- `pca_model.pkl`: Pickled PCA model (generated).
- `kmeans_model_nmi.pkl`: Pickled K-Means model based on NMI (generated).