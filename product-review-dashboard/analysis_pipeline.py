# analysis_pipeline.py

import pandas as pd
import numpy as np
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from collections import defaultdict, Counter

# Download required NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Initialize components
nlp = spacy.load('en_core_web_sm')
sia = SentimentIntensityAnalyzer()
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
keybert_model = KeyBERT(sentence_model)

# Custom stop words
custom_stop_words = {'a', 'i', 'it', 'you', 'the', 'they', 'them', 'that', 'this'}
stop_words = list(set(stopwords.words('english')).union(custom_stop_words))

def extract_topics_with_counts(reviews, n_topics=3, n_top_words=5):
    """Extract topics using LDA and return top words with their counts."""
    if isinstance(reviews, str):
        reviews = [reviews]
    
    vectorizer = CountVectorizer(
        stop_words=stop_words,
        max_features=1000,
        ngram_range=(1, 2),
        min_df=1
    )
    
    review_matrix = vectorizer.fit_transform(reviews)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(review_matrix)
    
    terms = vectorizer.get_feature_names_out()
    word_counts = np.asarray(review_matrix.sum(axis=0)).flatten()
    term_count_dict = dict(zip(terms, word_counts))
    
    unique_topics = set()
    for topic_idx, topic in enumerate(lda.components_):
        topic_terms = sorted(
            [(terms[i], term_count_dict[terms[i]]) for i in topic.argsort()[-n_top_words:]],
            key=lambda x: x[1],
            reverse=True
        )
        unique_topics.update(topic_terms)
    
    return list(unique_topics)

def extract_keywords_with_counts(reviews, model, top_n=5):
    """Extract keywords using KeyBERT and return with their counts."""
    if not isinstance(reviews, str):
        reviews = ' '.join(reviews)
    
    keywords = model.extract_keywords(
        reviews, 
        keyphrase_ngram_range=(1, 2),
        stop_words=stop_words,
        top_n=top_n
    )
    
    word_counter = Counter(reviews.split())
    return [(kw[0], word_counter.get(kw[0].split()[0], 0)) for kw in keywords]

def extract_noun_phrases_with_counts(text):
    """Extract noun phrases and calculate their frequency."""
    if not isinstance(text, str):
        text = ' '.join(text)
    
    doc = nlp(text)
    feature_counts = Counter()
    
    for chunk in doc.noun_chunks:
        feature_tokens = [token for token in chunk if token.lemma_.lower() not in stop_words]
        
        if not feature_tokens:
            continue
        
        feature = ' '.join(token.text for token in feature_tokens).lower().strip()
        
        if len(feature.split()) > 0 and len(feature.split()) <= 3:
            feature_counts[feature] += 1
    
    return list(feature_counts.items())

def calculate_sentiment(reviews, feature_list):
    """Calculate sentiment scores for given features."""
    if not isinstance(reviews, str):
        reviews = ' '.join(reviews)
    
    sentences = [s.strip() for s in reviews.split('.') if s.strip()]
    sentiment_data = {}
    
    for feature, freq in feature_list:
        feature_sentiments = []
        for sentence in sentences:
            if feature.lower() in sentence.lower():
                feature_sentiments.append(sia.polarity_scores(sentence)['compound'])
        
        avg_score = (sum(feature_sentiments) / len(feature_sentiments)) if feature_sentiments else 0
        sentiment_label = 'Positive' if avg_score > 0 else 'Negative' if avg_score < 0 else 'Neutral'
        
        sentiment_data[feature] = {
            'score': avg_score,
            'label': sentiment_label
        }
    
    return sentiment_data

def extract_combined_features(reviews_df, sentence_model, keybert_model, n_topics=3, n_top_words=5, top_keywords=5):
    """Combine features from different extraction methods with enhanced filtering."""
    reviews = ' '.join(reviews_df['comb_preprocessed_reviews'].tolist())
    # Extract features from different methods
    topics = extract_topics_with_counts([reviews], n_topics=n_topics, n_top_words=n_top_words)
    keywords = extract_keywords_with_counts(reviews, keybert_model, top_n=top_keywords)
    noun_phrases = extract_noun_phrases_with_counts(reviews)
    
    # Combine features
    combined_features = list(set(topics + keywords + noun_phrases))
    
    # Calculate sentiment scores and labels
    sentiment_data = calculate_sentiment(reviews, combined_features)
    
    # Step 1: Standardize features (lowercase, strip, etc.)
    standardized_features = {
        feature.lower().strip(): {
            "feature": feature,
            "frequency": freq,
            "sentiment": sentiment_data[feature]["score"],
            "sentiment_label": sentiment_data[feature]["label"]
        }
        for feature, freq in combined_features
    }
    
    # Step 2: Filter and merge redundant features
    filtered_features = {}
    feature_keys = list(standardized_features.keys())
    
    # Sort keys by length (descending) to process longer phrases first
    feature_keys.sort(key=lambda x: (-len(x.split()), x))
    
    for key in feature_keys:
        current_feature = standardized_features[key]
        
        # Look for any shorter features that are substrings of current feature
        for other_key in list(filtered_features.keys()):
            # If other_key is a substring and has same sentiment
            if other_key in key and current_feature["sentiment_label"] == filtered_features[other_key]["sentiment_label"]:
                # Add frequency from shorter feature
                current_feature["frequency"] += filtered_features[other_key]["frequency"]
                # Remove shorter feature
                filtered_features.pop(other_key)
        
        # Add current feature
        filtered_features[key] = current_feature
    
    # Step 3: Handle two-word feature equivalence
    visited = set()
    final_features = []
    
    for key, value in filtered_features.items():
        if key in visited:
            continue
        
        feature_words = key.split()
        if len(feature_words) == 2:
            reversed_key = " ".join(reversed(feature_words))
            if (reversed_key in filtered_features and 
                value["sentiment_label"] == filtered_features[reversed_key]["sentiment_label"]):
                # Merge frequencies and keep the primary ordering
                value["frequency"] += filtered_features[reversed_key]["frequency"]
                visited.add(reversed_key)
        
        final_features.append({
            "feature": value["feature"],
            "frequency": value["frequency"],
            "sentiment": value["sentiment"],
            "sentiment_label": value["sentiment_label"]
        })
    
    return final_features
    
def generate_summary_metrics(reviews_df, product_id=None):
    """Generate comprehensive metrics for product reviews."""
    if product_id:
        # Filter for specific product
        product_reviews = reviews_df[reviews_df['product_id'] == product_id]
    else:
        product_reviews = reviews_df

    # Calculate basic metrics
    total_reviews = product_reviews['review_count'].sum()
    average_rating = product_reviews['avg_rating'].mean()
    
    # Calculate sentiment distribution
    sentiment_counts = pd.DataFrame(list(product_reviews['sentiment_details'].apply(eval)))
    sentiment_counts_melt = pd.melt(
        sentiment_counts.reset_index(), 
        id_vars='index', 
        var_name='sentiment', 
        value_name='count'
    )
    sentiment_counts_melt['percentage'] = (
        sentiment_counts_melt['count'] / sentiment_counts_melt['count'].sum() * 100
    )
    
    # Aggregate helpful votes
    total_helpful_votes = product_reviews['total_helpful_votes'].sum()
    
    # Calculate sentiment score (percentage of positive reviews)
    positive_reviews = sentiment_counts_melt[
        sentiment_counts_melt['sentiment'] == 'positive'
    ]['count'].values[0]
    sentiment_score = positive_reviews / total_reviews * 100

    summary_metrics = {
        'total_reviews': total_reviews,
        'average_rating': average_rating,
        'sentiment_distribution': sentiment_counts_melt[
            ['sentiment', 'count', 'percentage']
        ],
        'total_helpful_votes': total_helpful_votes,
        'sentiment_score': sentiment_score
    }
    
    return summary_metrics

def generate_review_summary(reviews_df, product_id, features_df):
    """Create a human-readable summary of product reviews."""
    # Get summary metrics
    summary_metrics = generate_summary_metrics(reviews_df, product_id)
    
    # Extract top positive and negative features
    positive_features = features_df[
        features_df['Sentiment'] == 'Positive'
    ].sort_values('Mentions', ascending=False)
    negative_features = features_df[
        features_df['Sentiment'] == 'Negative'
    ].sort_values('Mentions', ascending=False)
    
    # Get top features
    top_pos = positive_features.head(2)['Features'].tolist() if len(positive_features) > 0 else None
    top_neg = negative_features.head(2)['Features'].tolist() if len(negative_features) > 0 else None
    
    # Determine overall sentiment
    if summary_metrics['sentiment_score'] >= 70:
        overall_sentiment = 'very positive'
    elif summary_metrics['sentiment_score'] >= 60:
        overall_sentiment = 'positive'
    elif summary_metrics['sentiment_score'] >= 40:
        overall_sentiment = 'mixed'
    elif summary_metrics['sentiment_score'] >= 30:
        overall_sentiment = 'negative'
    else:
        overall_sentiment = 'very negative'
    
    # Generate summary text
    summary = f"Based on the analysis of {summary_metrics['total_reviews']} reviews, "
    summary += f"the overall sentiment towards the product is {overall_sentiment}. "
    
    if top_pos:
        summary += f"The most frequently mentioned positive aspects include {' and '.join(top_pos)}. "
    
    if top_neg:
        summary += f"Common criticisms focus on {' and '.join(top_neg)}. "
    
    return summary

def analyze_product_reviews(reviews_df, product_id=None):
    """Complete analysis pipeline for product reviews."""
    # Step 1: Extract and combine features
    combined_features = extract_combined_features(
        reviews_df,
        sentence_model,
        keybert_model,
        n_topics=3,
        n_top_words=5,
        top_keywords=5
    )
    
    # Step 2: Create features DataFrame
    features_df = pd.DataFrame(combined_features)
    features_df.columns = ['Features', 'Mentions', 'Score', 'Sentiment']
    
    # Step 3: Generate summary metrics
    metrics = generate_summary_metrics(reviews_df, product_id)   

    # Step 4: Generate text summary
    summary = generate_review_summary(reviews_df, product_id, features_df)
    
    # Step 5: Compile complete results
    results = {
        'features': combined_features,
        'metrics': metrics,
        'summary': summary,
        'feature_analysis': {
            'top_positive': features_df[
                features_df['Sentiment'] == 'Positive'
            ].nlargest(5, 'Mentions').to_dict('records'),
            'top_negative': features_df[
                features_df['Sentiment'] == 'Negative'
            ].nlargest(5, 'Mentions').to_dict('records'),
            'feature_distribution': features_df['Sentiment'].value_counts().to_dict()
        }
    }
    
    return results
    