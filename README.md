## Product Review Analysis & Summarization System

### Executive summary
Successfully developed and implemented a comprehensive system to automatically analyze over half a million of unstructured review data across multiple products into actionable business insights. The system combines advanced text analysis with machine learning to help businesses and customers make informed decisions by:

**Identifying Key Features:** Successfully extracted and ranked product features that customers discuss most frequently.

**Analyzing Sentiment:** Developed a nuanced understanding of customer sentiment, going beyond simple positive/negative classifications to provide feature-specific sentiment analysis.

**Generating Insights:** Created clear, actionable summaries that highlight both strengths and areas for improvement in products, with detailed metrics on customer satisfaction.

Key Achievements:

- Successfully cleaned and processed over 568,454 reviews, implementing robust filtering and preprocessing techniques to ensure data quality
- Achieved 93.41% accuracy in sentiment classification using an optimized LinearSVC model
- Identified and ranked key product features using multiple feature extraction methods
- Created an intuitive visualization system for easy interpretation of results

Business Benefits:

**1. Time Efficiency:** Automated analysis of thousands of reviews in minutes

**2. Deep Insights:** Detailed breakdown of product features and associated customer sentiment

**3. Actionable Data:** Clear identification of product strengths and areas for improvement

**4. Customer Understanding:** Better grasp of customer preferences and pain points

The system demonstrates strong potential for helping businesses make data-driven decisions about product improvements while helping customers make informed purchasing decisions. An efficient analysis processing pipeline that successfully bridges the gap between raw customer feedback and actionable business insights, providing a scalable solution for modern e-commerce challenges.

### Rationale
With hundreds or thousands of reviews per product, it's nearly impossible for businesses or customers to read everything. Important feedback can get lost, and businesses can miss opportunities to improve their products. 

A system that can successfully identify what features customers talk about most, how they feel about them, and provide easy-to-understand summaries can help both businesses and customers make better decisions.

### Research Question
Can we develop an effective system to automatically analyze large volumes of product reviews to:
1. Identify key product features customers talk about most
2. Determine if feedback about these features is positive or negative
3. Create easy-to-understand summaries that help both businesses and customers

### Data Sources
The dataset used in this project is Amazon Product Reviews sourced from Kaggle and can be accessed by clicking [here](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews/data).

It contains over 568,000 consumer reviews for various products, including:
- Customer ratings (1-5 stars)
- Written review text
- Helpfulness votes from other customers
- Product identifiers and other metadata

### Methodology
The approach involved four comprehensive phases of development and analysis, each building upon the previous phase to create a comprehensive review analysis system:
1. Data Cleaning & Preprocessing

   a) Data Cleaning Process
   - After performing initial quality assessment and structure validation, selected relevant columns for analysis:
     - ProductId: For product identification
     - Text: Review content
     - Score: Rating (1-5)
     - Helpfulness Votes: Review usefulness
   - Checked for missing values and removed 565 duplicate reviews
   - Grouped reviews by product for comprehensive analysis
   - Implemented two-sided filtering approach:
     - Minimum threshold: Removed products with less than 2 reviews
     - Maximum threshold: Excluded products above 95th percentile

   b) Text Preprocessing Pipeline
   - Implemented comprehensive text cleaning:
     - Converted all text to lowercase for consistency
     - Expanded contractions (e.g., "don't" to "do not") for better analysis
     - Removed special characters, irrelevant symbols and stop words
     - Tokenized text into individual words
     - Applied lemmatization to standardize word forms

  
    This resulted in a filtered dataset of 40,289 high-quality product reviews to maintain text meaning while removing noise.
   
   With clean, standardized data in place, next focus was on converting the text reviews into a format that machines could understand and analyze effectively.

2. Feature Engineering & Representation
   
   Developed a comprehensive feature engineering pipeline combining three types of features:
     - TF-IDF vectorization (5000 features) to capture important terms
     - Word2Vec embeddings (100 features) for semantic relationships
     - Metadata features (3 features) from review statistics

   a) Feature Processing
   - TF-IDF Implementation:
     - Applied L2 normalization to handle varying review lengths
     - Included unigram and bigram support
     - Set maximum features to 5000 for computational efficiency

   - Word2Vec Configuration:
     - Trained on review text with context window of 5 words
     - Set minimum word frequency threshold of 2
     - Generated 100-dimensional word vectors
  
   b) Combined Feature Representation
   - Merged all feature types into unified representation
   - Applied feature standardization
   - Created final feature matrix of 5,103 dimensions


    Having created a rich feature set capturing both textual content and review metadata, then developed and tested several machine learning models to effectively process these features and generate insights.

3. Model Development & Evaluation

   a) Model Selection & Implementation
   - Implemented three different machine learning models:
     - Stochastic Gradient Descent (SGD): For efficient processing of large-scale text data
     - Linear Support Vector Classification: For handling high-dimensional feature space
     - Multi-Layer Perceptron (MLP): For capturing complex non-linear patterns

   b) Baseline Model Development
   - Implemented comprehensive evaluation framework based on multiple criteria:
     - Overall accuracy and classification performance
     - Training time and computational efficiency
     - Class-wise precision, recall, and F1-scores
   - Established baseline performance with default parameters
    
   c) Model Optimization
   - Implemented hyperparameter optimization using HalvingGridSearchCV
   - Addressed class imbalance through:
     - F1-macro scoring implementation
     - Balanced class weights
     - Stratified sampling
   - Performed cross-validation for robust performance estimation
  
   d) Performance Results
   - Base Models:
     - SGD: 88.56% accuracy (375.60s training)
     - LinearSVC: 89.85% accuracy (1852.29s training)
     - MLP: 84.45% accuracy (266.34s training)
   - Improved Models:
     - SGD: 89.94% accuracy with optimized parameters
     - LinearSVC: 93.41% accuracy with balanced class weights
     - MLP: 88.91% accuracy with tuned architecture

   While our models showed strong performance in classifying reviews, we needed to extract specific insights about product features and customer sentiment. This led to our final phase of detailed feature analysis.
   
4. Feature Extraction & Analysis

   a) Feature Extraction Pipeline
   - Implemented multiple extraction methods:
     - Topic modeling (LDA) to identify key themes and patterns
     - KeyBERT for semantic keyword extraction
     - spaCy for noun phrase identification and extraction
     - VADER for sentiment analysis

   b) Feature Processing & Integration
   - Consolidated features from different extraction methods
   - Applied deduplication and similarity grouping
   - Created unified feature representation including:
     - Feature frequency metrics
     - Sentiment scores
     - Feature importance rankings
   - Generated product-level feature summaries
    
   c) Sentiment Analysis
   - Implemented sentence-level sentiment analysis
   - Calculated feature-specific sentiment scores
   - Created sentiment distribution profiles
  
   d) Visualization & Reporting
   - Developed interactive visualization components for dashboard:
     - Key feature importance plots
     - Sentiment distribution charts
     - Generated overall product summary metrics for dashboard

### Results
1. Model Performance

- Best Overall Accuracy: 89.91% (LinearSVC)
- Best Minority Class Handling: MLP (F1-score: 0.64)
- Most Efficient: SGD Classifier (79.16s training time)


2. Key Findings

- Successfully handled class imbalance through balanced weights
- Achieved robust feature extraction using combined TF-IDF and Word2Vec
- Demonstrated effective trade-off between accuracy and processing time
- Reliable performance across different product categories


#### Next steps
What suggestions do you have for next steps?

#### Outline of project

- [Link to notebook 1](https://github.com/ejazalam831/product-review-analyzer/blob/main/01_data_cleaning_and_prep.ipynb)
- [Link to notebook 2](https://github.com/ejazalam831/product-review-analyzer/blob/main/02_Feature_Engineering_Implementation.ipynb)
- [Link to notebook 3](https://github.com/ejazalam831/product-review-analyzer/blob/main/03_model_implement_and_eval.ipynb)


##### Contact and Further Information
