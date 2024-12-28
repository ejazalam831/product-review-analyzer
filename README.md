## Product Review Analysis & Summarization System

### Executive summary

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
The approach involved four comprehensive phases of development and analysis:
1. Data Cleaning & Preprocessing

   a) Data Cleaning Process
   - After performing initial quality assessment and structure validation, selected relevant columns for analysis:
     - ProductId: For product identification
     - Text: Review content
     - Score: Rating (1-5)
     - Helpfulness Votes: Review usefulness
   - Checked for missing values and removed 565 duplicate reviews
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

  
    This resulted in a filtered dataset of 40,289 high-quality reviews by grouping reviews by product for comprehensive analysis to maintain text meaning while removing noise.

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


    This process created a rich feature set capturing both textual content and review metadata for subsequent model training and analysis.

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
  
   d) Model Optimization
   - Implemented hyperparameter optimization using HalvingGridSearchCV
   - Addressed class imbalance through:
     - F1-macro scoring implementation
     - Balanced class weights
     - Stratified sampling
   - Performed cross-validation for robust performance estimation
   - 
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
