### Product Review Analysis & Summarization System

#### Executive summary

#### Rationale
With hundreds or thousands of reviews per product, it's nearly impossible for businesses or customers to read everything. Important feedback can get lost, and businesses can miss opportunities to improve their products. 

A system that can successfully identify what features customers talk about most, how they feel about them, and provide easy-to-understand summaries can help both businesses and customers make better decisions.

#### Research Question
Can we develop an effective system to automatically analyze large volumes of product reviews to:
1. Identify key product features customers talk about most
2. Determine if feedback about these features is positive or negative
3. Create easy-to-understand summaries that help both businesses and customers

#### Data Sources
The dataset used in this project is Amazon Product Reviews sourced from Kaggle and can be accessed by clicking [here](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews/data).

It contains over 568,000 consumer reviews for various products, including:
- Customer ratings (1-5 stars)
- Written review text
- Helpfulness votes from other customers
- Product identifiers and other metadata

#### Methodology
What methods are you using to answer the question?

#### Results
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
