# Predicting the Difficulty of French Text Using AI

**Group OMEGA: Mathias Häberli, Sophie Daya**

## Project Description
This project aims to predict the difficulty level of French sentences using various machine learning models, including Logistic Regression, KNN, Decision Tree, Random Forest, and CamemBERT. Throughout this README and the accompanying GitHub repository, we will explore which model is the most capable of classifying French texts according to their difficulty level. Additionally, after demonstrating and training the best model, we will apply it in a real-world scenario: an application.

We are proud to present the first application from our startup, "LogoRank." We hope this README will not only convince you of our approach but also provide you with valuable insights into text classification. Join us on this exciting journey as we push the boundaries of natural language processing and machine learning!

## Deliverables
- **GitHub Link:** [GitHub Project Page](https://github.com/matiti5132421/Group_Omega_Mathias_Haberli_Sophie_Daya)
- **Code Link:** [Jupyter Notebook](https://colab.research.google.com/drive/1qFdVwjp82fv_aWV2Qpf-62F4_Zq41fXt?usp=sharing)
- **Youtube Link:** [App Presentation Video](https://youtu.be/yourvideo)
- **Kaggle Link:** [Kaggle Competition](https://www.kaggle.com/competitions/predicting-the-difficulty-of-a-french-text-e4s/overview)
- **Waitlist Link:** [Waitlist for our App](https://docs.google.com/forms/d/e/1FAIpQLSc-g1LlU-5fFoKpCgv2n0rtrsx3aghzOXvipW8a8PQBskdMQg/viewform)

## Table of Contents
1. [Introduction](#introduction)
2. [Data Presentation](#data-presentation)
3. [Models and Methodology](#models-and-methodology)
   - [Logistic Regression](#logistic-regression)
   - [KNN](#knn)
   - [Decision Tree](#decision-tree)
   - [Random Forest](#random-forest)
   - [CamemBERT](#camembert)
4. [Results](#results)
   - [Performance Metrics](#performance-metrics)
   - [Best Model Analysis](#best-model-analysis)
   - [Confusion Matrices](#confusion-matrices)
   - [Examples of Erroneous Predictions](#examples-of-erroneous-predictions)
5. [Additional Analysis](#additional-analysis)
   - [Sentence Length Analysis](#sentence-length-analysis)
   - [POS Tag Analysis](#pos-tag-analysis)
6. [Our App LogoRank](#our-app-logorank)
   - [Principle and functionality](#principle-and-functionality)
   - [Demonstration in a video](#Demonstration-in-a-video)
7. [Conclusion](#conclusion)
8. [References and Participation Report](#references)

## Introduction
To begin with, it is essential to understand that multiple approaches were available to develop our French text difficulty classification model. We experimented with five different machine learning models to achieve the best classification results. These models include:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- CamemBERT from BERT

Each model has its strengths, weaknesses, and specific use cases. Therefore, it was crucial for us to determine which of these models was the most suitable for our final application by directly testing each model on the data we had available.

These machine learning models follow a similar process. They are first trained on labeled data, and their parameters are optimized to achieve the best results on a portion of the data reserved for testing the model. Finally, we can use the trained model on real, unlabeled data.

Without further ado, let's begin with an overview of the datasets we used!


## Models and Methodology
### Logistic Regression
Description of the model, methodology, and hyper-parameter optimization details.

### KNN
Description of the model, methodology, and hyper-parameter optimization details.

### Decision Tree
Description of the model, methodology, and hyper-parameter optimization details.

### Random Forest
Description of the model, methodology, and hyper-parameter optimization details.

### CamemBERT
Description of the model, methodology, and hyper-parameter optimization details.

## Results
### Performance Metrics
| Metric        | Logistic Regression | KNN  | Decision Tree | Random Forest | CamemBERT |
|---------------|---------------------|------|---------------|---------------|-----------|
| Precision     |                     |      |               |               |           |
| Recall        |                     |      |               |               |           |
| F1-score      |                     |      |               |               |           |
| Accuracy      |                     |      |               |               |           |

### Best Model Analysis
Identify the best model and explain why it is the best model.

### Confusion Matrices
![Confusion Matrix for Logistic Regression](images/logistic_regression_confusion_matrix.png)
![Confusion Matrix for KNN](images/knn_confusion_matrix.png)
![Confusion Matrix for Decision Tree](images/decision_tree_confusion_matrix.png)
![Confusion Matrix for Random Forest](images/random_forest_confusion_matrix.png)
![Confusion Matrix for CamemBERT](images/camembert_confusion_matrix.png)

### Examples of Erroneous Predictions
Show examples of erroneous predictions and analyze the errors.

## Additional Analysis
### Sentence Length Analysis
![Distribution of Sentence Lengths in Erroneous Predictions](images/erroneous_predictions_length_distribution.png)
![Distribution of Sentence Lengths in Training Data](images/training_data_length_distribution.png)

### POS Tag Analysis
![POS Tag Frequency in Training Data](images/training_data_pos_tag_frequency.png)
![POS Tag Frequency in Erroneous Predictions](images/erroneous_predictions_pos_tag_frequency.png)

## Our App LogoRank

## Conclusion
Summarize the findings, model performance, and insights from the additional analyses. Discuss potential improvements and future work.

## References
- List of references and resources.
