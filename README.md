# Predicting the Difficulty of French Text Using AI

**Group OMEGA: Mathias Häberli, Sophie Daya**

## Project Description
This project aims to predict the difficulty level of French sentences using various machine learning models, including Logistic Regression, KNN, Decision Tree, Random Forest, and CamemBERT. Throughout this README and the accompanying GitHub repository, we will explore which model is the most capable of classifying French texts according to their difficulty level. Additionally, after demonstrating and training the best model, we will apply it in a real-world scenario: an application.

We are proud to present the first application from our startup, "LogoRank." We hope this README will not only convince you of our approach but also provide you with valuable insights into text classification. Join us on this exciting journey as we push the boundaries of natural language processing and machine learning!

## Deliverables
- **GitHub:** [GitHub Project Page](https://github.com/matiti5132421/Group_Omega_Mathias_Haberli_Sophie_Daya)
- **Code:** [Jupyter Notebook](https://colab.research.google.com/drive/1qFdVwjp82fv_aWV2Qpf-62F4_Zq41fXt?usp=sharing)
- **Youtube:** [App Presentation Video](https://youtu.be/yourvideo)
- **Kaggle:** [Kaggle Competition](https://www.kaggle.com/competitions/predicting-the-difficulty-of-a-french-text-e4s/overview)
- **Waitlist:** [Waitlist for our App](https://docs.google.com/forms/d/e/1FAIpQLSc-g1LlU-5fFoKpCgv2n0rtrsx3aghzOXvipW8a8PQBskdMQg/viewform)

## Table of Contents
1. [Introduction](#introduction)
2. [Data](#data-presentation)
   * [Data Presentation](#data-presentation)
   * [Data Preparation](#data-preparation)
4. [Models and Methodology](#models-and-methodology)
   - [Logistic Regression](#logistic-regression)
   - [KNN](#knn)
   - [Decision Tree](#decision-tree)
   - [Random Forest](#random-forest)
   - [CamemBERT](#camembert)
5. [Results](#results)
   - [Performance Metrics](#performance-metrics)
   - [Best Model Analysis](#best-model-analysis)
   - [Confusion Matrices](#confusion-matrices)
   - [Examples of Erroneous Predictions](#examples-of-erroneous-predictions)
6. [Additional Analysis](#additional-analysis)
   - [Sentence Length Analysis](#sentence-length-analysis)
   - [POS Tag Analysis](#pos-tag-analysis)
7. [Our App LogoRank](#our-app-logorank)
   - [Principle and functionality](#principle-and-functionality)
   - [Demonstration in a video](#Demonstration-in-a-video)
8. [Conclusion](#conclusion)
9. [References and Participation Report](#references)

## 1. Introduction
To begin with, it is essential to understand that multiple approaches were available to develop our French text difficulty classification model. We experimented with five different machine learning models to achieve the best classification results. These models include:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- CamemBERT from BERT

Each model has its strengths, weaknesses, and specific use cases. Therefore, it was crucial for us to determine which of these models was the most suitable for our final application by directly testing each model on the data we had available.

These machine learning models follow a similar process. They are first trained on labeled data, and their parameters are optimized to achieve the best results on a portion of the data reserved for testing the model. Finally, we can use the trained model on real, unlabeled data.

Without further ado, let's begin with an overview of the datasets we used!

## 2. Data
### Data Presentation
To train our models, we used various datasets. The most important was the `training_data.csv`, which you can find here: [training_data.csv](dataset/training_data.csv). This dataset consists of 4800 French texts organized as follows:

| id  | sentence                                                                 | difficulty |
| --- | ------------------------------------------------------------------------ | ---------- |
| 0   | Les coûts kilométriques réels peuvent diverger...                        | C1         |
| 1   | Le bleu, c'est ma couleur préférée mais je n'a...                        | A1         |
| 2   | Le test de niveau en français est sur le site ...                        | A1         |
| 3   | Est-ce que ton mari est aussi de Boston?                                 | A1         |
| 4   | Dans les écoles de commerce, dans les couloirs...                        | B1         |
| ... | ...                                                                      | ...        |
| 4795| C'est pourquoi, il décida de remplacer les hab...                        | B2         |
| 4796| Il avait une de ces pâleurs splendides qui don...                        | C1         |
| 4797| Et le premier samedi de chaque mois, venez ren...                        | A2         |
| 4798| Les coûts liés à la journalisation n'étant pas...                        | C2         |
| 4799| Sur le sable, la mer haletait de toute la resp...                        | C2         |

The `difficulty` column indicates the difficulty level of the sentences, ranging from A1 to C2, with A1 being the simplest and C2 being the most complex.

For each model, we divided the data into two parts: the training sample to train our models, and the validation sample to test the models and adjust parameters if necessary. We typically used an 80/20 split, with 80% of the data for training and 20% for validation. Other splits are also possible. Concretly, in python code it looks like this:

### Data Preparation

Concretly, in python code it looks like this:

```python
# Separate features and labels
X = training_data_pd['sentence']  # Features (text data)
y = training_data_pd['difficulty']  # Labels (difficulty level)
```
In this step, we first separate our dataset into features and labels. The features (X) are the sentences we want to classify, and the labels (y) are the difficulty levels assigned to each sentence.

```python
# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
```
Since machine learning models work better with numeric data, we need to convert the text labels (difficulty levels) into numerical values. The LabelEncoder is used to transform the difficulty levels into numeric form, making them suitable for training our model.

```python
# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
```
Here, we split our dataset into two parts: training and validation sets. The training set (X_train, y_train) is used to train the model, while the validation set (X_val, y_val) is used to evaluate its performance. We use 80% of the data for training and 20% for validation. The random_state parameter ensures that we get the same split every time we run the code.

```python
# Text Vectorization
vectorizer = TfidfVectorizer()
X_train_transformed = vectorizer.fit_transform(X_train)
X_val_transformed = vectorizer.transform(X_val)
```
To feed our text data into a machine learning model, we need to convert it into a numerical format. The TfidfVectorizer transforms the sentences into a matrix of numerical values, where each value represents the importance of a word in a sentence relative to the entire dataset. We first fit the vectorizer on the training data (fit_transform), and then apply the same transformation to the validation data (transform).

By following these steps, we prepare our text data for machine learning models, ensuring that it is in the right format and properly split into training and validation sets for effective model training and evaluation.

## 3. Models and Methodology

In this chapter, we will apply our various models to the data we have prepared. For each model, we will explore how it works, provide Python code to implement and optimize the model, and conclude by examining key metrics to evaluate which model performs the best. We will delve into the methodology behind each model, detailing the steps and considerations involved in their training and evaluation.

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
