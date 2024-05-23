
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import randint

def decision_tree_model(training_data_path):
    # Load the training data
    training_data_pd = pd.read_csv(training_data_path)

    # Separate features and labels
    X = training_data_pd['sentence']  # Features (text data)
    y = training_data_pd['difficulty']  # Labels (difficulty level)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Text Vectorization
    vectorizer = TfidfVectorizer()
    X_train_transformed = vectorizer.fit_transform(X_train)
    X_val_transformed = vectorizer.transform(X_val)

    # Decision Tree Model
    decision_tree = DecisionTreeClassifier()

    # Hyperparameter tuning using RandomizedSearchCV
    param_dist = {
        'max_depth': [None, 10, 20, 30, 40, 50, 60],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random']
    }

    random_search = RandomizedSearchCV(decision_tree, param_distributions=param_dist, n_iter=100, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)
    random_search.fit(X_train_transformed, y_train)

    # Best Decision Tree model
    best_decision_tree = random_search.best_estimator_

    # Predictions and evaluation
    y_pred = best_decision_tree.predict(X_val_transformed)
    report = classification_report(y_val, y_pred, target_names=label_encoder.classes_, output_dict=True)

    # Create a DataFrame to display class-wise results
    class_results_df = pd.DataFrame(report).transpose()

    # Exclude the last three rows which include accuracy, macro avg, and weighted avg
    class_results_df = class_results_df.drop(columns=['support']).iloc[:-3]

    # Creating a DataFrame to display overall results
    overall_results_df = pd.DataFrame({
        "Precision": [report['weighted avg']['precision']],
        "Recall": [report['weighted avg']['recall']],
        "F1-Score": [report['weighted avg']['f1-score']],
        "Accuracy": [accuracy_score(y_val, y_pred)]
    }, index=["Decision Tree"])

    # Displaying the DataFrames
    print("Decision Tree Model Evaluation (Class-wise)")
    print(class_results_df)
    print("\nDecision Tree Model Evaluation (Overall)")
    print(overall_results_df)

    return best_decision_tree, vectorize
