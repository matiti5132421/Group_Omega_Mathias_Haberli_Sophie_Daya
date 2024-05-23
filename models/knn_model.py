
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

def knn_model(training_data_path):
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

    # KNN Model
    knn = KNeighborsClassifier()

    # Hyperparameter tuning
    param_grid = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_transformed, y_train)

    # Best KNN model
    best_knn = grid_search.best_estimator_

    # Predictions and evaluation
    y_pred = best_knn.predict(X_val_transformed)
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
    }, index=["KNN"])

    # Displaying the DataFrames
    print("KNN Model Evaluation (Class-wise)")
    print(class_results_df)
    print("\nKNN Model Evaluation (Overall)")
    print(overall_results_df)

    return best_knn, vectorizer, label_encoder

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and evaluate a KNN model.")
    parser.add_argument('training_data_path', type=str, help="Path to the training data CSV file.")

    args = parser.parse_args()

    best_knn, vectorizer, label_encoder = knn_model(args.training_data_path)

    print("Model training and evaluation complete.")
