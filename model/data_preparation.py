
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

def data_preparation(training_data_path):
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

    return X_train_transformed, X_val_transformed, y_train, y_val, label_encoder, vectorizer

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare text data for machine learning.")
    parser.add_argument('training_data_path', type=str, help="Path to the training data CSV file.")

    args = parser.parse_args()

    X_train_transformed, X_val_transformed, y_train, y_val, label_encoder, vectorizer = data_preparation(args.training_data_path)

    print("Data preparation complete.")
    print("X_train_transformed shape:", X_train_transformed.shape)
    print("X_val_transformed shape:", X_val_transformed.shape)
    print("y_train shape:", y_train.shape)
    print("y_val shape:", y_val.shape)
