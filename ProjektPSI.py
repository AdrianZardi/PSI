import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import csv

# Load data
data_path = os.path.join(os.getcwd(), 'updated_pollution_dataset.csv')
data = pd.read_csv(data_path)

# Initialize results storage
results_file = "model_results.csv"
with open(results_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Model", "Accuracy", "Parameters"])

# Define main loop
while True:
    # Ask user for train-test split
    train_size = float(input("Enter the percentage of data for training (e.g., 0.8 for 80%): "))
    test_size = 1 - train_size

    # Split data
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]   # Target label
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Allow user to set parameters for each model
    print("Configure model parameters:")

    # Random Forest
    rf_trees = int(input("Random Forest - Number of trees (e.g., 100): "))
    rf_max_depth = int(input("Random Forest - Max depth (e.g., 10, enter 0 for no limit): "))
    rf_max_depth = None if rf_max_depth == 0 else rf_max_depth
    rf_min_samples_split = int(input("Random Forest - Min samples split (e.g., 2): "))
    rf_min_samples_leaf = int(input("Random Forest - Min samples leaf (e.g., 1): "))
    rf_model = RandomForestClassifier(
        n_estimators=rf_trees,
        max_depth=rf_max_depth,
        min_samples_split=rf_min_samples_split,
        min_samples_leaf=rf_min_samples_leaf,
        random_state=42
    )

    # Support Vector Machine
    svm_kernel = input("SVM - Kernel type (e.g., linear, rbf): ")
    svm_C = float(input("SVM - Regularization parameter C (e.g., 1.0): "))
    svm_gamma = input("SVM - Gamma (e.g., scale, auto, or a float value): ")
    svm_model = SVC(kernel=svm_kernel, C=svm_C, gamma=svm_gamma, random_state=42)

    # K-Nearest Neighbors
    knn_neighbors = int(input("KNN - Number of neighbors (e.g., 5): "))
    knn_algorithm = input("KNN - Algorithm (e.g., auto, ball_tree, kd_tree, brute): ")
    knn_metric = input("KNN - Metric (e.g., minkowski, euclidean, manhattan): ")
    knn_model = KNeighborsClassifier(
        n_neighbors=knn_neighbors,
        algorithm=knn_algorithm,
        metric=knn_metric
    )

    models = {
        "Random Forest": rf_model,
        "Support Vector Machine": svm_model,
        "K-Nearest Neighbors": knn_model
    }

    results = {}

    # Train and evaluate models
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[model_name] = {
            "Accuracy": accuracy,
            "Parameters": model.get_params()
        }

        # Save results to file
        with open(results_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([model_name, accuracy, model.get_params()])

    # Display results
    for model_name, result in results.items():
        print(f"Model: {model_name}")
        print(f"Accuracy: {result['Accuracy']:.2f}")
        print("Parameters:", result['Parameters'])
        print("-")

    # Allow user to classify new data
    while True:
        user_input = input("Do you want to classify new data? (yes/no): ").strip().lower()
        if user_input == "no":
            break
        elif user_input == "yes":
            print("Enter feature values (e.g., " + ", ".join(map(str, X.columns)) + "):")
            user_data = [float(input(f"{feature}: ")) for feature in X.columns]
            user_df = pd.DataFrame([user_data], columns=X.columns)
            best_model_name = max(results, key=lambda k: results[k]['Accuracy'])
            best_model = models[best_model_name]
            prediction = best_model.predict(user_df)
            print(f"Predicted class for the input data: {prediction[0]}")
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

    # Check if user wants to continue or exit
    continue_input = input("Do you want to test more models or parameters? (yes/no): ").strip().lower()
    if continue_input == "no":
        # Summarize the best results for each algorithm
        print("\nSummary of best results:")
        summary_data = pd.read_csv(results_file)
        for model_name in models.keys():
            model_data = summary_data[summary_data["Model"] == model_name]
            if not model_data.empty:
                best_row = model_data.loc[model_data["Accuracy"].idxmax()]
                print(f"Best {model_name}: Accuracy = {best_row['Accuracy']:.2f}, Parameters = {best_row['Parameters']}")
        print(f"Results saved to {results_file}. Exiting program.")
        break
