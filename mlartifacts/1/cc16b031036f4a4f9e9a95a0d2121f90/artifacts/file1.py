import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Connect to MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.10,
    random_state=42
)

# Define parameters
max_depth = 10
n_estimators = 5

# Set experiment
mlflow.set_experiment("MLFlow_Exp1")

with mlflow.start_run():

    rf = RandomForestClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=42
    )

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log parameters
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)

    # Log metric
    mlflow.log_metric("accuracy", accuracy)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=wine.target_names,
        yticklabels=wine.target_names
    )

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")

    # Save and log artifact
    plt.savefig("Confusion-matrix.png")
    mlflow.log_artifact("Confusion-matrix.png")

    # Log source code
    mlflow.log_artifact(__file__)

    # Tags
    mlflow.set_tags({
        "Author": "Vikash",
        "Project": "Wine Classification"
    })

    # Log model
    mlflow.sklearn.log_model(
        rf,
        "Random-Forest-Model"
    )

    # Print information
    print("Accuracy:", accuracy)
    print("Run ID:", mlflow.active_run().info.run_id)
    print("Experiment ID:", mlflow.active_run().info.experiment_id)
    print("Tracking URI:", mlflow.get_tracking_uri())