import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns


def train_advanced():
    dagshub.init(
        repo_owner='rstiannr',
        repo_name='Retail-Product-Classification',
        mlflow=True,
    )

    mlflow.set_experiment("Retail_Classification_Advanced")

    df = pd.read_csv('data_training_final.csv')
    X = df.drop(
        columns=['StockCode', 'Description', 'Label', 'Avg_Revenue']
    )
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    param_grid = {
        'max_depth': [5, 10],
        'criterion': ['gini', 'entropy'],
    }
    grid_search = GridSearchCV(
        DecisionTreeClassifier(
            class_weight='balanced', random_state=42
        ),
        param_grid,
        cv=3,
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    with mlflow.start_run(run_name="DecisionTree_Advanced"):
        mlflow.log_params(grid_search.best_params_)

        y_pred = best_model.predict(X_test)

        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric(
            "f1_weighted", f1_score(y_test, y_pred, average='weighted')
        )

        report = classification_report(y_test, y_pred)
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        mlflow.sklearn.log_model(best_model, "advanced_model")

        print(
            "Model Advanced berhasil dilatih dan dikirim ke DagsHub."
        )


if __name__ == "__main__":
    train_advanced()