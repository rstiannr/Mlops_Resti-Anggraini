import pandas as pd
import mlflow
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


def train_basic():
    dagshub.init(
        repo_owner='rstiannr',
        repo_name='Retail-Product-Classification',
        mlflow=True
    )

    mlflow.set_experiment("Retail_Product_Classification_Basic")

    mlflow.autolog()

    df = pd.read_csv('data_training_final.csv')

    drop_cols = ['StockCode', 'Description', 'Label', 'Avg_Revenue']
    X = df.drop(columns=drop_cols)
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run(run_name="DecisionTree_Autolog"):
        dt_model = DecisionTreeClassifier(
            random_state=42,
            class_weight='balanced',
            max_depth=5
        )

        dt_model.fit(X_train, y_train)

        predictions = dt_model.predict(X_test)
        report = classification_report(y_test, predictions)
        print("Classification Report:\n", report)


if __name__ == "__main__":
    train_basic()