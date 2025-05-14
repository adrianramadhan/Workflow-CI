import sys
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    n_estimators = int(sys.argv[3])
    max_depth = int(sys.argv[4])

    mlflow.set_experiment("MentalHealth_CI_Project")

    with mlflow.start_run():
        df_train = pd.read_csv(train_path)
        df_test  = pd.read_csv(test_path)
        X_train, y_train = df_train.drop("mental_health_risk", axis=1), df_train["mental_health_risk"]
        X_test,  y_test  = df_test.drop("mental_health_risk", axis=1), df_test["mental_health_risk"]

        mlflow.log_params({"n_estimators": n_estimators, "max_depth": max_depth})
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("test_accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        print(f"Run finished. Test accuracy: {acc:.4f}")
