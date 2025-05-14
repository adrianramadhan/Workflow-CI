import sys
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss, confusion_matrix
)

if __name__ == "__main__":
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    n_estimators = int(sys.argv[3])
    max_depth = int(sys.argv[4])

    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)
    X_train, y_train = df_train.drop("mental_health_risk", axis=1), df_train["mental_health_risk"]
    X_test,  y_test  = df_test.drop("mental_health_risk", axis=1), df_test["mental_health_risk"]

    # Start run manually — penting saat tidak pakai backend CLI run
    with mlflow.start_run():  # gunakan ini saat running dari script biasa
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "n_features": X_train.shape[1],
            "n_classes": len(y_train.unique())
        })

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)

        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average='weighted')
        recall = recall_score(y_test, preds, average='weighted')
        f1 = f1_score(y_test, preds, average='weighted')
        auc = roc_auc_score(y_test, proba, multi_class='ovr')
        loss = log_loss(y_test, proba)
        cm = confusion_matrix(y_test, preds)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("log_loss", loss)
        mlflow.log_metric("true_positive", cm[1][1] if cm.shape[0] > 1 else 0)
        mlflow.log_metric("true_negative", cm[0][0])
        mlflow.log_metric("false_positive", cm[0][1] if cm.shape[1] > 1 else 0)
        mlflow.log_metric("false_negative", cm[1][0] if cm.shape[0] > 1 else 0)

        mlflow.sklearn.log_model(model, "model")
        print(f"Run finished. Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")
