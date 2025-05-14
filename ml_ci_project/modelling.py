import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_path", type=str, required=True)
    p.add_argument("--test_path", type=str, required=True)
    p.add_argument("--n_estimators", type=int, default=100)
    p.add_argument("--max_depth", type=int, default=None)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    mlflow.set_experiment("MentalHealth_CI_Project")

    # start run
    with mlflow.start_run():
        # load data
        df_train = pd.read_csv(args.train_path)
        df_test  = pd.read_csv(args.test_path)
        X_train, y_train = df_train.drop("mental_health_risk", axis=1), df_train["mental_health_risk"]
        X_test,  y_test  = df_test.drop("mental_health_risk", axis=1), df_test["mental_health_risk"]

        # train
        mlflow.log_params({"n_estimators": args.n_estimators, "max_depth": args.max_depth})
        model = RandomForestClassifier(n_estimators=args.n_estimators,
                                       max_depth=args.max_depth,
                                       random_state=42)
        model.fit(X_train, y_train)

        # eval
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("test_accuracy", acc)

        # save model artifact
        mlflow.sklearn.log_model(model, "model")
        print(f"Run finished. Test accuracy: {acc:.4f}")
