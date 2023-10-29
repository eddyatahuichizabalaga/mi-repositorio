from sklearnex import patch_sklearn
patch_sklearn()

import argparse
from pathlib import Path
from uuid import uuid4
from datetime import datetime
import os
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import mlflow
import mlflow.sklearn

def main():
    # obtener parámetros:
    parser = argparse.ArgumentParser("train")
    parser.add_argument("--dataset_path", type=str, help="File path to training data")
    parser.add_argument("--weights", type=str, help="Weights: uniform or distance")
    parser.add_argument("--metric", type=str, help="Metric: euclidean or manhattan or minkowski")

    args = parser.parse_args()

    mlflow.start_run()
    mlflow.sklearn.autolog()

    lines = [
        f"Data file: {args.dataset_path}",
        f"Weights: {args.weights}",
        f"Metric: {args.metric}",
    ]

    print("Parametros: ...")

    # imprimir parámetros:
    for line in lines:
        print(line)

    # log en mlflow
    mlflow.log_param('Data file', str(args.dataset_path))
    mlflow.log_param('Weights', str(args.weights))
    mlflow.log_param('Metric', str(args.metric))

    # leer dataset
    data = pd.read_csv(args.dataset_path)

    # separar el ds
    X = data[data.columns[:-1]] # todas las columnas menos la ultima
    y = data.iloc[:, -1] # target: DRK_YN, si toma o no alcohol

    # separar el ds en train/tesst
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # entrenar modelo
    dt = KNeighborsClassifier(metric=args.metric, weights=args.weights)
    dt.fit(X_train,y_train)

    # evaluar el modelo
    y_pred = dt.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(confusion_matrix(y_test, y_pred))

    # imprimir metrica en mlflow
    mlflow.log_metric('accuracy', float(f1))

    registered_model_name="sklearn-KNeighborsClassifier"

    print("Registrar el modelo via MLFlow")
    mlflow.sklearn.log_model(
        sk_model=dt,
        registered_model_name=registered_model_name,
        artifact_path=registered_model_name
    )

    print("Guardar el modelo via MLFlow")
    mlflow.sklearn.save_model(
        sk_model=dt,
        path=os.path.join(registered_model_name, "trained_model"),
    )

    mlflow.end_run()

if __name__ == '__main__':
    main()
