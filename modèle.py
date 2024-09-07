# Importer les bibliothèques nécessaires
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
# Install the following librairies (it is better to create a venv (or conda) virtual environment first and install these librairies in it)
!pip install mlflow
!pip install --upgrade jinja2
!pip install --upgrade Flask
!pip install setuptools
# Charger le fichier CSV
data = pd.read_csv('Loan_Data(1).csv')

# Configurer l'URI de suivi pour MLflow (assurez-vous que le serveur MLflow est en cours d'exécution)
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Utilisez l'URI de votre serveur MLflow
mlflow.set_experiment("Customer_Default_Prediction")

# Préparer les données
X = data.drop(columns=["default"])  # Assurez-vous que 'default' est la colonne cible dans votre CSV
y = data["default"]

# Diviser les données en ensembles d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Définir et tester plusieurs modèles
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
}

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        # Entraîner le modèle
        model.fit(X_train, y_train)
        
        # Faire des prédictions
        y_pred = model.predict(X_val)
        
        # Calculer les métriques
        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred),
            "recall": recall_score(y_val, y_pred),
            "f1_score": f1_score(y_val, y_pred)
        }
        
        # Enregistrer les paramètres et les métriques avec MLflow
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        
        # Enregistrer le modèle avec MLflow
        mlflow.sklearn.log_model(model, artifact_path=f"{model_name}_model")

        # Sauvegarder le modèle au format .pkl
        with open(f'{model_name}_model.pkl', 'wb') as f:
            pickle.dump(model, f)

print("Les modèles ont été entraînés et enregistrés avec succès.")
