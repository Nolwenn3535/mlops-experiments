#pip install mlflow

#pip install --upgrade jinja2
#pip install --upgrade Flask
#pip install setuptools
#  Importer les bibliothèques nécessaires
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

# Charger le fichier CSV
data = pd.read_csv('Loan_Data(1).csv')

# Configurer l'URI de suivi pour MLflow (assurez-vous que le serveur MLflow est en cours d'exécution)
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Utilisez l'URI de votre serveur MLflow
mlflow.set_experiment("Customer_Default_Prediction")

# Préparer les données
X = data.drop(columns=["default"])  # Assurez-vous que 'default' est la colonne cible dans votre CSV
y = data["default"]
