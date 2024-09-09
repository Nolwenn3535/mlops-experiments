from flask import Flask, render_template, request
import pickle
import pandas as pd
import os
from arize.pandas.logger import Client, Schema
from arize.utils.types import ModelTypes, Environments
from dotenv import load_dotenv
import datetime

# Charger les variables d'environnement
load_dotenv()

# Récupérer les clés API d'Arize à partir des variables d'environnement
ARIZE_SPACE_KEY = os.getenv("SPACE_KEY")
ARIZE_API_KEY = os.getenv("API_KEY")

# Initialiser le client Arize avec les clés
arize_client = Client(space_key=ARIZE_SPACE_KEY, api_key=ARIZE_API_KEY)

# Définir le schéma pour les données
schema = Schema(
    prediction_id_column_name="prediction_id",
    timestamp_column_name="timestamp",
    feature_column_names=["credit_lines_outstanding", "loan_amt_outstanding", "total_debt_outstanding", "income", "years_employed", "fico_score"],
    prediction_label_column_name="prediction_label",
    actual_label_column_name="actual_label"
)

# Initialiser l'application Flask
app = Flask(__name__)
model = pickle.load(open("random_forest_model.pkl", "rb"))

def model_pred(features):
    test_data = pd.DataFrame([features])
    prediction = model.predict(test_data)
    return int(prediction[0])

@app.route("/", methods=["GET"])
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        credit_lines_outstanding = int(request.form["lignes de crédit en cours"])
        loan_amt_outstanding = int(request.form["montant du prêt en cours"])
        total_debt_outstanding = int(request.form["dette totale en cours"])
        income = float(request.form["revenu"])
        years_employed = int(request.form["années d'emploi"])
        fico_score = int(request.form["score FICO"])

        # Préparer les données pour la prédiction
        features = {
            'credit_lines_outstanding': credit_lines_outstanding,
            'loan_amt_outstanding': loan_amt_outstanding,
            'total_debt_outstanding': total_debt_outstanding,
            'income': income,
            'years_employed': years_employed,
            'fico_score': fico_score
        }

        # Faire la prédiction avec le modèle chargé
        prediction = model_pred(features)

        # Assumer que tu as des étiquettes réelles disponibles pour l'évaluation
        actual_label = int(request.form.get("actual_label", -1))

        # Créer un horodatage pour la prédiction
        timestamp = pd.Timestamp.now()

        # Créer le dataframe avec les données de la prédiction pour Arize
        data = {
            "prediction_id": [str(timestamp.timestamp())],  # ID unique pour chaque prédiction
            "timestamp": [timestamp],
            "credit_lines_outstanding": [credit_lines_outstanding],
            "loan_amt_outstanding": [loan_amt_outstanding],
            "total_debt_outstanding": [total_debt_outstanding],
            "income": [income],
            "years_employed": [years_employed],
            "fico_score": [fico_score],
            "prediction_label": [prediction],
            "actual_label": [actual_label]
        }
        dataframe = pd.DataFrame(data)

        # Essayer de log les données de prédiction vers Arize
        try:
            response = arize_client.log(
                dataframe=dataframe,
                model_id="Random_Forest_Model",
                model_version="v1",
                model_type=ModelTypes.SCORE_CATEGORICAL,
                environment=Environments.PRODUCTION,
                schema=schema
            )

            if response.status_code != 200:
                print(f"Failed to log data to Arize: {response.text}")
            else:
                print("Successfully logged data to Arize")
        except Exception as e:
            print(f"An error occurred: {e}")

        # Afficher le résultat de la prédiction
        if prediction == 1:
            prediction_text = "Le client est à risque de défaut de paiement."
        else:
            prediction_text = "Le client n'est pas à risque de défaut de paiement."

        return render_template("index.html", prediction_text=prediction_text)
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
