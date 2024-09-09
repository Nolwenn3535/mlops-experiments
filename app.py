from flask import Flask, render_template, request
import pickle
import pandas as pd


app = Flask(__name__)
model = pickle.load(open("random_forest_model2.pkl", "rb"))


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

        if prediction == 1:
            prediction_text = "Le client est à risque de défaut de paiement."
        else:
            prediction_text = "Le client n'est pas à risque de défaut de paiement."

        # Renvoyer la page avec le résultat de la prédiction
        return render_template("index.html", prediction_text=prediction_text)
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
