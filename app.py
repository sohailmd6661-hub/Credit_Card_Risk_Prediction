from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load model and scaler
with open("Model.pkl", "rb") as f:
    model = pickle.load(f)

with open("standard_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Collect form inputs
            features = [
                float(request.form["RevolvingUtilizationOfUnsecuredLines_trim"]),
                float(request.form["Age"]),
                float(request.form["NumberOfOpenCreditLinesAndLoans"]),
                float(request.form["NumberRealEstateLoansOrLines"]),
                float(request.form["MonthlyIncome_replaced"]),
                float(request.form["NumberOfDependents_replaced_trim"]),
                float(request.form["Gender_Male"]),
                float(request.form["Region_East"]),
                float(request.form["Region_North"]),
                float(request.form["Region_South"]),
                float(request.form["Region_West"]),
                float(request.form["Rented_OwnHouse_con"]),
                float(request.form["Occupation_con"]),
                float(request.form["Education_con"])
            ]

            # Convert to numpy array
            features_array = np.array([features])

            # Scale features
            features_scaled = scaler.transform(features_array)

            # Make prediction
            prediction = model.predict(features_scaled)[0]
            if prediction == 0:
                return render_template("index.html", prediction='Bad Customer')
            else:
                return render_template("index.html", prediction="Good Customer")

        except Exception as e:
            # If an error occurs during POST, render the template with the error message
            prediction = f"Error: {str(e)}"
            return render_template("index.html", prediction=prediction) # <--- Added return here for error handling

    # This handles the initial GET request (when the page is loaded)
    # and any scenario where the POST block didn't execute or encountered an error.
    return render_template("index.html", prediction=prediction) # <--- ADDED REQUIRED RETURN STATEMENT

if __name__ == "__main__":
    app.run(debug=True)