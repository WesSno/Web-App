from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sympy import Symbol, solve, Eq

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/step_test.html", methods=["POST", "GET"])
def step_test():

    if request.method == "POST":
        try:
            SWL = float(request.form["SWL"])
            Pump_settings = float(request.form["Pump_settings"])
            Buffer_ = float(request.form["Buffer"])
        except ValueError:
            return render_template("step_test.html", failure="Please enter values in the value fields")
        else:
            s_max = Pump_settings - SWL - Buffer_

            filename = request.form["Filename"]
            sheet_name = request.form["Sheetname"]

            if len(filename) > 0 and len(sheet_name) > 0:
                try:
                    data = pd.read_excel(filename, sheet_name=sheet_name)
                except FileNotFoundError:
                    return render_template("step_test.html", failure="File Not Found")
                else:
                    df = data.dropna(subset=(["s", "Q"]))
                    q_values = np.array(df["Q"].tolist())
                    s_values = np.array(df["s"].tolist())

                    # Plotting the graph
                    plt.scatter(q_values, s_values)
                    plt.xlabel("Q(L/min)")
                    plt.ylabel("s(m)")


                    # Plotting The TrendLine
                    popt, popc = curve_fit(fxn, q_values, s_values)
                    s_hat = fxn(q_values, *popt)
                    plt.plot(q_values, s_hat, "r--", lw=1)
                    text = f"$y={popt[0]:0.4f}\;x^2{popt[1]:+0.4f}\;x$\n$R^2 = {r2_score(s_values, s_hat):0.3f}$"
                    plt.gca().text(0.05, 0.95, text, transform=plt.gca().transAxes,
                                   fontsize=14, verticalalignment='top')
                    plt.grid()

                    #Calculating for Qm using the symbols module
                    Qm = Symbol("Qm")
                    equation = Eq(s_max, popt[0] * Qm ** 2 + popt[1] * Qm)
                    roots = solve(equation, Qm)
                    positive_roots = roots[1]

                    Q_ = f"Q = {round(positive_roots, 1)}L/mins"

                    q_max = positive_roots * 1.44

                    Q_max = f"Qm = {round(q_max, 1)}m\u00b3/day"

                    output_filename = "image.png"
                    output_path = os.path.join('static', output_filename)

                    image = plt.savefig(output_path)

                    return render_template("step_test.html", image_filename=output_filename, Q=Q_, Q_max=Q_max, image=image)
    else:
        return render_template("step_test.html")

@app.route("/byield.html", methods=["GET", "POST"])
def byield():

    if request.method == "POST":
        try:
            SWL = float(request.form["SWL"])
            Last_DWL = float(request.form["Last DWL"])
            P_settings = float(request.form["P_setting"])
            Q_test = float(request.form["Qtest"])
            Buffer_ = float(request.form["Buffer"])
        except ValueError:
            return render_template("byield.html", failure="Please make sure all fields are values")
        else:
            S_max = P_settings - SWL - Buffer_

            S_test =  Last_DWL - SWL

            Q_max = (Q_test / S_test ) * S_max
            values = [Q_max, S_max, S_test]
            return render_template("byield.html", Q=Q_max, S_t=S_test, S_m=S_max)
    else:
        return render_template("byield.html")

@app.route("/lm_webApp.html")
def lm_webApp():
    return render_template("lm_webApp.html")

@app.route("/lm_steptest.html")
def lm_steptest():
    return render_template("lm_steptest.html")

@app.route("/lm_maxpy.html")
def lm_maxpy():
    return render_template("lm_maxpy.html")

@app.route("/contact.html")
def contact():
    return render_template("contact.html")

def fxn(x, a, b):
    return a * x ** 2 + b * x

if __name__ == "__main__":
    app.run(debug=True)
