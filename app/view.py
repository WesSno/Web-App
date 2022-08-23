from app import app
from flask import render_template, request, send_from_directory, abort, session, redirect, url_for
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sympy import Symbol, solve, Eq
from werkzeug.utils import secure_filename

plt.switch_backend('agg')

@app.route("/")
def home():
    print(app.config)
    return render_template("home.html")

def allowed_image(filename):
    if not "." in filename:
        return False
    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False

@app.route("/step_test.html", methods=["POST", "GET"])
def step_test():
    download = False

    if request.method == "POST":
        try:
            SWL = float(request.form["SWL"])
            Pump_settings = float(request.form["Pump_settings"])
            Buffer_ = float(request.form["Buffer"])
        except ValueError:
            return render_template("step_test.html", failure="Please enter values in the value fields")
        else:
            s_max = Pump_settings - SWL - Buffer_
            sheet_name = request.form["Sheetname"]
            file = request.files["Filename"]

            if len(sheet_name) > 0 and len(file.filename) > 0:
                try:
                    if not allowed_image(file.filename):
                        return render_template("step_test.html", failure="Please choose the correct file type")


                    filename = secure_filename(file.filename)
                    file_path = os.path.join(app.config["CLIENT_UPLOAD"], filename)
                    file.save(file_path)
                    session["FILENAME"] = file.filename
                    time.sleep(2)

                    data = pd.read_excel(file_path, sheet_name=sheet_name)
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
                    plt.gca().text(0.05, 0.95, text, transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
                    plt.grid()

                    #Calculating for Qm using the symbols module
                    Qm = Symbol("Qm")
                    equation = Eq(s_max, popt[0] * Qm ** 2 + popt[1] * Qm)
                    roots = solve(equation, Qm)
                    positive_roots = roots[1]

                    Q_ = str(round(positive_roots, 1))+"L/mins"

                    q_max = positive_roots * 1.44

                    Q_max = str(round(q_max, 1)) + "m\u00b3/day"

                    output_filename = "plot.png"
                    output_path = os.path.join(app.config["CLIENT_UPLOAD"], output_filename)

                    plt.savefig(output_path)

                    time.sleep(2)
                    output_path = output_path.split("/", 1)[1] 
                    
                    return render_template("step_test.html", image_filename=output_path, Q=Q_, Q_max=Q_max, width=500, height=300, download=download)
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

            Q_max = round((Q_test / S_test ) * S_max, 4)
            
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

@app.route("/download/<image>")
def download(image):
    if session.get("FILENAME", None) is not None:
        path1 = os.path.join(app.static_folder, session.get("FILENAME"))
        os.remove(path1)
        session.pop("FILENAME", None)
        try:
            return send_from_directory(app.config["CLIENT_DOWNLOAD"], image, as_attachment=True)
        except FileNotFoundError:
            abort(404)
    else:
        return redirect(url_for('step_test'))

def fxn(x, a, b):
    return a * x ** 2 + b * x