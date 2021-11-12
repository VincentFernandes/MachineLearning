import flask 

import numpy as np
import pickle



app = flask.Flask(__name__, template_folder='templates')

@app.route('/predict', methods=['POST'])
def predict():
    model = pickle.load(open('model/model_classifier.pkl', 'rb'))
    int_features = [float(x) for x in flask.request.form.values()]
    # age = request.form["inputage"]
    # sex = request.form["inputsex"]
    # cp = request.form["inputcp"]
    # trtbps = request.form["inputtrtbps"]
    # chol = request.form["inputchol"]
    # fbs = request.form["inputfbs"]
    # restecg = request.form["inputrest_ecg"]
    # thalachh = request.form["inputthalachh"]
    # exng = request.form["inputexng"]
    # oldpeak = request.form["inputoldpeak"]
    # slp = request.form["inputslop"]
    # caa = request.form["inputcaa"]
    # thall = request.form["inputthall"]

    # final_features = [np.array([age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall])]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = {0: '0', 1: '1'}

    return flask.render_template('main.html', prediction_text='{}'.format(output[prediction[0]]))

@app.route('/')
def main():
    return(flask.render_template('main.html'))
if __name__ == "__main__":
    app.run(debug=True)