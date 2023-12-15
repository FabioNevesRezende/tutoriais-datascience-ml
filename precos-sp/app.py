import flask
import pandas as pd 
from joblib import load

app = flask.Flask(__name__, template_folder="templates")

new_model = load('models/model.joblib') 
features = load('models/features.names') 

print(f'model type: {type(new_model)}')

@app.route("/", methods=["GET","POST"])
def main():
    if flask.request.method == "GET":
        return flask.render_template('hello.html')

    if flask.request.method == "POST":
        user_inputs = {
            'Condo': flask.request.form['condominio'],
            'Size': flask.request.form['area'],
            'Rooms': flask.request.form['quartos'],
            'Suites': flask.request.form['suites']
        }

        df = pd.DataFrame(index=[0], columns=features)
        df.fillna(value=0)

        for i in user_inputs.items():
            df[i[0]] = i[1]
        df = df.astype(float)

        print(df)

        y_pred = new_model.predict(df)[0]

        return flask.render_template('hello.html', preco_venda=y_pred)

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)