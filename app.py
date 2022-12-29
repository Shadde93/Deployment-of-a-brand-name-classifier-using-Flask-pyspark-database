from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
from Classify import main_function, pred_input
import pickle
#Hej
app = Flask(__name__)
#Bootstrap(app)

@app.route('/')
def index():
    return(render_template('index.html'))

@app.route('/main', methods = ['GET', 'POST'])
def main():

    if request.method == 'POST':
        industry_select = '(%s)' % ', '.join(map(repr, request.form.getlist('options')))
        main_function(industry_select)
    return(render_template('main.html'))

@app.route('/result', methods = ['GET', 'POST'])
def result():

    if request.method == 'POST':
        name = request.form['namequery']
        country = request.form['countryquery']
        prediction = pred_input(name, country)
    return(render_template('result.html', prediction = prediction, name = name, country = country))	


if __name__ == '__main__':
    app.run(debug = True, port = 300)
