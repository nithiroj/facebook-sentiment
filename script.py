import flask
from flask import Flask, render_template, request

import os
import numpy as np
from predict import predict

# IMAGE_FOLDER = os.path.join('static', images)
IMAGE_FOLDER = 'static' # name 'static' only

app=Flask(__name__)
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER

@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        txt = to_predict_list[0]
        pred, token = predict(txt)

        if pred[0]=='pos':
#           prediction='++ บวก ++'
            prediction='pos.png'

        else:
#           prediction='-- ลบ --'
            prediction='neg.png'

        full_filename = os.path.join(app.config['IMAGE_FOLDER'], prediction)

        return render_template("result.html", prediction=full_filename, token=token, txt=txt)
