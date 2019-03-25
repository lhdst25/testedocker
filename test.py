import os
import flask
import base64
import numpy               as     np
import tensorflow          as     tf
from   swiftclient.service import Connection
from pattern_recognition import main_patterns
import jsonpickle
import pandas as pd
import pickle

app       = flask.Flask(__name__)
app.debug = False
graph     = tf.Graph()
labels    = []


@app.route('/init', methods=['POST'])
def init():
    print(tf.__version__)
    try:

        message = flask.request.get_json(force=True, silent=True)
        
        if message and not isinstance(message, dict):
            flask.abort(404)

        print(message)


    except Exception as e:
        print("Error in downloading content")
        print(e)
        response = flask.jsonify({'error downloading models': e})
        response.status_code = 512

    return ('OK', 200)


@app.route('/run', methods=['POST'])
def run():
    print(tf.__version__)
    def error():
        response = flask.jsonify({'error': 'The action did not receive a dictionary as an argument.'})
        response.status_code = 404
        return response

    message = flask.request.get_json(force=True, silent=True)
    print(message)
    if message and not isinstance(message, dict):
        return error()
    else:
        args = message.get('value', {}) if message else {}

        if not isinstance(args, dict):
        
            return error()

        dict_out = main_patterns(message)

    
        params = ['minSamples', 'samplesDistance', 'epsAll',
                  'slopeAngle', 'predictOutlier', 'threshold', 'mse',
                  'predictPattern', 'scalerAnomaly', 'modelAnomaly',
                  'modelPattern', 'nClusters', 'resultsPattern']


        dict_encode = {}
        for i in params:
            dict_encode[i] = jsonpickle.encode(pickle.dumps(dict_out[i]))

        jsondictOut = jsonpickle.encode(dict_encode)

        print("=====================================")
        
        print(message)
        answer = {"msg": jsondictOut}
        response = flask.jsonify(answer)
        response.status_code = 200

    return response


if __name__ == '__main__':
    port = int(os.getenv('FLASK_PROXY_PORT', 8080))
    app.run(host='0.0.0.0', port=port)