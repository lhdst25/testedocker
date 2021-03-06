import os
import json
import flask
import numpy as np
import tensorflow as tf
from pattern_recognition import main_patterns
from pattern_prediction import main_predict_pattern
import jsonpickle
import pickle

app = flask.Flask(__name__)
app.debug = False
graph = tf.Graph()
labels = []


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

    def error():
        response = flask.jsonify(
            {'error': 'The action did not receive a dictionary as an argument.'})
        response.status_code = 404
        return response
    
    def occurredat(dict_in):
        jsons = dict_in['jsons']
        dates = []
        for i in range(len(jsons)):
            dates.append(jsons[i]["occurredAt"])
    
        return dates
    
    
    def organize_dict(dict_in):
    
        info = dict_in.copy()
    
        listResults = []
    
        for i in range(len(info["dates"])):
            document = {"occurredAt": info["dates"][i],
                        "fwVersion": info["fwVersion"],
                        "anomaly": info["anomaly"][i], "mse": info["mse"][i],
                        "predictPattern": info["predictPattern"][i],
                        "deviceId": info["deviceId"], "type": "anomaly"}
            listResults.append(document)
    
        return listResults


    message = flask.request.get_json(force=True, silent=True)
    print(message)
    if message and not isinstance(message, dict):
        return error()
    else:
        args = message.get('value', {}) if message else {}

        if not isinstance(args, dict):

            return error()

        dict_encode = {}

        if int(args["type"]) == 1:

            try:
                dict_out = main_patterns(args)

                params = ['minSamples', 'samplesDistance', 'epsAll',
                          'slopeAngle', 'predictOutlier', 'threshold', 'mse',
                          'predictPattern', 'scalerAnomaly', 'modelAnomaly',
                          'modelPattern', 'nClusters', 'resultsPattern']

                for i in params:
                    dict_encode[i] = jsonpickle.encode(
                        pickle.dumps(dict_out[i]))

                dict_encode["deviceId"] = args["deviceId"]
                dict_encode["fwVersion"] = args["fwVersion"]
                
                jsondictOut = jsonpickle.encode(dict_encode)
                answer = {"msg": jsondictOut}
                response = flask.jsonify(answer)
                response.status_code = 200

                return response

            except Exception as error:
                return {'error': str(error)}

        elif int(args["type"]) == 2:

            try:
                keys = ['threshold', 'scalerAnomaly', 'modelAnomaly',
                        'modelPattern']

                dict_in = {}

                for i in keys:
                    dict_in[i] = pickle.loads(jsonpickle.decode(args[i]))

                dict_in["jsons"] = args["jsons"]

                pred = main_predict_pattern(dict_in)

                print(pred)
                for i, j in pred.items():
                    pred[i] = j.tolist()

                pred["deviceId"] = args["deviceId"]
                pred["type"] = "anomaly"
                pred["fwVersion"] = args["fwVersion"]
                pred["dates"] = occurredat(args)
        
                listResults = organize_dict(pred)
                jsondictOut1 = json.dumps(listResults)

                answer = {"msg": jsondictOut1}
                response = flask.jsonify(answer)
                response.status_code = 200

                return response

            except Exception as error:
                return {'error': str(error)}


if __name__ == '__main__':
    port = int(os.getenv('FLASK_PROXY_PORT', 8080))
    app.run(host='0.0.0.0', port=port)
