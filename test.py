import os
import flask
import base64
import numpy               as     np
import tensorflow          as     tf
from   swiftclient.service import Connection

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

        conn = Connection(key='xxxxx',
                          authurl='https://identity.open.softlayer.com/v3',
                          auth_version='3',
                          os_options={"project_id": 'xxxxxx',
                                      "user_id": 'xxxxxx',
                                      "region_name": 'dallas'}
                          )

        obj       = conn.get_object("tensorflow", "retrained_graph.pb")
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(obj[1])
        with graph.as_default():
            tf.import_graph_def(graph_def)

        obj    = conn.get_object("tensorflow", "retrained_labels.txt")
        for i in obj[1].decode("utf-8").split():
            labels.append(i)

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

    if message and not isinstance(message, dict):
        return error()
    else:
        args = message.get('value', {}) if message else {}

        if not isinstance(args, dict):
        
            return error()

        print(args)

        print("=====================================")
        
        print(message)
        answer = {"msg": "o luiz é gayzao"}
        response = flask.jsonify(answer)
        response.status_code = 200

    return response


if __name__ == '__main__':
    port = int(os.getenv('FLASK_PROXY_PORT', 8080))
    app.run(host='0.0.0.0', port=port)