FROM tensorflow/tensorflow:1.4.0-py3

WORKDIR /tensorflow
COPY requirements.txt requirements.txt
RUN  pip install -r   requirements.txt

COPY main.py main.py
COPY pattern_recognition.py pattern_recognition.py
COPY pattern_prediction.py pattern_prediction.py
COPY DBN /tensorflow/DBN 

CMD python -u main.py