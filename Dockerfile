FROM tensorflow/tensorflow:1.4.0-py3

WORKDIR /tensorflow
COPY requirements.txt requirements.txt
RUN  pip install -r   requirements.txt


COPY time_series.py time_series.py
COPY DBN /tensorflow/DBN 

CMD python -u time_series.py