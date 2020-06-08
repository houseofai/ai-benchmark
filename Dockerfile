FROM tensorflow/tensorflow:latest-gpu

RUN apt -y install git
RUN pip install --upgrade pip

RUN git clone https://github.com/OdysseeT/ai-benchmark.git
WORKDIR ai-benchmark
RUN pip install -e .
RUN chmod a+x bin/ai-benchmark
RUN ./bin/ai-benchmark
