FROM nvidia/cuda:11.0.3-devel-ubuntu20.04

ENV TZ=Europe/Vienna
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update && apt install -yq rustc libjpeg-dev openssl pkg-config build-essential libsqlite3-dev zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget
RUN wget https://www.python.org/ftp/python/3.6.15/Python-3.6.15.tgz && tar -xf Python-3.6.15.tgz
RUN cd Python-3.6.15 && ./configure --enable-optimizations && make install && cd ..
RUN rm -R Python-3.6.15 Python-3.6.15.tgz

WORKDIR /usr/src/app

COPY requirements.txt requirements.txt

RUN /usr/local/bin/pip3 install --no-cache-dir -r requirements.txt
# RUN ./download_data.sh

RUN echo 'import nltk' >> punkt_downloader.py && echo 'nltk.download("punkt")' >> punkt_downloader.py

RUN /usr/local/bin/python3 punkt_downloader.py

COPY . .

# small, medium, large
ENV MODEL_SIZE=medium
# -1 = all of the history is saved
ENV MAX_HISTORY=-1

CMD [ "sh", "-c", "python3 ./main.py --api --model_size=$MODEL_SIZE --max_history=$MAX_HISTORY" ]