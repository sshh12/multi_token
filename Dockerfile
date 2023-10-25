FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

WORKDIR /app

RUN apt-get update && apt-get install -y git curl nano wget unzip rsync jq

RUN git clone https://github.com/sshh12/multi_token \
        && cd multi_token \
        && pip install -r requirements.txt \
        && pip install -e .

RUN pip install flash-attn --no-build-isolation

CMD bash