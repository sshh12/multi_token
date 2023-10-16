FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

WORKDIR /app

RUN apt-get update && apt-get install -y git curl nano

RUN git clone https://github.com/sshh12/lmm_multi_token \
        && cd lmm_multi_token \
        && pip install requirements.txt \
        && pip install -e .

RUN pip install flash-attn --no-build-isolation

CMD bash