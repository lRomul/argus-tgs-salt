FROM floydhub/pytorch:0.4.1-gpu.cuda9cudnn7-py3.34

RUN pip3 install --upgrade pip &&\
    pip3 install \
    pycocotools==2.0.0\
    torchsummary\
    scikit-optimize\
    pretrainedmodels\
    pytorch-argus==0.0.5\
    cffi\
    tqdm\
    shapely

ENV PYTHONPATH $PYTHONPATH:/workdir
ENV TORCH_HOME=/workdir/data/.torch

WORKDIR /workdir
