FROM public.ecr.aws/sagemaker/sagemaker-distribution:latest-gpu

ARG NB_USER="sagemaker-user"
ARG NB_UID=1000
ARG NB_GID=100

ENV MAMBA_USER=$NB_USER

USER root

RUN apt-get update
RUN micromamba install sagemaker-inference --freeze-installed --yes --channel conda-forge --name base

# Install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy only the streaming-related files we need
COPY data/s3_data_fetcher.py /opt/ml/processing/data/
COPY data/streaming_feature_engineering.py /opt/ml/processing/data/
COPY data/StreamingHFTDataset.py /opt/ml/processing/data/
COPY pipelines/processing_script.py /opt/ml/processing/

# Set working directory
WORKDIR /opt/ml/processing

USER $MAMBA_USER

# Set up entrypoint
ENTRYPOINT ["python3", "processing_script.py"]
