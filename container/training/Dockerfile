FROM public.ecr.aws/sagemaker/sagemaker-distribution:latest-gpu

# Test change to trigger GitHub Actions - testing new ECR push approach
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

# Copy only what's needed for training
COPY data/StreamingHFTDataset.py /opt/ml/code/data/
COPY training/sagemaker_trainer.py /opt/ml/code/training/

# Set working directory
WORKDIR /opt/ml/code

# Set Python path to include our modules
ENV PYTHONPATH=/opt/ml/code

USER $MAMBA_USER

# Set up entrypoint for training
ENTRYPOINT ["python3", "-m", "training.sagemaker_trainer"]
