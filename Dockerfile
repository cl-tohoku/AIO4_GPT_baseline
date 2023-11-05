FROM nvcr.io/nvidia/pytorch:23.05-py3

# Copy codes
COPY config /code/config
# COPY prediction_loop.py /code/prediction_loop.py
COPY prediction_api.py /code/prediction_api.py
COPY modules/models.py /code/modules/models.py
COPY modules/mylogger.py /code/modules/mylogger.py
COPY modules/model_pipeline.py /code/modules/model_pipeline.py
COPY requirements.txt /code/requirements.txt
COPY setup.sh /code/setup.sh

# Install dependencies
WORKDIR /code
RUN source setup.sh

# Download transformers models in advance
ARG TRANSFORMERS_BASE_MODEL_NAME="rinna/japanese-gpt-1b"
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('${TRANSFORMERS_BASE_MODEL_NAME}')"
RUN python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('${TRANSFORMERS_BASE_MODEL_NAME}')"
ENV TRANSFORMERS_OFFLINE=1

# Start the prediction loop
WORKDIR /code
 # ENTRYPOINT ["uvicorn"]
CMD ["uvicorn", "prediction_api:app", "--host", "0.0.0.0", "--port", "8000"]