import os
from hdm2 import HallucinationDetectionModel


os.environ['TRANSFORMERS_CACHE'] = '/root/.cache/huggingface'
os.environ['HF_HOME'] = '/root/.cache/huggingface'


hdm = HallucinationDetectionModel()
