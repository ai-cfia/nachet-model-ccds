#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import transformers
import accelerate
import peft

print(f"Transformers version: {transformers.__version__}")
print(f"Accelerate version: {accelerate.__version__}")
print(f"PEFT version: {peft.__version__}")


# In[ ]:


# from huggingface_hub import notebook_login

# notebook_login()


# In[2]:


MODEL_STRING = "microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft"


# In[ ]:


get_ipython().system('accelerate estimate-memory "microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft" --library_name transformers')


# In[3]:


from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


# In[ ]:


print_gpu_utilization()


# In[ ]:


import torch

with torch.no_grad():
    torch.cuda.empty_cache()


# In[ ]:


# from transformers import AutoImageProcessor, Swinv2Model
from transformers import Swinv2Model

# image_processor = AutoImageProcessor.from_pretrained(MODEL_STRING)
model = Swinv2Model.from_pretrained(MODEL_STRING)
model = model.to("cuda")
print_gpu_utilization()

