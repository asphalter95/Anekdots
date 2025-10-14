import pandas as pd
import dill
import sys
import os
import zipfile
import torch
import torch.nn as nn
import numpy as np
from Model import Transformer_mix
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

parent_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)
print(parent_dir)

path_to_vocab = os.path.join(parent_dir, 'anek_vocab.pkl')
print(path_to_vocab)
weight_path = os.path.join(parent_dir, 'weights_my5638.pt')
weight_path1 = os.path.join(parent_dir, 'weights_part_aa')
weight_path2 = os.path.join(parent_dir, 'weights_part_ab')

#deep_pavlov_path = os.path.join(parent_dir, 'DeepPavlov')
#sber_path = os.path.join(parent_dir, 'final_robert_full_training')

with open('weights_my5638.pt.zip', 'wb') as f_out:
    for part in [weight_path1, weight_path2]:
        with open(part, 'rb') as f_in:
            f_out.write(f_in.read())

with zipfile.ZipFile("weights_my5638.pt.zip", 'r') as zip_ref:
    zip_ref.extractall(".")
classes = {0: 'aforizmi',
            1: 'meditsinskie',
            2: 'narodnie',
            3: 'poshlie-i-intimnie',
            4: 'pro-alkogolikov',
            5: 'pro-armiu',
            6: 'pro-detey',
            7: 'pro-evreev',
            8: 'pro-militsiyu',
            9: 'pro-mugchin',
            10: 'pro-novih-russkih',
            11: 'pro-semyu',
            12: 'pro-studentov',
            13: 'pro-vovochku',
            14: 'raznie',
            15: 'shkolnie-i-pro-shkolu',
            16: 'tsitati'}

class Form(BaseModel):
    anekdot: list

class Predict(BaseModel):
    pred:str

model = Transformer_mix(path_to_vocab=path_to_vocab)
state = torch.load(weight_path, map_location="cpu")
state = {k: v for k, v in state.items() if not k.startswith("segments")}
model.load_state_dict(state, strict=False)
model.eval()
tokenizer_pavlov = AutoTokenizer.from_pretrained('asphalter95/Anekdots_bert')
model_pavlov = AutoModelForSequenceClassification.from_pretrained('asphalter95/Anekdots_bert')
tokenizer_sber = AutoTokenizer.from_pretrained('asphalter95/Anekdots_bert')
model_sber = AutoModelForSequenceClassification.from_pretrained('asphalter95/Anekdots_bert')

def predict_model(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).tolist()
        return preds

app = FastAPI()

@app.get('/status')

def status():
    return 'I am OK'

@app.post('/predict')

def predict(form: Form):
    data = dict(form)['anekdot']
    preds_mix = [model.predict(text, device="cpu") for text in data]
    predict_mix = dict(zip(data, [classes[pred] for pred in preds_mix]))

    preds_pavlov = predict_model(model_pavlov, tokenizer_pavlov, data)
    predict_pavlov = dict(zip(data, [classes[pred] for pred in preds_pavlov]))

    preds_sber = predict_model(model_sber, tokenizer_sber, data)
    predict_sber = dict(zip(data, [classes[pred] for pred in preds_sber]))
    return {'Sber': predict_sber, 'Pavlov': predict_pavlov, 'Mix': predict_mix}