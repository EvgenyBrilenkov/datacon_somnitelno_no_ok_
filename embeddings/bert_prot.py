import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
import json
import tqdm
from sys import argv

from transformers import BertModel, BertTokenizer
import re
df = pd.read_csv('E:\\My_projects\\poseidon_without.csv')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
access_token = "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"
df = df.dropna(subset=['Sequence'])
with torch.no_grad():
    model = BertModel.from_pretrained("Rostlab/prot_bert", token=access_token).to(device)
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", token=access_token, do_lower_case=False )
    results = []
    for idx in tqdm.tqdm(df.index):
        reaction = df.loc[idx, 'Sequence']
        reaction = re.sub(r"[UZOB]", "X", reaction)
        encoded_input = tokenizer(reaction, return_tensors='pt').to(device)
        output = model(**encoded_input).last_hidden_state.cpu()[0][-1]
        results.append({
               "input": reaction,
               "embedding": output.tolist()
       })

with open('bert_result.json', "w") as file:
   json.dump(
       list(results),
       file)