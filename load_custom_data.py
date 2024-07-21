import pandas as pd
import json
from datasets import load_dataset
from typing import List
from transformers import AutoTokenizer

def from_hf_tokens_to_valid(df: pd.DataFrame) -> pd.DataFrame:
  data = []

  for index, row in df.iterrows():
      tokens = row['tokens']
      ner_tags = row['ner_tags']
      sentence_id = row['id']

      for i, token in enumerate(tokens):
          new_row = {
              'words': token,
              'labels': entity_map[ner_tags[i]],
              'sentence_id': sentence_id
          }
          data.append(new_row)

  format_df = pd.DataFrame(data)

  return format_df

def parse_bcdr_to_format(df: pd.DataFrame) -> pd.DataFrame:
  new_data = []
  checkpoint_path = './pretrained_models/bioformer-cased-v1.0/'
  tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True, do_lower_case=True)

  for index, row in df.iterrows():
    for passage in row['passages']:
      text = passage['text']
      entities = passage['entities']
      tokens = tokenizer.tokenize(text)
      token_offsets = tokenizer(text, return_offsets_mapping=True)['offset_mapping']

      labels = ['O'] * len(token_offsets)
      for entity in entities:
          entity_text = entity['text'][0]
          entity_offsets = entity['offsets'][0]
          start_offset, end_offset = entity_offsets

          # Encontrar los tokens correspondientes a la entidad
          for i, (start, end) in enumerate(token_offsets):
              if start >= start_offset and end <= end_offset:
                  if labels[i] == 'O':
                      labels[i] = 'B-' + entity['type']
                  else:
                      labels[i] = 'I-' + entity['type']

      # Agregar los tokens y etiquetas al nuevo dataset
      for i, token in enumerate(tokens):
          new_data.append({
            'words': token,
            'labels': labels[i],
            'sentence_id': passage['document_id']
          })


def from_hf_to_pubtator(df: pd.DataFrame) -> List[str]:
    data = []

    for index, row in df.iterrows():
        tokens = row['tokens']
        sentence_id = row['id']
        text = ' '.join(tokens)

        data.append(f'{sentence_id}|t|{text}\n')

    return data

def parse_bcdr_to_format(df: pd.DataFrame) -> List[str]:
    new_data = []

    for index, row in df.iterrows():
        line = ''
        for passage in row['passages']:
            text = passage['text']
            type = passage['type']
            
            line += f'{passage["document_id"]}|{type[0]}|{text}\n'

        new_data.append(line)
    
    return new_data

class NERBenchmarkDataWrangler:
  def __init__(self, dataset_name: str, datasets: dict):
    self.dataset_name = datasets[dataset_name]['dataset_name']
    self.parse_function = datasets[dataset_name]['parse_function']
    self.to_pubtator = datasets[dataset_name]['to_pubtator']
    self.dataset = load_dataset(self.dataset_name, trust_remote_code=True)
    self.train_df = self.dataset['train'].to_pandas()
    self.test_df = self.dataset['test'].to_pandas()
    self.valid_df = self.dataset['validation'].to_pandas()

  def get_train_df(self):
    return self.parse_function(self.train_df)

  def get_test_df(self):
    return self.parse_function(self.test_df)

  def get_valid_df(self):
    return self.parse_function(self.valid_df)
  
  def to_pubtator(self, type: str) -> List[str]:
    if type == 'train':
      return self.to_pubtator(self.train_df)
    elif type == 'test':
      return self.to_pubtator(self.test_df)
    elif type == 'valid':
      return self.to_pubtator(self.valid_df)
    else:
       return self.to_pubtator(self.train_df)
  
  def get_labels(self):
    return self.parse_function(self.train_df)['labels'].unique().tolist()

entity_map = {
    0: 'O',
    1: 'I-Entity',
    2: 'B-Entity'
}

datasets_map = {
    "ncbi": {
        "dataset_name": "ncbi_disease",
        "parse_function": from_hf_tokens_to_valid,
        "to_pubtator": from_hf_to_pubtator
    },
    "bc2gm": {
        "dataset_name": "spyysalo/bc2gm_corpus",
        "parse_function": from_hf_tokens_to_valid,
        "to_pubtator": from_hf_to_pubtator
    },
    # genia: {
    #     dataset_name: "bigbio/genia_term_corpus",
    #     format_type: "pubtator",
    # },
    "linnaeus": {
        "dataset_name": "cambridgeltl/linnaeus",
        "parse_function": from_hf_tokens_to_valid,
        "to_pubtator": from_hf_to_pubtator
    },
    "bc5dr": {
        "dataset_name": "bigbio/bc5cdr",
        "parse_function": parse_bcdr_to_format,
        "to_pubtator": parse_bcdr_to_format
    },
    "jnlpba": {
        "dataset_name": "jnlpba/jnlpba",
        "parse_function": from_hf_tokens_to_valid,
        "to_pubtator": from_hf_to_pubtator
    }
}

for dataset in datasets_map.keys():
  wrangler = NERBenchmarkDataWrangler(dataset, datasets_map)
  train_df = wrangler.get_train_df()
  train_pubtator = wrangler.to_pubtator('train')
  labels = wrangler.get_labels()

  with open(f'./custom_datasets/{dataset}', 'w') as f:
    f.write(''.join(train_pubtator))

  with open(f'./custom_vocab/{dataset}', 'w') as f:
    f.write('\n'.join(labels))