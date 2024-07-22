import pandas as pd
import os
from datasets import load_dataset
from typing import Dict, List, Optional
from transformers import AutoTokenizer

def from_hf_tokens_to_valid(df: pd.DataFrame, entity_map: Optional[Dict[int, str]]) -> pd.DataFrame:
  if entity_map is None:
     raise ValueError('Entity map is required for this dataset')
  
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

def parse_bcdr_to_format(df: pd.DataFrame, entity_map: Optional[Dict[int, str]]) -> pd.DataFrame:
  new_data = []
  checkpoint_path = '/content/AIONER/pretrained_models/bioformer-cased-v1.0/'

  print(os.path.exists(checkpoint_path))
  print(os.listdir(checkpoint_path))

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
      if index == 0:
        print(labels)
        print(tokens)
      # Agregar los tokens y etiquetas al nuevo dataset
      for i, token in enumerate(tokens):
          new_data.append({
            'words': token,
            'labels': labels[i],
            'sentence_id': passage['document_id']
          })

  format_df = pd.DataFrame(new_data)

  return format_df 


def from_hf_to_pubtator(df: pd.DataFrame) -> List[str]:
    data = []
    print(df)
    for index, row in df.iterrows():
        tokens = row['tokens']
        sentence_id = row['id']
        text = ' '.join(tokens)

        data.append(f'{sentence_id}|t|{text}\n')

    return data

def parse_bcdr_to_pubtator(df: pd.DataFrame) -> List[str]:
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
    self.to_pubtator_fn = datasets[dataset_name]['to_pubtator']
    self.entity_map = datasets[dataset_name].get('entity_map', None)
    self.dataset = load_dataset(self.dataset_name, trust_remote_code=True)

    if 'train' not in self.dataset:
      raise ValueError('Dataset does not have a train split')
    
    ## Split data so now we have train, test and validation

    if 'test' not in self.dataset and 'validation' in self.dataset:
        splited_data = self.dataset['validation'].train_test_split(test_size=0.3)

        self.train_df = self.dataset['train'].to_pandas()
        self.test_df = splited_data['test'].to_pandas()
        self.valid_df = splited_data['train'].to_pandas()

        return
    
    if 'test' not in self.dataset and 'validation' not in self.dataset:
        splited_data = self.dataset['train'].train_test_split(test_size=0.3)
        val_splited_data = splited_data['test'].train_test_split(test_size=0.5)

        self.train_df = splited_data['train'].to_pandas()
        self.test_df = val_splited_data['train'].to_pandas()
        self.valid_df = val_splited_data['test'].to_pandas()

        return
    
    if 'test' in self.dataset and 'validation' not in self.dataset:
        splited_data = self.dataset['train'].train_test_split(test_size=0.3)

        self.train_df = self.dataset['train'].to_pandas()
        self.test_df = self.dataset['test'].to_pandas()
        self.valid_df = splited_data['test'].to_pandas()

        return

    self.test_df = self.dataset['test'].to_pandas()
    self.valid_df = self.dataset['validation'].to_pandas()
    self.train_df = self.dataset['train'].to_pandas()

  def get_format_data(self, type: str) -> pd.DataFrame:
    if type == 'train':
      return self.parse_function(self.train_df, self.entity_map)
    elif type == 'test':
      return self.parse_function(self.test_df, self.entity_map)
    elif type == 'valid':
      return self.parse_function(self.valid_df, self.entity_map)
    else:
       return self.parse_function(self.train_df, self.entity_map)
  
  def to_pubtator(self, type: str) -> List[str]:
    if type == 'train':
      return self.to_pubtator_fn(self.train_df)
    elif type == 'test':
      return self.to_pubtator_fn(self.test_df)
    elif type == 'valid':
      return self.to_pubtator_fn(self.valid_df)
    else:
       return self.to_pubtator_fn(self.train_df)
  
  def get_labels(self):
    parsed_data = self.get_format_data('train')

    return parsed_data['labels'].unique().tolist()

entity_map = {
    0: 'O',
    1: 'I-Entity',
    2: 'B-Entity'
}

jnlpba_mapping = {
    0: 'O',
    1: 'B-DNA',
    2: 'I-DNA',
    3: 'B-RNA',
    4: 'I-RNA',
    5: 'B-cell_line',
    6: 'I-cell_line',
    7: 'B-cell_type',
    8: 'I-cell_type',
    9: 'B-protein',
    10: 'I-protein'
}


datasets_map = {
    "ncbi": {
        "dataset_name": "ncbi_disease",
        "parse_function": from_hf_tokens_to_valid,
        "to_pubtator": from_hf_to_pubtator,
        "entity_map": entity_map,
    },
    "bc2gm": {
        "dataset_name": "spyysalo/bc2gm_corpus",
        "parse_function": from_hf_tokens_to_valid,
        "to_pubtator": from_hf_to_pubtator,
        "entity_map": entity_map,
    },
    "linnaeus": {
        "dataset_name": "cambridgeltl/linnaeus",
        "parse_function": from_hf_tokens_to_valid,
        "to_pubtator": from_hf_to_pubtator,
        "entity_map": entity_map,
    },
    "bc5dr": {
        "dataset_name": "bigbio/bc5cdr",
        "parse_function": parse_bcdr_to_format,
        "to_pubtator": parse_bcdr_to_pubtator,
    },
    "jnlpba": {
        "dataset_name": "jnlpba/jnlpba",
        "parse_function": from_hf_tokens_to_valid,
        "to_pubtator": from_hf_to_pubtator,
        "entity_map": jnlpba_mapping
    }
}

if not os.path.exists('./custom_datasets'):
    os.makedirs('./custom_datasets')

if not os.path.exists('./custom_vocab'):
    os.makedirs('./custom_vocab')

for dataset in datasets_map.keys():
    wrangler = NERBenchmarkDataWrangler(dataset, datasets_map)
    train_df = wrangler.get_format_data('train')
    train_pubtator = wrangler.to_pubtator('train')
    print(train_df)
    labels = train_df['labels'].unique().tolist()

    print(labels)

    if not os.path.exists(f'./custom_datasets/{dataset}'):
        os.makedirs(f'./custom_datasets/{dataset}')

    if not os.path.exists(f'./custom_vocab/{dataset}'):
        os.makedirs(f'./custom_vocab/{dataset}')

    with open(f'./custom_datasets/{dataset}/{dataset}.txt', 'w') as f:
        f.write(''.join(train_pubtator))

    with open(f'./custom_vocab/{dataset}/{dataset}.txt', 'w') as f:
        f.write('\n'.join(labels))