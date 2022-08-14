import logging
import pandas as pd
from simpletransformers.classification import MultiLabelClassificationModel
from sklearn.model_selection import train_test_split
import torch
from transformers import ElectraTokenizer, ElectraTokenizer
from transformers import XLNetTokenizer, XLNetModel
from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel
import os
from simpletransformers.seq2seq import (
    Seq2SeqModel,
    Seq2SeqArgs,
)


try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
train_data = pd.read_csv('data/train/meqsum_nli_prerqe_mqp_covid.csv',encoding='utf-8')
train_data['input_text'] = train_data['CHQ']
train_data['target_text'] = train_data['Summary']
print(train_data['input_text'])
print(train_data['target_text'])
eval_data = pd.read_excel('data/val/MEDIQA2021_Task1_QS_TestSet_Questions.xlsx')
eval_data['input_text'] = eval_data['NLM question']





model_args = {
    'n_gpu' : 1,
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    'evaluate_during_training': True,
    'evaluate_generated_text' : True,
    'evaluate_during_training_verbose' : True,
    'use_early_stopping' : True,
    'early_stopping_metric' : 'eval_loss',
    'early_stopping_metric_minimize' : True,
    'early_stopping_patience' : 5,
    'num_train_epochs': 100,
    'train_batch_size': 8,
    'eval_batch_size': 8,
    'gradient_accumulation_steps':128, 
    'learning_rate': 5e-5, 
    'max_seq_length': 512,
    'fp16': False}


# Initialize model
model = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name="/workplace/phchen/mediqa2021_20210220/bart_large_xsum_batch8_gas512_lr5e5_noval/checkpoint-44-epoch-4",
    args=model_args,
)



def count_matches(labels, preds):
    print(labels)
    print(preds)
    return sum(
        [
            1 if label == pred else 0
            for label, pred in zip(labels, preds)
        ]
    )


preds = model.predict(eval_data['input_text'])
pred_df = pd.DataFrame(preds)
pred_df.to_csv(r'/workplace/phchen/mediqa2021_20210220/bart_large_xsum_batch8_gas512_lr5e5_noval/fortestonly_bart_large_xsum_checkpoint-44-epoch-4.csv')

print(
    model.predict(
        eval_data['input_text']
    )
)