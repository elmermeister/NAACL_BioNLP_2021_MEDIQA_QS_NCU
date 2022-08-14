from transformers import PegasusForConditionalGeneration, PegasusTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pandas as pd

print('start predicting')
src = pd.read_excel('data/val/MEDIQA2021_Task1_QS_TestSet_Questions.xlsx')
src_text = list(src['NLM question'][75:])
model_name = '/workplace/phchen/mediqa2021_20210220/run_test/pegasus_xsum_batch12_gas256_lr5e5_noval/checkpoint-140'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")

model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
batch = tokenizer.prepare_seq2seq_batch(src_text, truncation=True, padding='longest', return_tensors="pt").to(torch_device)
translated = model.generate(**batch)
tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)


df = pd.DataFrame(tgt_text, columns=["0"])
df.to_csv('/workplace/phchen/mediqa2021_20210220/run_test/pegasus_xsum_batch12_gas256_lr5e5_noval/onlyfortest_pegasus_xsum-4.csv', index=False)
print('finished predicting')


print('finished predicting')