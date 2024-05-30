import datasets
from transformers import BartModel, BartConfig, BartForConditionalGeneration, BertModel, BartTokenizer
from transformers import AdamW, get_scheduler
import torch
from datasets import load_dataset
import custom_datasets
import custom_tokenizer
import evaluate

from tqdm import tqdm
# from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors

data_name = "wmt14"
subset = "de-en"
batch_size = 16
tokenizer_path = "tokenizer/wmt14_de-en_BPEtokenizer.json"
gpu = 0
device = "cuda:"+str(gpu)
learning_rate = 1e-5
epoch = 10

# wmt 14 train bart model
dataset = load_dataset(data_name, subset)
print(dataset)

# tokenizer = custom_tokenizer.get_tokenizer(tokenizer_path)
# bartconfig = BartConfig(n_layer=6, 
#                         activation_function="relu", 
#                         pad_token_id=tokenizer.pad_token_id, 
#                         eos_token_id=tokenizer.eos_token_id, 
#                         bos_token_id=tokenizer.bos_token_id, 
#                         decoder_start_token_id=tokenizer.bos_token_id, 
#                         is_encoder_decoder=True, forced_eos_token_id=tokenizer.eos_token_id, 
#                         vocab_size=len(tokenizer))

# model = BartForConditionalGeneration(config=bartconfig)
# model.to(device)
# print(model)

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
model.to(device)
print(model)

gen_args = {'do_sample': True, 'top_k': 10, 'num_return_sequences': 1, 'eos_token_id': [tokenizer.eos_token_id, tokenizer.pad_token_id], 'max_new_tokens': 200, "early_stopping":True, "temperature":0.7, "num_beams":4}

train_dataset = custom_datasets.WmtDataset(dataset["train"], tokenizer=tokenizer, src_lang="en", tgt_lang="de")
val_dataset = custom_datasets.WmtDataset(dataset["validation"], tokenizer=tokenizer, src_lang="en", tgt_lang="de")
test_dataset = custom_datasets.WmtDataset(dataset["test"], tokenizer=tokenizer, src_lang="en", tgt_lang="de")

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, collate_fn=val_dataset.collate_fn)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_dataset.collate_fn)

num_training = len(train_dataloader) * epoch
optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = get_scheduler("linear", optimizer, num_warmup_steps=100, num_training_steps=num_training)

matric = evaluate.load("sacrebleu")

cur_step = 0

refers = []
preds = []
model.train()
for E in range(epoch):
    print(f"Epoch {E}")

    for batch in tqdm(train_dataloader):
        for i in batch.keys():
            batch[i] = batch[i].to(device)
            # print(i, batch[i].shape)

        out = model(**batch)
        out.loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # print(out.loss)
        cur_step += 1
        if cur_step > 10000:
            break

    for batch in tqdm(val_dataloader):
        for i in batch.keys():
            batch[i] = batch[i].to(device)

        out = model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], **gen_args)
        pred_str = tokenizer.batch_decode(out, skip_special_tokens=True)

        target_str = tokenizer.batch_decode(batch["decoder_input_ids"], skip_special_tokens=True)
        print(pred_str)
        print(target_str)
        
        refers.extend(target_str)
        preds.extend(pred_str)

    # matric.add_batch(predictions=preds, references=refers)
    print(matric.compute(predictions=preds, references=refers))



