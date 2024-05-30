from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from datasets import load_dataset
import os
from tqdm import tqdm

data_name = "wmt14"
subset = "de-en"

def to_list_casting(data):
    data["en"] = wmt14["train"]["translation"]["en"]
    data["de"] = wmt14["train"]["translation"]["de"]
    return data

# Load the WMT14 en-de dataset
wmt14 = load_dataset(data_name, subset)

data = []
for i in tqdm(range(len(wmt14["train"]["translation"]))):
    data.append(wmt14["train"][i]["translation"]["en"])
    data.append(wmt14["train"][i]["translation"]["de"])

# wmt14 = wmt14.map(to_list_casting, num_proc=4)
# print(wmt14)

# Initialize a tokenizer
tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# Set up pre-tokenization
# tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Initialize a trainer
trainer = trainers.BpeTrainer(vocab_size=37000, min_frequency=2, special_tokens=["<unk>", "<s>", "</s>", "<pad>"], show_progress=True)

tokenizer.train_from_iterator(data, trainer=trainer)

os.makedirs("tokenizer", exist_ok=True)
# Save the trained tokenizer
tokenizer.save(f"tokenizer/{data_name}_{subset}_BPEtokenizer.json")