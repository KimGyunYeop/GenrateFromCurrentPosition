import torch
from torch.utils.data import Dataset

class WmtDataset(Dataset):
    def __init__(self, dataset, tokenizer, src_lang="en", tgt_lang="de") -> None:
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]["translation"]
    
    def collate_fn(self, batch):
        src = [item[self.src_lang] for item in batch]
        tgt = [item[self.tgt_lang] for item in batch]

        src_token = self.tokenizer(src, return_tensors="pt", padding=True, truncation=True, max_length=512)
        tgt_token = self.tokenizer(tgt, return_tensors="pt", padding=True, truncation=True, max_length=512)

        labels = torch.where(tgt_token["input_ids"] == self.tokenizer.pad_token_id, -100, tgt_token["input_ids"])
        
        # tgt_token["input_ids"] = tgt_token["input_ids"][:, :-1].contiguous()
        # tgt_token["attention_mask"] = tgt_token["attention_mask"][:, :-1].contiguous()
        
        # labels = labels[:, 1:].contiguous()

        # return {"input_ids": src_token["input_ids"], "attention_mask": src_token["attention_mask"], "decoder_input_ids": tgt_token["input_ids"], "decoder_attention_mask": tgt_token["attention_mask"], "labels": labels}
        
        return {"input_ids": src_token["input_ids"], "attention_mask": src_token["attention_mask"], "labels": labels}
    
    