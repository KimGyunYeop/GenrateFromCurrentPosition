
# #check data maxlen
# print("train data max lenght...")
# td = tqdm(train_dataloader)
# max_enc_input = 0
# max_dec_input = 0
# tmp_enc_input = None
# tmp_dec_input = None
# for batch in td:
#     if max_enc_input < batch["input_ids"].shape[1]:
#         max_enc_input = batch["input_ids"].shape[1]
#         tmp_enc_input = batch
#         print("max_enc_input", max_enc_input)
        
#     if max_enc_input * 0.9 <= batch["input_ids"].shape[1]:
#         print("max_enc_input 90%", max_enc_input)

#     if max_dec_input <= batch["labels"].shape[1]:
#         max_dec_input = batch["labels"].shape[1]
#         tmp_dec_input = batch
#         print("max_dec_input", max_dec_input)
    
#     if max_dec_input * 0.9 <= batch["labels"].shape[1]:
#         print("max_dec_input 90%", max_dec_input)

# print(max_enc_input, max_dec_input)

# for i in batch.keys():
#     tmp_enc_input[i] = tmp_enc_input[i].to(device)

# out = model(**batch)
# out.loss.backward()
# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
# td.set_postfix(loss=out.loss.item())

# optimizer.step()
# scheduler.step()
# optimizer.zero_grad()

# for i in batch.keys():
#     tmp_dec_input[i] = tmp_dec_input[i].to(device)

# out = model(**batch)
# out.loss.backward()
# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
# td.set_postfix(loss=out.loss.item())

# optimizer.step()
# scheduler.step()
# optimizer.zero_grad()



# import datasets
# from transformers import EncoderDecoderModel, BertTokenizerFast, BertConfig, GPT2Config, EncoderDecoderConfig, GPT2Model, T5Model, BartModel, BartConfig, T5Config
# import torch

# # # load pure transformer using transfomers package without pre-trained parameter
# # model_name = "bert-base-uncased"
# # tokenizer = BertTokenizerFast.from_pretrained(model_name)
# # print(len(tokenizer))
# # bert_config = BertConfig()
# # gpt2_config = GPT2Config(n_layer=5)
# # config = EncoderDecoderConfig.from_encoder_decoder_configs(bert_config, gpt2_config)
# # model = EncoderDecoderModel(config=config)  
# # print(model)

# # input_ids = torch.randint(0, 30522, (1, 5))
# # target_ids = torch.randint(0, 30522, (1, 10))

# # out = model(input_ids=input_ids, decoder_input_ids=target_ids, output_attentions=True, output_hidden_states=True)
# # # print(out)

# # # print(out.decoder_attentions)
# # # print(len(out.decoder_attentions))
# # # print(len(out.cross_attentions))
# # print(out.cross_attentions[0].shape)
# # print(out.decoder_attentions[0].shape)
# # print(out.keys())

# # bartconfig = BartConfig(n_layer=5)
# # bartmodel = BartModel(config=bartconfig)
# # print(bartmodel)

# # out = bartmodel(input_ids=input_ids, decoder_input_ids=target_ids, output_attentions=True, output_hidden_states=True)
# # print(out.keys())

# # t5config = T5Config(n_layer=5)
# # t5model = T5Model(config=t5config)
# # print(t5model)
# # out = t5model(input_ids=input_ids, decoder_input_ids=target_ids, output_attentions=True, output_hidden_states=True)
# # print(out.keys())

# from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors, decoders
# from transformers import PreTrainedTokenizerFast

# tokenizer = Tokenizer.from_file("tokenizer/wmt14_de-en_BPEtokenizer.json")
# # tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
# tokenizer.decoder = decoders.ByteLevel()
# tokenizer.post_processor = processors.RobertaProcessing(
#             ("</s>", tokenizer.token_to_id("</s>")),
#             ("<s>", tokenizer.token_to_id("<s>")),
#         )

# wrapped_tokenizer = PreTrainedTokenizerFast(
#     tokenizer_object=tokenizer,
#     bos_token="<s>",
#     eos_token="</s>",
#     unk_token="<unk>",
#     pad_token="<pad>",
# )
# print(wrapped_tokenizer.bos_token_id)
# print(wrapped_tokenizer("Hello, how are you?", add_special_tokens=True, return_special_tokens_mask=True))
# print(wrapped_tokenizer.convert_ids_to_tokens(wrapped_tokenizer("Hello, how are you?")["input_ids"]))
# print(wrapped_tokenizer.batch_decode([wrapped_tokenizer("Hello, how are you?")["input_ids"]]))
# print(wrapped_tokenizer.batch_decode([wrapped_tokenizer("Hello, how are you?")["input_ids"]], skip_special_tokens=True))

# # toeknizer.post_processor = pre_tokenizers.Sequence([pre_tokenizers.Metaspace()])

# print(tokenizer.encode("Hello, how are you?"))

# bartconfig = BartConfig(n_layer=6, 
#                         activation_function="relu", 
#                         pad_token_id=wrapped_tokenizer.pad_token_id, 
#                         eos_token_id=wrapped_tokenizer.eos_token_id, 
#                         bos_token_id=wrapped_tokenizer.bos_token_id, 
#                         decoder_start_token_id=wrapped_tokenizer.bos_token_id, 
#                         is_encoder_decoder=True, forced_eos_token_id=wrapped_tokenizer.eos_token_id, 
#                         vocab_size=len(wrapped_tokenizer))

# print(bartconfig)

from transformers import AutoTokenizer, BartForConditionalGeneration

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

ARTICLE_TO_SUMMARIZE = (
    "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
    "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
    "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
)
inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="pt")

# Generate Summary
summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=20)
print(tokenizer.batch_decode(summary_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0])