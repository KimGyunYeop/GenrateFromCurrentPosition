from tokenizers import Tokenizer, processors, decoders
from transformers import PreTrainedTokenizerFast

def get_tokenizer(path):
    # tokenizer = Tokenizer.from_file("tokenizer/wmt14_de-en_BPEtokenizer.json")
    tokenizer = Tokenizer.from_file(path)
    # tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.post_processor = processors.RobertaProcessing(
                ("</s>", tokenizer.token_to_id("</s>")),
                ("<s>", tokenizer.token_to_id("<s>")),
            )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.enable_truncation(max_length=512)

    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )

    return wrapped_tokenizer