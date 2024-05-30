import torch
from transformers import BartTokenizer, BartForConditionalGeneration


def beam_search(model, tokenizer, input_text, beam_size=3, max_length=50):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    input_ids = input_ids.to(model.device)

    # Initialize the beam with the input text
    beams = [{'input_ids': input_ids, 'score': 0.0}]

    # Perform beam search
    for _ in range(max_length):
        new_beams = []
        for beam in beams:
            # Generate next token predictions
            with torch.no_grad():
                outputs = model.forward(input_ids=beam['input_ids'])
                logits = outputs.logits[:, -1, :]
                scores = torch.nn.functional.log_softmax(logits, dim=-1)

            # Get top-k predictions and their scores
            top_scores, top_indices = torch.topk(scores, k=beam_size)

            # Expand the beam with top-k predictions
            for score, index in zip(top_scores[0], top_indices[0]):
                new_input_ids = torch.cat((beam['input_ids'], index.unsqueeze(0)), dim=-1)
                new_score = beam['score'] + score.item()
                new_beams.append({'input_ids': new_input_ids, 'score': new_score})

        # Select top-k beams
        beams = sorted(new_beams, key=lambda x: x['score'], reverse=True)[:beam_size]

        # Check if all beams have reached the EOS token
        all_eos = all(torch.any(beam['input_ids'][:, -1] == tokenizer.eos_token_id) for beam in beams)
        if all_eos:
            break

    # Decode the generated output
    decoded_outputs = []
    for beam in beams:
        decoded_output = tokenizer.decode(beam['input_ids'][0], skip_special_tokens=True)
        decoded_outputs.append(decoded_output)

    return decoded_outputs

# Load the BART model and tokenizer
model_name = 'facebook/bart-base'
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

# Example usage
input_text = "Translate this sentence."
beam_size = 3
max_length = 50

output_sentences = beam_search(model, tokenizer, input_text, beam_size, max_length)
print(output_sentences)