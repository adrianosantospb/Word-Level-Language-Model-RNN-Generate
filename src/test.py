import torch
from src.core.helpers import get_top_p_sampling

def generate(model, tokenize, word2id, word_array, seed_str, len_generated_text=50, temperature=1, top_p=0.95, device="cuda"):

    seed_tokens = tokenize(seed_str)

    encoded_input = torch.tensor([word2id[t] for t in seed_tokens])
    encoded_input = torch.reshape(encoded_input, (1, -1)).to(device)

    generated_str = seed_str

    model.eval()
    with torch.inference_mode():
      hidden, cell = model.init_hidden(1)
      hidden = hidden.to(device)
      cell = cell.to(device)
      for w in range(len(seed_tokens)-1):
          _, hidden, cell = model(encoded_input[:, w].view(1), hidden, cell)

      last_word = encoded_input[:, -1]
      for i in range(len_generated_text):
          logits, hidden, cell = model(last_word.view(1), hidden, cell)
          logits = torch.squeeze(logits, 0)
          last_word = get_top_p_sampling(logits, temperature, top_p)
          generated_str += " " + str(word_array[last_word])

    return generated_str.replace(" . ", ". ")