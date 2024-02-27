
import torch
import string
import numpy as np

def get_text_from(file_path):
    with open(file_path, 'r', encoding="utf8") as f:
        text=f.read()
    return text

def tokenize(doc):
    # Exclude !"#$%&'()*+,-/:;<=>?@[\]^_`{|}~
    punctuation_to_remove = string.punctuation.replace('.', '')

    # Create translation table that removes specified punctuation except period
    table = str.maketrans('', '', punctuation_to_remove)
    tokens = doc.split()

    # Further split tokens by period and keep periods as separate tokens
    split_tokens = []
    for token in tokens:
        split_tokens.extend(token.replace('.', ' .').split())

    tokens = [w.translate(table) for w in split_tokens]
    tokens = [word for word in tokens if word.isalpha() or word == '.']
    tokens = [word.lower() for word in tokens]

    return tokens

def get_vocabulary(tokens):
    return sorted(set(tokens))

def get_word_to_id(vocabulary):
    return {word:i for i, word in enumerate(vocabulary)}

def get_text_encoded(tokens, word2id):
    return np.array([word2id[word] for word in tokens], dtype=np.int32)

def get_text_chunks(text_encoded, chunk_size=60):
    return [text_encoded[i:i+chunk_size]
               for i in range(len(text_encoded)-chunk_size+1)]


def save_weights(model, dir_base, epoch, best_loss):
    print("New best model.")
    # Save the best model
    weights = "{}/{}.pt".format(dir_base,str("best"))
    chkpt = __get_checkpoint (model, epoch, best_loss) 
    torch.save(chkpt, weights)

def __get_checkpoint(model, epoch, best_loss):
    chkpt = {'epoch': epoch,'model': model.state_dict(), "best_loss": best_loss}
    return chkpt

def get_top_p_sampling(logits, temperature=1.0, top_p=0.9, device="cuda"):
    # Apply temperature scaling
    scaled_logits = logits / temperature

    # Convert logits to probabilities using softmax
    probabilities = torch.softmax(scaled_logits, dim=-1)

    # Sort probabilities and compute cumulative sum
    sorted_indices = torch.argsort(probabilities, descending=True)
    sorted_probabilities = probabilities[sorted_indices]
    cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)

    # Apply top-p filtering
    indices_to_keep = cumulative_probabilities <= top_p
    truncated_probabilities = sorted_probabilities[indices_to_keep]

    # Rescale the probabilities
    truncated_probabilities /= torch.sum(truncated_probabilities)

    # Convert to numpy arrays for random choice
    truncated_probabilities = truncated_probabilities.cpu().numpy()
    sorted_indices = sorted_indices.cpu().numpy()
    indices_to_keep = indices_to_keep.cpu().numpy()

    # Sample from the truncated distribution
    if not indices_to_keep.any():
        # Handle the empty case - for example, using regular sampling without top-p
        probabilities = torch.softmax(logits / temperature, dim=-1)
        next_word_index = torch.multinomial(probabilities, 1).item()
    else:
        # Existing sampling process
        next_word_index = np.random.choice(sorted_indices[indices_to_keep], p=truncated_probabilities)

    return torch.tensor(next_word_index).to(device)