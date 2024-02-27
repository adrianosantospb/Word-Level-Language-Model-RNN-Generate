from torch.utils.data import Dataset

class FairyTalesDataset(Dataset):
    
    def __init__(self, text_chunked):
        self.text_chunked = text_chunked

    def __len__(self):
        return len(self.text_chunked)

    def __getitem__(self, idx):
        text_chunk = self.text_chunked[idx]
        return text_chunk[:-1], text_chunk[1:]