from tqdm import tqdm

def training(model, criterion, optimizer, dataloader, seq_length, device):
    
    print('Training step')
    model.train()
    
    for _, (seq_batch, target_batch) in tqdm(enumerate(dataloader), total=len(dataloader)):
        
        seq_batch = seq_batch.long().to(device)
        target_batch = target_batch.long().to(device)
        hidden, cell = model.init_hidden(seq_batch.size()[0])

        loss = 0
        for w in range(seq_length):
            pred, hidden, cell = model(seq_batch[:, w], hidden, cell)
            loss += criterion(pred, target_batch[:, w])
        
        # Backward and optimize
        for param in model.parameters():
            param.grad = None
        
        loss.backward()
        optimizer.step()

    return pred, hidden, cell, loss