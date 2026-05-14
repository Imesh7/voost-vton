

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    
    for batch in dataloader:
        # Move data to the appropriate device
        images = batch['images'].to(device)
        tasks = batch['tasks'].to(device)
        time_emb = batch['time_emb'].to(device)
        
        # Forward pass
        outputs = model(tasks, images, time_emb)
        
        # Compute loss
        loss = criterion(outputs, batch['labels'].to(device))
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    average_loss = total_loss / len(dataloader)
    return average_loss