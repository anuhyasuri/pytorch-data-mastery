import torch
from tqdm.auto import tqdm

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
  """Trains Model"""
  model.train()
  running_loss = 0.0
  total_samples = 0
  correct_predictions = 0

  progress_bar = tqdm(dataloader, desc="Training batches", leave=False)

  for batches, (images, labels) in enumerate(progress_bar):
    images, labels = images.to(device), labels.to(device)

    optimizer.zero_grad()
    # Forward pass
    logits = model(images)
    # Calculate loss
    loss = loss_fn(logits)
    # Back propogation
    loss.backward()
    optimizer.step()

    running_loss +=loss.item()*images.size(0)
    _,predictions = torch.max(logits, dim=1)

    correct_predictions += (predictions==labels).sum().item()
  epoch_loss = running_loss / total_samples
  epoch_accuracy = (correct_predictions / total_samples) * 100

  return epoch_loss, epoch_accuracy

def validate_one_epoch(model, dataloader, loss_fn, device):
  """Evaluates Model Performance"""
  model.eval()
  running_loss = 0.0
  total_samples = 0
  correct_predictions = 0

  with torch.no_grad():
    progress_bar = tqdm(dataloader, desc="Validation", leave=False)
    for images, labels in progress_bar:
      images, labels = images.to(device), labels.to(device)
      # Forward pass
      logits = model(images)
      # Calculate loss
      loss = loss_fn(logits)

      running_loss +=loss.item()*images.size(0)
      _,predictions = torch.max(logits, dim=1)

      correct_predictions += (predictions == labels).sum().item()
      total_samples += labels.size(0)
            
  epoch_loss = running_loss / total_samples
  epoch_accuracy = (correct_predictions / total_samples) * 100
    
  return epoch_loss, epoch_accuracy







