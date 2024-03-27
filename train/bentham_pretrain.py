import time
import string

import torch
import torchvision.transforms as T
import numpy as np

from data_preprocess.tokenizer import Tokenizer
from model.model import make_model
from model.label import LabelSmoothing
from evaluate import evaluate
from data_preprocess.data_gen_bentham import DataGenerator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
charset_base = string.printable[:95]
tokenizer = Tokenizer(charset_base)
model = make_model(vocab_len = tokenizer.vocab_size)
_=model.to(device)
lr = .0001 # learning rate

transform = T.Compose([
    T.ToTensor()])

batch_size = 16
epochs = 200

optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=.0004)
criterion = LabelSmoothing(size=tokenizer.vocab_size, padding_idx=0, smoothing=0.1)
criterion.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=.0004)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

source_path = "data_bentham/bentham/data_transform_bentham/bentham.hdf5"
target_path = "data_bentham/bentham/pretrain.pth"

max_text_length = 128

train_loader = torch.utils.data.DataLoader(DataGenerator(source_path,charset_base,max_text_length,'train',transform), batch_size=batch_size, shuffle=False, num_workers=2)
val_loader = torch.utils.data.DataLoader(DataGenerator(source_path,charset_base,max_text_length,'valid',transform), batch_size=batch_size, shuffle=False, num_workers=2)


def train(model, criterion, optimiser, scheduler,dataloader):
    
    model.train()
    total_loss = 0
    for batch, (imgs, labels_y,) in enumerate(dataloader):
          imgs = imgs.to(device)
          labels_y = labels_y.to(device)
    
          optimiser.zero_grad()
          output = model(imgs.float(),labels_y.long()[:,:-1])
 
          norm = (labels_y != 0).sum()
          print("Output shape:", output.log_softmax(-1).contiguous().view(-1, tokenizer.vocab_size).shape)
          loss = criterion(output.log_softmax(-1).contiguous().view(-1, tokenizer.vocab_size), labels_y[:,1:].contiguous().view(-1).long()) / norm
 
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2)
          optimizer.step()
          total_loss += loss.item() * norm
 
    return total_loss / len(dataloader)
 


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
 
best_valid_loss = np.inf
c = 0
for epoch in range(200):
 
    print(f'Epoch: {epoch+1:02}','learning rate{}'.format(scheduler.get_last_lr()))
    
    start_time = time.time()
 
    train_loss = train(model,  criterion, optimizer, scheduler, train_loader)
    valid_loss = evaluate(model, criterion, val_loader)
 
    epoch_mins, epoch_secs = epoch_time(start_time, time.time())
 
    c+=1
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), target_path)
        c=0
 
    if c>4:
        scheduler.step() #decrease lr if loss does not decrease after 5 steps
        c=0
 
    print(f'Time: {epoch_mins}m {epoch_secs}s') 
    print(f'Train Loss: {train_loss:.3f}')
    print(f'Val   Loss: {valid_loss:.3f}')

torch.save(model.state_dict(), 'weights/bentham_pretrained_model.pt')