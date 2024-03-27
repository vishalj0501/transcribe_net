import time
import string
import os

import torch
import torchvision.transforms as T
import numpy as np
import cv2
import pytesseract
import pandas as pd

from data_preprocess.tokenizer import Tokenizer
from model.model import make_model
from model.label import LabelSmoothing
from evaluate import evaluate
from data_preprocess.data_gen_spanish import DataGenerator_Spanish
from data_preprocess.data_gen_spanish import crop_dict



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pretrained_model = make_model(vocab_len = 100)
_=pretrained_model.to(device)
charset_base = string.printable[:95]
tokenizer = Tokenizer(charset_base)
max_text_length = 128   

pretrained_model.load_state_dict(torch.load('weights/bentham_pretrained_model.pt'))
target_path = "output/span/fine_tune.pth"

criterion = LabelSmoothing(size=100, padding_idx=0, smoothing=0.1)
criterion.to(device)

transform = T.Compose([
    T.ToTensor()])

lr = .0001 # learning rate
optimizer = torch.optim.AdamW(pretrained_model.parameters(), lr=lr,weight_decay=.0004)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)


cropped = os.listdir("spanish_test_dataset/cropped_pages")
cropped_sorted = sorted([filename for filename in cropped if filename.endswith('.jpg')], key=lambda x: int(os.path.splitext(x)[0]))
print(cropped_sorted)
c_list = []

for i in cropped_sorted:
    c_list.append(crop_dict(f"/spanish_test_dataset/cropped_pages/{i}"))

c_dict_train = {}

for c_dict in c_list[:22]:
    for key, value in c_dict.items():
        if key in c_dict_train:
            c_dict_train[key].extend(value)
        else:
            c_dict_train[key] = value[:]

c_dict_valid = {}

for c_dict in c_list[22:28]:
    for key, value in c_dict.items():
        if key in c_dict_valid:
            c_dict_valid[key].extend(value)
        else:
            c_dict_valid[key] = value[:]


span_train_loader = torch.utils.data.DataLoader(DataGenerator_Spanish(c_dict_train, charset_base,max_text_length,transform), batch_size=1, shuffle=False, num_workers=2)
span_valid_loader =  torch.utils.data.DataLoader(DataGenerator_Spanish(c_dict_valid, charset_base,max_text_length,transform), batch_size=1, shuffle=False, num_workers=2)

def train(model, criterion, optimiser, scheduler,dataloader):
 
    model.train()
    total_loss = 0
    for batch, (imgs, labels_y,) in enumerate(dataloader):
          imgs = imgs.to(device)
          labels_y = labels_y.to(device)
    
          optimiser.zero_grad()
          output = model(imgs.float(),labels_y.long()[:,:-1])
 
          norm = (labels_y != 0).sum()
#           print("Output shape:", output.log_softmax(-1).contiguous().view(-1, tokenizer.vocab_size+1).shape)
          loss = criterion(output.log_softmax(-1).contiguous().view(-1, tokenizer.vocab_size+1), labels_y[:,1:].contiguous().view(-1).long()) / norm
 
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
 

train_losses = []
valid_losses = []
best_valid_loss = np.inf
c = 0


for epoch in range(150):
 
    print(f'Epoch: {epoch+1:02}','learning rate{}'.format(scheduler.get_last_lr()))
    
    start_time = time.time()
 
    train_loss = train(pretrained_model,  criterion, optimizer, scheduler, span_train_loader)
    valid_loss = evaluate(pretrained_model, criterion, span_valid_loader)
 
    epoch_mins, epoch_secs = epoch_time(start_time, time.time())
 
    c+=1
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(pretrained_model.state_dict(), target_path)
        c=0
 
    if c>4:
        scheduler.step()
        c=0
 
    print(f'Time: {epoch_mins}m {epoch_secs}s') 
    print(f'Train Loss: {train_loss:.3f}')
    print(f'Val   Loss: {valid_loss:.3f}')
    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

        
torch.save(pretrained_model.state_dict(), 'span_trained_model.pt')
