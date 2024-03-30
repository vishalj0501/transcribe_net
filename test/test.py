import string

import torch
import torchvision.transforms as T

from data_preprocess.tokenizer import Tokenizer
from model.model import make_model
from data_preprocess.data_gen_spanish import DataGenerator_Spanish, crop_dict



charset_base = string.printable[:95]
tokenizer = Tokenizer(charset_base)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_memory(model,imgs):
    x = model.conv(model.get_feature(imgs))
    bs,_,H, W = x.shapex
    pos = torch.cat([
            model.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            model.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

    return model.transformer.encoder(pos +  0.1 * x.flatten(2).permute(2, 0, 1))
    

def test(model, test_loader, max_text_length):
    model.eval()
    predicts = []
    gt = []
    imgs = []
    with torch.no_grad():
        for batch in test_loader:
            src, trg = batch
            imgs.append(src.flatten(0,1))
            src, trg = src.cuda(), trg.cuda()            
            memory = get_memory(model,src.float())
            out_indexes = [tokenizer.chars.index('SOS'), ]
            for i in range(max_text_length):
                mask = model.generate_square_subsequent_mask(i+1).to('cuda')
                trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(device)
                output = model.vocab(model.transformer.decoder(model.query_pos(model.decoder(trg_tensor)), memory,tgt_mask=mask))
                out_token = output.argmax(2)[-1].item()
                out_indexes.append(out_token)
                if out_token == tokenizer.chars.index('EOS'):
                    break
            predicts.append(tokenizer.decode(out_indexes))
            gt.append(tokenizer.decode(trg.flatten(0,1)))
    return predicts, gt, imgs


#inference
if __name__ == "__main__":
    pretrained_model = make_model(vocab_len = 100)
    _=pretrained_model.to(device)

    pretrained_model.load_state_dict(torch.load('weights/span_fine_tuned_model.pt'))
    img_path = "spanish_test_dataset/cropped_pages/27.jpg"
    max_text_length = 128
    transform = T.Compose([T.ToTensor()])
    sp_loader = torch.utils.data.DataLoader(DataGenerator_Spanish(crop_dict(img_path),charset_base,max_text_length ,transform, shuffle=False), batch_size=1, shuffle=False, num_workers=2)
    predicts2, gt2, imgs = test(pretrained_model, sp_loader, max_text_length)

    predicts2 = list(map(lambda x : x.replace('SOS','').replace('EOS',''),predicts2))
    
    final_pred_str = ""
    for s in predicts2:
        final_pred_str += s+"\n"
        
    print(final_pred_str)


