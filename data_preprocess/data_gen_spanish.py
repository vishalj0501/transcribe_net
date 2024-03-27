import os


from torch.utils.data import Dataset
import torch
import cv2
import numpy as np
import pytesseract
import pandas as pd

from tokenizer import Tokenizer
from preprocess import preprocess, normalization


class DataGenerator_Spanish(Dataset):
    def __init__(self, source_dict, charset, max_text_length, transform):
        self.tokenizer = Tokenizer(charset, max_text_length)
        self.transform = transform
        
        self.dataset = source_dict.copy() 
        
        randomize = np.arange(len(self.dataset['gt']))
        np.random.seed(42)
        np.random.shuffle(randomize)

        self.dataset['dt'] = np.array(self.dataset['dt'])[randomize]
        self.dataset['gt'] = np.array(self.dataset['gt'])[randomize]

        self.dataset['gt'] = [x.decode() for x in self.dataset['gt']]
            
        self.size = len(self.dataset['gt'])

    def __getitem__(self, i):
        img = self.dataset['dt'][i]
    
        img = np.repeat(img[..., np.newaxis], 3, -1)    
        img = normalization(img)
        
        if self.transform is not None:
            img = self.transform(img)

        y_train = self.tokenizer.encode(self.dataset['gt'][i]) 
 
        y_train = np.pad(y_train, (0, self.tokenizer.maxlen - len(y_train)))

        gt = torch.Tensor(y_train)

        return img, gt          

    def __len__(self):
        return self.size


def crop_dict(page):

    page = cv2.imread(page)
    master_page_par_line_list = []

    image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2GRAY)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    data = pytesseract.image_to_data(image,config='--oem 3 --psm 6', output_type='dict')

    page_num = 1

    df = pd.DataFrame(data)
    df = df[df["conf"] > 0]
    df["page_num"] = page_num

    page_par_line_dict = {}
    for index, row in df.iterrows():
        page_par_line = f"{page_num}_{row['par_num']}_{row['line_num']}"
        if(page_par_line not in page_par_line_dict):
            page_par_line_dict[page_par_line] = {"text": str(row["text"]) + " ", "box": (row['left'], row['top'], row['left'] + row['width'], row['top'] + row['height'])}
        else:
            page_par_line_dict[page_par_line]["text"] = page_par_line_dict[page_par_line]["text"] + str(row["text"]) + " "
            page_par_line_dict[page_par_line]['box'] = (min(page_par_line_dict[page_par_line]['box'][0], row['left']), 
                                                    min(page_par_line_dict[page_par_line]['box'][1], row['top']), 
                                                    max(page_par_line_dict[page_par_line]['box'][2], row['left'] + row['width']), 
                                                    max(page_par_line_dict[page_par_line]['box'][3], row['top'] + row['height']))


    for entry in page_par_line_dict:
        splitted_key = entry.split('_')
        entry_value = page_par_line_dict[entry]
        master_page_par_line_list.append({
            'page_number' : splitted_key[0],
            'paragraph_number' : splitted_key[1],
            'line_number' : splitted_key[2],
            'entry_text' : entry_value['text'],
            'bounding_box' : entry_value['box']
        })

    imgs_cropped = {}


    img_text_dict = {"dt" : [], "gt" : []}

    for line in page_par_line_dict.values():
        if line['box'] is not None:
            cv2.rectangle(image, (line['box'][0], line['box'][1]), (line['box'][2], line['box'][3]), (0, 0, 255), 2)
            img_cropped = image[line['box'][1]:line['box'][3], line['box'][0]:line['box'][2]]
            if not os.path.exists('cropped_lines'):
                os.makedirs('cropped_lines')
            cv2.imwrite(f"cropped_lines/{line['box'][1]}.jpg", img_cropped)
#             print(line['text'])
            imgs_cropped[line['box'][1]] = img_cropped 
            assert os.path.exists(f'cropped_lines/{line["box"][1]}.jpg')
            img_text_dict["dt"].append(preprocess(f"cropped_lines/{line['box'][1]}.jpg",(1024,128,1)))
            img_text_dict["gt"].append(line['text'].encode())

    return img_text_dict
