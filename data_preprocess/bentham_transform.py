
import os
import html
import string
import multiprocessing
import xml.etree.ElementTree as ET
from preprocess import text_standardize, preprocess

from glob import glob
from tqdm import tqdm
from functools import partial
import h5py
from tokenizer import Tokenizer



class Dataset():
    """Dataset class to read images and sentences from base (raw files)"""

    def __init__(self, source, name):
        self.source = source
        self.name = name
        self.dataset = None
        self.partitions = ['train', 'valid', 'test']

    def read_partitions(self):
        """Read images and sentences from dataset"""

        dataset = getattr(self, f"_{self.name}")()

        if not self.dataset:
            self.dataset = dict()

            for y in self.partitions:
                self.dataset[y] = {'dt': [], 'gt': []} 

        for y in self.partitions:
            self.dataset[y]['dt'] += dataset[y]['dt']
            self.dataset[y]['gt'] += dataset[y]['gt']

    def preprocess_partitions(self, input_size):
        """Preprocess images and sentences from partitions"""

        for y in self.partitions:
            arange = range(len(self.dataset[y]['gt']))

            for i in reversed(arange):
                text = text_standardize(self.dataset[y]['gt'][i])

                if not self.check_text(text):
                    self.dataset[y]['gt'].pop(i)
                    self.dataset[y]['dt'].pop(i)
                    continue

                self.dataset[y]['gt'][i] = text.encode()

            results = []
            with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
                print(f"Partition: {y}")
                for result in tqdm(pool.imap(partial(preprocess, input_size=input_size), self.dataset[y]['dt']),
                                   total=len(self.dataset[y]['dt'])):
                    results.append(result)
                pool.close()
                pool.join()

            self.dataset[y]['dt'] = results
        
    @staticmethod
    def check_text(text):
        """Make sure text has more characters instead of punctuation marks"""

        strip_punc = text.strip(string.punctuation).strip()
        no_punc = text.translate(str.maketrans("", "", string.punctuation)).strip()

        if len(text) == 0 or len(strip_punc) == 0 or len(no_punc) == 0:
            return False

        punc_percent = (len(strip_punc) - len(no_punc)) / len(strip_punc)

        return len(no_punc) > 2 and punc_percent <= 0.1
    

    def _bentham(self):
        """Bentham dataset reader"""
        source = os.path.join(self.source, "BenthamDatasetR0-GT")
        pt_path = os.path.join(self.source, "Partitions")

        paths = {"train": open(os.path.join(pt_path, "TrainLines.lst")).read().splitlines(),
                 "valid": open(os.path.join(pt_path, "ValidationLines.lst")).read().splitlines(),
                 "test": open(os.path.join(pt_path, "TestLines.lst")).read().splitlines()}

        transcriptions = os.path.join(source, "Transcriptions")
        gt = os.listdir(transcriptions)
        gt_dict = dict()

        for index, x in enumerate(gt):
            text = " ".join(open(os.path.join(transcriptions, x)).read().splitlines())
            text = html.unescape(text).replace("<gap/>", "")
            gt_dict[os.path.splitext(x)[0]] = " ".join(text.split())

        img_path = os.path.join(source, "Images", "Lines")
        dataset = dict()

        for i in self.partitions:
            dataset[i] = {"dt": [], "gt": []}

            for line in paths[i]:
                dataset[i]['dt'].append(os.path.join(img_path, f"{line}.png"))
                dataset[i]['gt'].append(gt_dict[line])

        return dataset
    



if __name__ == "__main__":

    raw_path = os.path.join( "data_bentham", "bentham")
    source_path = os.path.join("data_bentham", "bentham" ,"data_transform_bentham", f"{"bentham"}.hdf5")
    input_size = (1024, 128, 1)
    max_text_length = 128
    charset_base = string.printable[:95]
    tokenizer = Tokenizer(chars=charset_base, max_text_length=max_text_length)            

    ds = Dataset(source=raw_path, name="bentham")
    ds.read_partitions()

    print("Partitions will be preprocessed...")
    ds.preprocess_partitions(input_size=input_size)

    print("Partitions will be saved...")
    os.makedirs(os.path.dirname(source_path), exist_ok=True)

    for i in ds.partitions:
        with h5py.File(source_path, "a") as hf:
            hf.create_dataset(f"{i}/dt", data=ds.dataset[i]['dt'], compression="gzip", compression_opts=9)
            hf.create_dataset(f"{i}/gt", data=ds.dataset[i]['gt'], compression="gzip", compression_opts=9)
            print(f"[OK] {i} partition.")

    print(f"Transformation finished.")