import re 
import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer # pip install torchtext==0.14.1 필요 


def preprocess_text(text):
    text = text.lower() # 소문자 변환 
    text = re.sub(r"[^\w\s]", "", text) # 특수문자 제거 
    text = re.sub(r"<.*?>", "", text) # HTML태그 제거 
    text = re.sub(r"\s+", " ", text) # 연속된 띄어쓰기 제거
    text = text.strip() # 문장 앞뒤 불필요한 공백 제거
    return text 

class IMDBDataset(Dataset):
    def __init__(self, args, split='train', vocab=None):
        # 데이터 불러오기 
        raw_dataset = load_dataset('imdb', split=split)
        
        # 전처리
        self.tokenizer = get_tokenizer('basic_english')

        self.texts = [] 
        self.token_texts = [] 
        self.all_tokens = [] 
        
        prog = tqdm.tqdm(raw_dataset['text'])
        for line in prog:
            prog.set_description(f"Processing {split} dataset..")
            proc_line = preprocess_text(line)
            self.texts.append(proc_line)

            tokend_line = self.tokenizer(proc_line)
            tokend_line = tokend_line[:args.max_length]
            self.token_texts.append(tokend_line)
            
            self.all_tokens += tokend_line
        
        # unknown 단어 처리
        self.all_tokens.append('<unk>')
        self.labels = raw_dataset['label']

        self.vocab = list(set(self.all_tokens)) if split == 'train' else vocab
        self.vocab_size = len(self.vocab)
        args.vocab_size = self.vocab_size

        self.word2idx = {word:idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx:word for idx, word in enumerate(self.vocab)}

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        tokened_line = self.token_texts[idx]
        idxed_line = [self.word2idx.get(word, self.word2idx['<unk>']) for word in tokened_line]
        label = self.labels[idx]
        
        return idxed_line, label