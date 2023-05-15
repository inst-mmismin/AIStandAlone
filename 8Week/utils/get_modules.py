from misc.tools import *
from torch.utils.data import DataLoader

from modules.dataset import IMDBDataset
from torch.nn.utils.rnn import pad_sequence

def get_dataset(args): 
    train_dataset = IMDBDataset(args, split='train')
    test_dataset = IMDBDataset(args, split='test', vocab=train_dataset.vocab)
    return train_dataset, test_dataset

def collate_fn(batch): 
    texts, labels = zip(*batch)
    
    # 텍스트 길이를 기준으로 정렬  
    sorted_indices = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
    texts = [torch.LongTensor(texts[i]) for i in sorted_indices]
    labels = [torch.LongTensor([labels[i]]) for i in sorted_indices]

    # 패딩 넣기 
    padded_texts = pad_sequence(texts, batch_first=True)

    # 텐서로 변환 
    padded_texts = torch.LongTensor(padded_texts)
    labels = torch.LongTensor(labels)

    return padded_texts, labels

def get_dataloader(args) : 
    train_dataset, test_dataset = get_dataset(args)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)

    return train_loader, test_loader



def get_model(args) : 
    if args.model == 'transformerEncoder' : 
        from networks.transformers import IMDBClassifier
        model = IMDBClassifier(args)
    else : 
        raise ValueError

    return model