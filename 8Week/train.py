# 필요 패키지 및 라이브러리 import
import os
import json 
import tqdm 

import torch 
import torch.nn as nn
from torch.optim import Adam 

from utils.parser import parse_args
from utils.get_path import get_save_folder_path
from utils.get_modules import get_dataloader, get_model
from misc.tools import evaluation

def main():
    # 세팅값 설정 
    args = parse_args()
    print(args)

    # 저장 폴더 생성 및 args 저장 
    save_folder_path = get_save_folder_path(args)
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    with open(os.path.join(save_folder_path, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    # 데이터셋 불러오기
    train_loader, test_loader = get_dataloader(args)

    # 모델 가져오기
    model = get_model(args).to(args.device)
    
    # Loss, Optimizer 생성
    criteria = nn.CrossEntropyLoss() 
    optim = Adam(model.parameters(), lr=args.lr)

    # 학습 시작
    best_acc = 0 
    for epoch in range(args.epochs):
        for i, (text, labels) in enumerate(tqdm.tqdm(train_loader)):
            text = text.to(args.device)
            labels = labels.to(args.device)

            outputs = model(text)
            loss = criteria(outputs, labels)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if (i+1) % args.save_itv == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                acc = evaluation(model, test_loader, args)
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), os.path.join(save_folder_path, 'best_model.pth'))
                    print(f'Best model saved!! accuracy : {acc*100:.2f} %')

if __name__ == '__main__':
    main()