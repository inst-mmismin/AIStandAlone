import torch.nn as nn 
from torch.optim import Adam 

from utils.parser import parse_args
from utils.get_modules import get_dataloader, get_model

from misc.tools import eval, eval_by_class


def main() : 
    # 하이퍼파라메터 설정 -> argparse 사용
    args = parse_args()
    print(args)

    # 데이터셋 불러오기
    train_loader, test_loader = get_dataloader(args)

    # 모델 가져오기 
    model = get_model(args)

    # Loss, Optimizer 생성 
    criteria = nn.CrossEntropyLoss() 
    optim = Adam(model.parameters(), lr=args.lr)

    # 학습 loop 
    losses = []
    accs1 = []
    accs2 = []

    for epoch in range(args.epochs) : 
        for idx, (image, label) in enumerate(train_loader) : 
            image = image.to(args.device)
            label = label.to(args.device) 

            output = model(image)
            loss = criteria(output, label)
            optim.zero_grad()
            loss.backward() 
            optim.step() 
            
            if idx % 100 == 0: 
                print(f'{epoch}/{args.epochs} , {idx} step | Loss : {loss.item():.4f}')
                losses.append(loss.item())
                accs1.append(eval(model, test_loader, args))
                # accs2.append(eval_by_class(model, test_loader, args)[1])

    
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.plot(accs1)
    plt.plot(accs2)
    plt.show()


if __name__ == '__main__' : 
    main()