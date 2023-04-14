# 필요 패키지 업로드 
import os 
import json 
import torch 
from PIL import Image
import torch.nn.functional as F

from utils.parser import infer_parse_args, load_args_from_dict
from utils.get_modules import get_model, get_cifar_transform
from misc.tools import cifar_classes

def main() : 
    # args 세팅 및 유효성 확인 
    args = infer_parse_args()
    print(args)

    assert os.path.exists(args.trained_folder), 'trained_folder does not exist'
    assert os.path.exists(args.image_path), 'image_path does not exist'

    # 학습 상황에서 사용한 학습 세팅 값 불러오기 
    with open(os.path.join(args.trained_folder, 'args.json'), 'r') as f:
        train_args = load_args_from_dict(json.load(f))

    # 모델 불러오기 
    # 1. 구조 불러오기 
    model = get_model(train_args).to(args.device)
    # 2. 가중치 불러오기
    model.load_state_dict(
        torch.load(
            os.path.join(args.trained_folder, 'best_model.pth')
        )
    )

    # 데이터 전처리 및 추론을 위한 데이터 준비하기
    target_img = Image.open(args.image_path)
    trans = get_cifar_transform(train_args)
    
    input_img = trans(target_img).unsqueeze(0).to(args.device)

    # 모델 추론하기 
    model.eval()
    with torch.no_grad():
        output = model(input_img)

    # 결과 후처리 하기 
    output = F.softmax(output)
    conf, pred = torch.max(output, 1)
    conf = conf.item()
    pred = cifar_classes[pred.item()]

    print(f'Prediction: {pred} with confidence {conf*100:.2f}%')


if __name__ == '__main__':
    main()