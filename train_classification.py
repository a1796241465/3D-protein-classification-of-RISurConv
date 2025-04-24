import os
import sys
import csv
import torch
import numpy as np
import datetime
import logging
import pandas as pd
import pyvista as pv
import importlib
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset

from test_dataloader import ProteinTestDataset
from train_dataloader import ProteinTrainDataset
import provider

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0,1,2,3', help='specify gpu device')  # 多个gpu这里改
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size in training')  # 24G gpu  batch最大256，根据个数设置batch值256*n
    parser.add_argument('--model', default='risurconv_cls', help='model name')
    parser.add_argument('--num_category', type=int, default=97, help='number of classes')
    parser.add_argument('--epoch', default=450, type=int, help='number of epoch in training')  # 迭代次数
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=True, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=True, help='use uniform sampiling')
    parser.add_argument('--vtk_dir_train', type=str, required=False, help='vtk_root/train/', default='vtk_root/train/')
    parser.add_argument('--vtk_dir_test', type=str, required=False, help='vtk_root/test/', default='vtk_root/test/')
    parser.add_argument('--csv_path_train', type=str, required=False, help='train_set.csv', default='train_set.csv')
    parser.add_argument('--csv_path_test', type=str, required=False, help='test_set.csv', default='test_set.csv')
    parser.add_argument('--pretrain_weight', type=str, default=None, help='pretrained weight path')
    return parser.parse_args()


def test(model, loader, num_class=97):
    mean_correct = []
    model.eval()
    res = []
    for batch in tqdm(loader, total=len(loader)):
        points = batch[0]
        if torch.cuda.is_available():
            points = points.cuda()

        points = points.unsqueeze(0)
        pred = model(points)
        pred_choice = torch.argmax(pred[0], dim=1)
        pred_choice = pred_choice.item()
        print('pred_choice======', pred_choice)
        res.append(pred_choice)
    print('预测标签：')
    print(res)
    try:
        with open('res/output.csv', 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['label'])  
            for item in res:
                writer.writerow([item])
        print("写入成功！")
    except Exception as e:
        print(f"写入失败: {e}")


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('protein_classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')

   
    train_dataset = ProteinTrainDataset(
        vtk_dir=args.vtk_dir_train,  
        csv_path=args.csv_path_train,  
        args=args
    )

    
    test_dataset = ProteinTestDataset(  
        vtk_dir=args.vtk_dir_test,  
        csv_path=args.csv_path_test,  
        args=args
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=10,
        drop_last=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,  
        num_workers=10
    )
    '''MODEL LOADING'''
    model = importlib.import_module(args.model)
    classifier = model.get_model(args.num_category, normal_channel=args.use_normals)
    criterion = model.get_loss()

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        if torch.cuda.device_count() > 1:  
            print(f"使用 {torch.cuda.device_count()} 块GPU (DataParallel)")
            classifier = torch.nn.DataParallel(classifier) 

    '''OPTIMIZER'''
    optimizer = torch.optim.Adam(
        classifier.module.parameters() if hasattr(classifier, 'module') else classifier.parameters(),  
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.decay_rate
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    '''TRAINING'''
    best_acc = 0.0
    for epoch in range(args.epoch):
        classifier.train()
        for batch in tqdm(train_loader, desc=f'Epoch {epoch}'):
            points, target = batch
            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            # 数据增强
            points = points.cpu().numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points).cuda()

            optimizer.zero_grad()
            pred = classifier(points)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

        scheduler.step()

        torch.save({
            'epoch': epoch,
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, 'model_save/best_model.pth')
        test(classifier, test_loader, args.num_category)
       


if __name__ == '__main__':
    args = parse_args()
    main(args)
