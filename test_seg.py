from monai.utils import first, set_determinism

from monai.transforms import AsDiscrete
from networks.UXNet_3D.network_backbone import UXNET
from monai.networks.nets import UNETR, SwinUNETR
from networks.nnFormer.nnFormer_seg import nnFormer
from networks.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, decollate_batch

from networks.swintransformer.swin_transformer3D import SwinTransformer3D

from networks.TransBTSAgent.TransBTS_downsample8x_skipconnection import TransBTSAgent
# from networks.TransBTSAgentTest.TransBTS_downsample8x_skipconnection import TransAgent

# from networks.test5_2.TransBTS_downsample8x_skipconnection import TransBTSAgent
from networks.test6_4.TransBTS_downsample8x_skipconnection import TransAgent
# from networks.TransBTSAgent.Deep.TransBTS_downsample8x_skipconnection import TransBTSAgent

from networks.BiTrUnet.BiTrUnet.BiTrUnet import BiTrUnet
from networks.MISSU.model import MISSU3D

from networks.unetrpp.acdc.unetr_pp_acdc import UNETR_PP
from networks.MISSFormer.MISSFormer import MISSFormer_3D

from networks.DFormer.Dformer import SegNetwork

from networks.NEW.BiTrUnet.BiTrUnet import NEW

from networks.UNET3D.segmentation.unet import UNet3D

from networks.Unet3D.unet3d import UNet

# 消融实验
from networks.ourFlatten.TransBTS_downsample8x_skipconnection import Flatten
from networks.our_noMF.TransBTS_downsample8x_skipconnection import NoMF
from networks.our_noAT.TransBTS_downsample8x_skipconnection import NoAT
from networks.ourSOFT.TransBTS_downsample8x_skipconnection import SOFT

from networks.ConResNet.ConResNet import ConResNet
from networks.VTUnet.vtunet_tumor import VTUNet

from networks.ENC_test1.ENC import ENC

from networks.ENC_test6_8_2.BEFUnet import BEFUnet
import networks.ENC_test6_8_2.BEFUnet_configs as configs

from networks.SparseNet.main import SparseNet

import torch
from load_datasets_transforms import data_loader, data_transforms, infer_post_transforms

import os
import argparse

parser = argparse.ArgumentParser(description='3D UX-Net inference hyperparameters for medical image segmentation')
## Input data hyperparameters
parser.add_argument('--root', type=str, default='', required=True, help='Root folder of all your images and labels')
parser.add_argument('--output', type=str, default='', required=True, help='Output folder for both tensorboard and the best model')
parser.add_argument('--dataset', type=str, default='flare', required=True, help='Datasets: {feta, flare, amos}, Fyi: You can add your dataset here')

## Input model & training hyperparameters
parser.add_argument('--network', type=str, default='3DUXNET', required=True, help='Network models: {TransBTS, nnFormer, UNETR, SwinUNETR, 3DUXNET}')
parser.add_argument('--trained_weights', default='', required=True, help='Path of pretrained/fine-tuned weights')
parser.add_argument('--mode', type=str, default='test', help='Training or testing mode')
parser.add_argument('--sw_batch_size', type=int, default=4, help='Sliding window batch size for inference')
parser.add_argument('--overlap', type=float, default=0.5, help='Sub-volume overlapped percentage')

## Efficiency hyperparameters
parser.add_argument('--gpu', type=str, default='0', help='your GPU number')
parser.add_argument('--cache_rate', type=float, default=0.1, help='Cache rate to cache your dataset into GPUs')
parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

test_samples, out_classes = data_loader(args)

test_files = [
    {"image": image_name} for image_name in zip(test_samples['images'])
]

set_determinism(seed=0)

test_transforms = data_transforms(args)
post_transforms = infer_post_transforms(args, test_transforms, out_classes)

## Inference Pytorch Data Loader and Caching
test_ds = CacheDataset(
    data=test_files, transform=test_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=args.num_workers)

## Load Networks
device = torch.device("cuda:0")
if args.network == '3DUXNET':
    model = UXNET(
        in_chans=1,
        out_chans=out_classes,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        spatial_dims=3,
    ).to(device)

elif args.network == 'SwinUNETR':
    model = SwinUNETR(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=out_classes,
        feature_size=48,
        use_checkpoint=False,
    ).to(device)

elif args.network == 'Sparse':
    model = SparseNet(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=out_classes,
        feature_size=12,
        use_checkpoint=False,
    ).to(device)

elif args.network == 'nnFormer':
    model = nnFormer(input_channels=1, num_classes=out_classes).to(device)

elif args.network == 'UNETR':
    model = UNETR(
        in_channels=1,
        out_channels=out_classes,
        img_size=(96, 96, 96),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    ).to(device)

elif args.network == '3D-UNET':
    model = UNet3D(
        # in_channels=1,
        # out_channels=out_classes,
        # img_size=(96, 96, 96),
        # feature_size=16,
        # hidden_size=768,
        # mlp_dim=3072,
        # num_heads=12,
        # pos_embed="perceptron",
        # norm_name="instance",
        # res_block=True,
        # dropout_rate=0.0,
    ).to(device)

elif args.network == 'TransBTS':
    _, model = TransBTS(dataset=args.dataset, _conv_repr=True, _pe_type='learned')
    model = model.to(device)

elif args.network == 'SwinTransformer3D':
    model = SwinTransformer3D(
        img_size=(96, 96, 96),
        in_chans=1,
        num_classes=out_classes,
    ).to(device)

elif args.network == 'TransBTSAgent':
    _, model = TransBTSAgent(dataset=args.dataset, _conv_repr=True, _pe_type='learned')
    model = model.to(device)

elif args.network == 'MRFTA':
    _, model = TransAgent(dataset=args.dataset, _conv_repr=True, _pe_type='learned')
    model = model.to(device)

elif args.network == 'ENC':
    _, model = ENC(dataset=args.dataset, _conv_repr=True, _pe_type='learned')
    model = model.to(device)

elif args.network == 'ENCTest3':
    CONFIGS = {
        'BEFUnet': configs.get_BEFUnet_configs(),
    }
    model = BEFUnet(config=CONFIGS["BEFUnet"], img_size=96, n_classes=out_classes).to(device)

elif args.network == 'BiTrUnet':
    _, model = BiTrUnet(dataset=args.dataset, _conv_repr=True, _pe_type='learned')
    model = model.to(device)

elif args.network == 'MISSU':
    _, model = MISSU3D(dataset=args.dataset, _conv_repr=True, _pe_type='learned')
    model = model.to(device)

elif args.network == 'NEW':
    _, model = NEW(dataset=args.dataset, _conv_repr=True, _pe_type='learned')
    model = model.to(device)

elif args.network == 'DFormer':
    model = SegNetwork(num_classes=out_classes,deep_supervision=False).to(device)

elif args.network == "unetrpp":
    model = UNETR_PP(in_channels=1,
                             out_channels=out_classes,
                             feature_size=16,
                             num_heads=4,
                             depths=[3, 3, 3, 3],
                             dims=[32, 64, 128, 256],
                             do_ds=False,
                             ).to(device)
    
elif args.network == 'MISSFormer':
    model = MISSFormer_3D(num_classes=out_classes,).to(device)

elif args.network == 'Flatten':
    _, model = Flatten(dataset=args.dataset, _conv_repr=True, _pe_type='learned')
    model = model.to(device)

elif args.network == 'NoMF':
    _, model = NoMF(dataset=args.dataset, _conv_repr=True, _pe_type='learned')
    model = model.to(device)

elif args.network == 'NoAT':
    _, model = NoAT(dataset=args.dataset, _conv_repr=True, _pe_type='learned')
    model = model.to(device)

elif args.network == 'SOFT':
    _, model = SOFT(dataset=args.dataset, _conv_repr=True, _pe_type='learned')
    model = model.to(device)

elif args.network == 'ConResNet':
    model = ConResNet(shape=(96, 96, 96),num_classes=out_classes)
    model = model.to(device)

elif args.network == 'VTUNet':
    model = VTUNet(num_classes=out_classes,).to(device)

elif args.network == 'UNet':
    model = UNet(
        n_channels=1,
        n_classes=out_classes,
    ).to(device)


# 加载并过滤不匹配的 state_dict
# state_dict = torch.load(args.trained_weights)
state_dict = torch.load(args.trained_weights, map_location=device)
model_state_dict = model.state_dict()

# 过滤掉不匹配的键
filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}

# 加载过滤后的 state_dict
model.load_state_dict(filtered_state_dict, strict=False)


# model.load_state_dict(torch.load(args.trained_weights))
model.eval()
with torch.no_grad():
    for i, test_data in enumerate(test_loader):
        images = test_data["image"].to(device)
        roi_size = (96, 96, 96)
        test_data['pred'] = sliding_window_inference(
            images, roi_size, args.sw_batch_size, model, overlap=args.overlap
        )
        test_data = [post_transforms(i) for i in decollate_batch(test_data)]





# import os
# import torch
# from monai.data import CacheDataset, DataLoader, decollate_batch
# from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld, ToTensord
# from monai.networks.nets import UNet
# from monai.inferers import sliding_window_inference
# from argparse import ArgumentParser

# # 解析命令行参数
# parser = ArgumentParser()
# parser.add_argument('--gpu', type=str, default='0', help='指定可见的GPU设备')
# parser.add_argument('--network', type=str, default='3DUXNET', help='网络类型')
# parser.add_argument('--trained_weights', type=str, default='your_model.pth', help='训练好的模型权重路径')
# parser.add_argument('--cache_rate', type=float, default=1.0, help='缓存率')
# parser.add_argument('--num_workers', type=int, default=4, help='数据加载器的线程数')
# parser.add_argument('--sw_batch_size', type=int, default=4, help='滑动窗口批量大小')
# parser.add_argument('--overlap', type=float, default=0.5, help='滑动窗口重叠率')
# args = parser.parse_args()

# # 设置可见的GPU设备
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# # 获取数据
# test_samples, out_classes = data_loader(args)

# # 准备测试文件
# test_files = [{"image": image_name} for image_name in test_samples['images']]

# # 设置确定性
# set_determinism(seed=0)

# # 获取数据变换
# test_transforms = data_transforms(args)
# post_transforms = infer_post_transforms(args, test_transforms, out_classes)

# # 初始化缓存数据集和数据加载器
# test_ds = CacheDataset(
#     data=test_files, transform=test_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers
# )
# test_loader = DataLoader(test_ds, batch_size=1, num_workers=args.num_workers)

# # 加载网络
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# if args.network == '3DUXNET':
#     model = UXNET(
#         in_chans=1,
#         out_chans=out_classes,
#         depths=[2, 2, 2, 2],
#         feat_size=[48, 96, 192, 384],
#         drop_path_rate=0,
#         layer_scale_init_value=1e-6,
#         spatial_dims=3,
#     ).to(device)

# # 加载并过滤不匹配的 state_dict
# state_dict = torch.load(args.trained_weights, map_location=device)
# model_state_dict = model.state_dict()

# # 过滤掉不匹配的键
# filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}

# # 加载过滤后的 state_dict
# model.load_state_dict(filtered_state_dict, strict=False)

# # 设置模型为评估模式
# model.eval()

# with torch.no_grad():
#     for i, test_data in enumerate(test_loader):
#         images = test_data["image"].to(device)
#         roi_size = (96, 96, 96)
#         test_data['pred'] = sliding_window_inference(
#             images, roi_size, args.sw_batch_size, model, overlap=args.overlap
#         )
#         test_data = [post_transforms(i) for i in decollate_batch(test_data)]
