import pickle
import torch
import sys
import torch.nn as nn
from glob import glob
from torch import optim
import PIL.Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
CenterCropVideo,
NormalizeVideo,
)
from torchvision import transforms
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
ApplyTransformToKey,
ShortSideScale,
UniformTemporalSubsample,
UniformCropVideo,
RandomResizedCrop
)
from torch import nn, Tensor

import random
import cv2
import os
from gen_gif import make_gif
import clip
import torch.nn.functional as F
from einops import rearrange
# from mmflow.apis import inference_model, init_model
import math
import numpy as np
import timm
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_utils import training_stats
from torch.nn.parallel import DistributedDataParallel as DDP

print(torch.__version__)
import time
import lpips

from itertools import chain

from collections import namedtuple
from collections import OrderedDict

from torch.nn import Conv2d, BatchNorm2d, PReLU, ReLU, Sigmoid, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Dropout, Sequential, Module
import torchvision.transforms.functional as TF

def make_transform(translate, angle):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

"""
ArcFace implementation from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch)
"""

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    """ A named tuple describing a ResNet block. """


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    else:
        raise ValueError("Invalid number of layers: {}. Must be one of [50, 100, 152]".format(num_layers))
    return blocks



class MakeCutouts(torch.nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)



class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Discriminator2D(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator2D, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout2d(0.25),
            # state size (ndf) x 128 x 128
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout2d(0.25),
            # state size (ndf * 2) x 64 x 64
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout2d(0.25),
            # state size (ndf * 4) x 32 x 32
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout2d(0.25),
            # state size (ndf * 8) x 16 x 16
            nn.Conv2d(ndf*8, ndf*8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout2d(0.25),
            
            # state size (ndf * 8) x 8 x 8
            nn.Conv2d(ndf*8, ndf*8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            #nn.Dropout2d(0.25),
            # state size (ndf * 8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output 

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=32, T=5, ngpu=1):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
        # input is (nc) x T x 256 x 256
        nn.Conv3d(nc, ndf, (1, 3, 3), (1, 2, 2), (0, 1, 1)),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout3d(0.25),
        # state size. (ndf) x T x 128 x 128
        nn.Conv3d(ndf, ndf * 2, (1, 3, 3), (1, 2, 2), (0, 1, 1)),
        nn.InstanceNorm3d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout3d(0.25),
        # state size. (ndf*2) x T x 64 x 64
        nn.Conv3d(ndf * 2, ndf * 4, (1, 3, 3), (1, 2, 2), (0, 1, 1)),
        nn.InstanceNorm3d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout3d(0.25),
        # state size. (ndf*4) x T x 32 x 32
        nn.Conv3d(ndf * 4, ndf * 8, (1, 3, 3), (1, 2, 2), (0, 1, 1)),
        nn.InstanceNorm3d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout3d(0.25),


        nn.Conv3d(ndf * 8, ndf * 8, (1, 1, 1), 1, 0),

        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class SceneVideoDataset(Dataset):
    def __init__(self):
        self.video_paths = glob("./datasets/scene_video/splashing_water/*.mp4") + glob("./datasets/scene_video/squishing_water/*.mp4") + glob("./datasets/scene_video/thunder/*.mp4") + glob("./datasets/scene_video/waterfall_burbling/*.mp4") + glob("./datasets/scene_video/volcano/*.mp4")+ glob("./datasets/scene_video/raining/*.mp4")

        self.audio_paths = glob("./datasets/curation/dataset_curation/*.npy")
        self.side_size = 256
        self.mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        self.crop_size = 256
        self.num_frames = 5
        self.sampling_rate = 2
        self.frames_per_second = 1
        self.alpha = 4
        self.time_length = 864
        self.n_mels = 128
        self.width_resolution = 512 // 2
        self.frame_per_audio = self.time_length // self.num_frames
        self.transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(self.num_frames),
                    
                    # NormalizeVideo(self.mean, std),
                    ShortSideScale(
                    size=self.side_size
                    ),
                    CenterCropVideo(crop_size=(self.crop_size, self.crop_size)),
                    Lambda(lambda x: x / 127.5 - 1.0),
                ]
            ),
        )

    def __getitem__(self, idx):
        try:
            video_path = self.video_paths[idx]
            

            file_name = video_path.split("/")[-1].split(".")[0]
            file_name = os.path.join('./datasets/curation/dataset_curation', file_name)
            file_name = file_name + '.npy'
            index = self.audio_paths.index(file_name)
            npy_name = self.audio_paths[index]
            audio_inputs = np.load(npy_name, allow_pickle=True)
            c, h, w = audio_inputs.shape
            if w >= self.time_length:
                j = random.randint(0, w-self.time_length)
                full_audio = audio_inputs[:,:,j:j+self.time_length]
            elif w < self.time_length:
                zero = np.zeros((1, self.n_mels, self.time_length))
                j = random.randint(0, self.time_length - w - 1)
                zero[:,:,j:j+w] = audio_inputs[:,:,:w]
                full_audio = zero

            full_audio = cv2.resize(full_audio[0], (self.n_mels, self.width_resolution))
            full_audio = torch.from_numpy(full_audio).float()

            return full_audio, video_path.split("/")[3]
        except Exception as e:
            print("Wo ", e)
            return self.__getitem__((idx+1) % len(self.video_paths))

    def __len__(self):
        return len(self.video_paths)


class MeadVideoDataset(Dataset):
    def __init__(self):
        self.video_paths = glob("/home/lsh/Downloads/MEAD/*/video/front/*/*/*.mp4")
        
        
        self.side_size = 384
        self.mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        self.crop_size = 256
        self.num_frames = 5
        self.sampling_rate = 2
        self.frames_per_second = 1
        self.alpha = 4
        self.time_length = 864
        self.n_mels = 128
        self.width_resolution = 512
        self.frame_per_audio = self.time_length // self.num_frames
        self.transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(self.num_frames),
                    
                    ShortSideScale(
                    size=self.side_size
                    ),
                    CenterCropVideo(crop_size=(self.crop_size, self.crop_size)),
                    Lambda(lambda x: x / 127.5 - 1.0),
                ]
            ),
        )

    def __getitem__(self, idx):
        try:
            video_path = self.video_paths[idx]
            
            video = EncodedVideo.from_path(video_path)
            clip_start_sec = 0.0 # secs
            clip_duration = 2.0 # secs
            video_data = video.get_clip(start_sec=clip_start_sec, end_sec=clip_start_sec + clip_duration)
            video_data = self.transform(video_data)

        
            return video_data["video"], 0, 0, video_path.split("/")[-3]
        except Exception as e:
            print("Wo ", e)
            return self.__getitem__((idx+1) % len(self.video_paths))

    def __len__(self):
        return len(self.video_paths)

class SubUrmpVideoDataset(Dataset):
    def __init__(self):
        self.video_paths = sorted(os.listdir("./datasets/Sub-URMP/img/train/"))
        print(self.video_paths)
        self.side_size = 256
        self.mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        self.crop_size = 256
        self.num_frames = 8
        self.sampling_rate = 2
        self.frames_per_second = 1
        self.alpha = 4
        self.transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0)])

    def __getitem__(self, idx):
        try:

            video_lists = sorted(glob(os.path.join("./datasets/Sub-URMP/img/train/", self.video_paths[idx]) + "/*"), key=lambda x : int(x.split("_")[-1].split(".")[0]))
            frame_num = random.randint(0, len(video_lists))
            name = video_lists[frame_num].split("/")[-1].split("_")[0]

            video_lists = sorted(glob(f"./datasets/Sub-URMP/img/train/*/{name}_*.jpg"), key=lambda x : int(x.split("_")[-1].split(".")[0]))

            frame_num = random.randint(0, len(video_lists)-self.num_frames)

            video_sequence = []
            for f_idx in range(frame_num, frame_num+self.num_frames):
                frame = self.transform(PIL.Image.open(video_lists[f_idx]))
                frame_name = video_lists[f_idx].split("/")[-1].split("_")[0]
                if name!=frame_name:
                    print(name, frame_name)
                video_sequence.append(frame.unsqueeze(0))

            video_sequence = torch.cat(video_sequence)
            return video_sequence
        except:
            return self.__getitem__(idx)
    def __len__(self):
        return len(self.video_paths)

class LandscapeVideoDataset(Dataset):
    def __init__(self):
        self.video_paths = sorted(os.listdir("./datasets/scene_frames/splashing_water/")) + sorted(os.listdir("./datasets/scene_frames/thunder/")) + sorted(os.listdir("./datasets/scene_frames/squishing_water/")) + sorted(os.listdir("./datasets/scene_frames/waterfall_burbling/")) + sorted(os.listdir("./datasets/scene_frames/wind_noise/"))
        # print(self.video_paths)
        self.side_size = 256
        self.mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        self.crop_size = 256
        self.num_frames = 5
        self.sampling_rate = 2
        self.frames_per_second = 1
        self.alpha = 4
        self.transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0)])

    def __getitem__(self, idx):
        try:
            # print(self.video_paths[idx].split("/"))

            # video_lists = sorted(glob(os.path.join("./datasets/scene_frames/*/", self.video_paths[idx]) + "/*"), key=lambda x : int(x.split("_")[-1].split(".")[0]))
            name = self.video_paths[idx].split("/")[0]
            video_lists = sorted(glob(f"./datasets/scene_frames/*/{name}/*.jpg"))
            # print(video_lists)
            # frame_num = random.randint(0, len(video_lists))
            # name = video_lists[frame_num].split("/")[-1].split("_")[0]
            # video_lists = sorted(glob(f"./datasets/scene_frames/*/*/{name}_*.jpg"), key=lambda x : int(x.split("_")[-1].split(".")[0]))
            frame_num = random.randint(0, len(video_lists)-self.num_frames-1)
            # print(len(video_lists), name)
            video_sequence = []
            for f_idx in range(frame_num, frame_num+self.num_frames):
                frame = self.transform(PIL.Image.open(video_lists[f_idx]))
                # frame_name = video_lists[f_idx].split("/")[-1].split("_")[0]
                # print(video_lists[f_idx])
                # if name!=frame_name:
                #     print(name, frame_name)
                video_sequence.append(frame.unsqueeze(0))

            video_sequence = torch.cat(video_sequence)
            return video_sequence, video_lists[0].split("/")[3]
        except:
            return self.__getitem__(idx+1)
    def __len__(self):
        return len(self.video_paths)

class AudioEncoder(torch.nn.Module):
    def __init__(self, backbone_name="resnet18"):
        super(AudioEncoder, self).__init__()
        self.backbone_name = backbone_name
        self.conv = torch.nn.Conv2d(1, 3, (3, 3))
        self.feature_extractor = timm.create_model(self.backbone_name, num_classes=512, pretrained=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.feature_extractor(x)
        return x    

class MakeCutouts(torch.nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)

def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

def prompts_dist_loss(x, targets, loss):
    if len(targets) == 1: # Keeps consitent results vs previous method for single objective guidance 
        return loss(x, targets[0])
    distances = [loss(x, target) for target in targets]
    return torch.stack(distances, dim=-1).sum(dim=-1)  

def norm1(prompt):
    "Normalize to the unit sphere."
    return prompt / prompt.square().sum(dim=-1,keepdim=True).sqrt()

class CLIP(object):
  def __init__(self, device):
    clip_model = "ViT-B/32"
    self.model, _ = clip.load(clip_model)
    self.model = self.model.requires_grad_(False).to(device)
    self.device = device

    self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                          std=[0.26862954, 0.26130258, 0.27577711])

  @torch.no_grad()
  def embed_text(self, prompt):
      "Normalized clip text embedding."
      return norm1(self.model.encode_text(clip.tokenize(prompt).to(self.device)).float())

  def embed_cutout(self, image):
      "Normalized clip image embedding."
      # return norm1(self.model.encode_image(self.normalize(image)))
      return norm1(self.model.encode_image(image))


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    rest_dim = [1] * (input.ndim - bias.ndim - 1)
    input = input.cuda()
    if input.ndim == 3:
        return (
            F.leaky_relu(
                input + bias.view(1, *rest_dim, bias.shape[0]), negative_slope=negative_slope
            )
            * scale
        )
    else:
        return (
            F.leaky_relu(
                input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=negative_slope
            )
            * scale
        )

class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class Mapper(Module):

    def __init__(self, latent_dim=512):
        super(Mapper, self).__init__()

        # self.opts = opts
        layers = [PixelNorm()]

        for i in range(4):
            layers.append(
                EqualLinear(
                    latent_dim, latent_dim, lr_mul=0.01, activation='fused_lrelu'
                )
            )

        self.mapping = nn.Sequential(*layers)


    def forward(self, x):
        x = self.mapping(x)
        return x

class LevelsMapper(Module):

    def __init__(self):
        super(LevelsMapper, self).__init__()

        # self.opts = opts

        # if not opts.no_coarse_mapper:
        self.course_mapping = Mapper()
        # if not opts.no_medium_mapper:
        self.medium_mapping = Mapper()
        # if not opts.no_fine_mapper:
        self.fine_mapping = Mapper()

    def forward(self, x):
        x_coarse = x[:, :4, :]
        x_medium = x[:, 4:8, :]
        x_fine = x[:, 8:, :]

        x_coarse = self.course_mapping(x_coarse)
        x_medium = self.medium_mapping(x_medium)
        x_fine = self.fine_mapping(x_fine)

        out = torch.cat([x_coarse, x_medium, x_fine], dim=1)

        return out

class CLIPLoss(torch.nn.Module):

    def __init__(self, device):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=256 // 32)

    def forward(self, image, text):
        image = self.avg_pool(self.upsample(image))
        similarity = 1 - self.model(image, text)[0] / 100
        return similarity ** 2

class LatentRNN(nn.Module):
    def __init__(self, gpu, type = "coarse", hidden_size=512, num_layers=5, dropout=0.0, bidrectional=True):
        super(LatentRNN, self).__init__()
        self.type = type
        
        self.num_layers = num_layers

        if self.type=="coarse":
            input_size = 4 * 512
        if self.type=="mid":
            input_size = 4 * 512
        if self.type=="fine":
            input_size = 8 * 512
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = getattr(nn, "GRU")(input_size, self.hidden_size, self.num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hidden_size * 2, input_size)

        self.audio_noise_fc = nn.Linear(5120, 5120)   # frame_num * embedding dim
        self.tanh = nn.Tanh()
        self.args_gpu = gpu
        self.weight_noise = 1.0

        
    def forward(self, x, h):

        h = self.audio_noise_fc(h.view(-1, 5120))
        out, h1 = self.rnn(x, h.view(10, -1, self.hidden_size))
        out = x + self.fc(out)
        return out, h1

def subprocess_fn(rank, num_gpus, timestring):
    torch.distributed.init_process_group(backend='nccl',  rank=rank, world_size=num_gpus)

    # Init torch_utils
    torch.cuda.set_device(rank)
    sync_device = torch.device('cuda', rank) if num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)

    make_cutouts = MakeCutouts(224, 32, 0.5)
    dataset = SceneVideoDataset()

    print(len(dataset))

    device = torch.device('cuda', rank)
    print('Using device:', device, file=sys.stderr)

    model_url = "./pretrained_models/lhq-256.pkl"


    with open(model_url, 'rb') as fp:
        G = pickle.load(fp)['G_ema'].to(device)


    sound_inversion = DDP(Encoder().to(device), device_ids=[rank])
    sound_inversion.train()


    def embed_image(image):
        n = image.shape[0]
        cutouts = make_cutouts(image)
        embeds = clip_model.embed_cutout(cutouts)
        embeds = rearrange(embeds, '(cc n) c -> cc n c', n=n)
        return embeds


    # clip_model = CLIP()
    
    print("Model Load!")
    # torch.set_num_threads(8)
    print(torch.get_num_threads())
    # 1 x 512
    # z = torch.randn([5, G.z_dim]).to(device)
    # c = None
    # w = G.mapping(z, c, truncation_psi=0.7)
    # print(w.size())

    zs = torch.randn([10000, G.mapping.z_dim], device=device)
    w_stds = G.mapping(zs, None).std(0)

    num_epochs = 40000
    sequence_length = 5
    batch_size = 1

    iteration = 40000
    cnt_iteration = 0
    train_size = int(0.95 * len(dataset))

    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    traindataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=0)

    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=1,
    )
    # lr = 1e-5
    lr = 0.0001
    lr = 1e-3
    betas=(0.5, 0.999)


    criterion_l2 = nn.MSELoss()
    criterion_video = nn.MSELoss()


    clip_model = CLIP(device)


    G.eval()

    former_coarse = DDP(LatentRNN(device, type="coarse").to(device), device_ids=[rank])
    former_mid = DDP(LatentRNN(device, type="mid").to(device), device_ids=[rank])
    former_fine = DDP(LatentRNN(device, type="fine").to(device), device_ids=[rank])


    discriminator_3d = Discriminator(T=sequence_length).to(device)
    discriminator_2d = Discriminator2D().to(device)

    
    discriminator_2d.train()
    discriminator_3d.train()
    former_coarse.train()
    former_mid.train()
    former_fine.train()
    sound_inversion.train()

    optimizer = optim.AdamW(chain(former_coarse.parameters(), former_fine.parameters(), former_mid.parameters(), sound_inversion.parameters()), lr=lr)
    # optimizer = optim.AdamW(chain(former_coarse.parameters(), former_fine.parameters(), former_mid.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)

    shift = G.synthesis.input.affine(G.mapping.w_avg.unsqueeze(0))
    G.synthesis.input.affine.bias.data.add_(shift.squeeze(0))
    G.synthesis.input.affine.weight.data.zero_()
    
    os.makedirs(f'samples/{timestring}', exist_ok=True)
    loader = iter(traindataloader)
    for epoch in range(num_epochs):

        for batchidx, (batchaudio, batchtext) in enumerate(traindataloader):
            batchrecon, batchrecon_s = [], []
            
            audio_input = torch.cat([batchaudio.unsqueeze(0), batchaudio.unsqueeze(0), batchaudio.unsqueeze(0)], axis=1).to(device)
            reg_loss = 0
            l2_loss = 0
            for i in range(batch_size):
                
                w_s = sound_inversion(audio_input[i].unsqueeze(0))

                targets = [clip_model.embed_text(text) for text in batchtext[i]]
                with torch.no_grad():
                    qs = []
                    losses = []
                    for _ in range(8):
                        q = (G.mapping(torch.randn([4, G.mapping.z_dim], device=device), None, truncation_psi=0.7) - G.mapping.w_avg) / w_stds
                        images = G.synthesis(q * w_stds + G.mapping.w_avg)
                        embeds = embed_image(images.add(1).div(2))
                        loss = prompts_dist_loss(embeds, targets, spherical_dist_loss).mean(0)
                        i = torch.argmin(loss)
                        qs.append(q[i])
                        losses.append(loss[i])
                    qs = torch.stack(qs)
                    losses = torch.stack(losses)
                    i = torch.argmin(losses)
                    i_max = torch.argmax(losses)
                    q = qs[i].unsqueeze(0)
                    q_max = qs[i_max].unsqueeze(0)

                with torch.no_grad():
                    w = (q * w_stds + G.mapping.w_avg).clone().detach()
                
                w0 = w.clone().detach()

                w_list, w_s_list = [], []

                reg_loss += criterion_l2(w_s, w0)

                with torch.no_grad():
                    targets = torch.cat([clip.tokenize(batchtext)]).to(device)

                h_coarse = torch.zeros(1, 5 * 2, 512).to(device)
                h_mid = torch.zeros(1, 5 * 2, 512).to(device)
                h_fine = torch.zeros(1, 5 * 2, 512).to(device)

                h_s_coarse = torch.zeros(1, 5 * 2, 512).to(device)
                h_s_mid = torch.zeros(1, 5 * 2, 512).to(device)
                h_s_fine = torch.zeros(1, 5 * 2, 512).to(device)

                for s in range(sequence_length):

 
                    w_coarse, h_coarse = former_coarse(w.view(-1, 16, 512)[:,:4,:].view(-1, 1, 4 * 512), h_coarse)
                    w_mid, h_mid = former_mid(w.view(-1, 16, 512)[:,4:8,:].view(-1, 1, 4 * 512), h_mid)
                    w_fine, h_fine = former_fine(w.view(-1, 16, 512)[:,8:,:].view(-1, 1, 8 * 512), h_fine)

                    w_s_coarse, h_s_coarse = former_coarse(w_s.view(-1, 16, 512)[:,:4,:].view(-1, 1, 4 * 512), h_s_coarse)
                    w_s_mid, h_s_mid = former_mid(w_s.view(-1, 16, 512)[:,4:8,:].view(-1, 1, 4 * 512), h_s_mid)
                    w_s_fine, h_s_fine = former_fine(w_s.view(-1, 16, 512)[:,8:,:].view(-1, 1, 8 * 512), h_s_fine)
                
                    w_next = torch.cat([w_coarse.view(1, -1, 512), w_mid.view(1, -1, 512), w_fine.view(1, -1, 512)], axis=1).view(-1, 16, 512) # + 0.05 * w_s # - G.mapping.w_avg
                    w_s_next = torch.cat([w_s_coarse.view(1, -1, 512), w_s_mid.view(1, -1, 512), w_s_fine.view(1, -1, 512)], axis=1).view(-1, 16, 512) # + 0.05 * w_s # - G.mapping.w_avg

                    l2_loss += criterion_l2(w_next, w0)

                    w_list.append(w)
                    w_s_list.append(w_s)
                    
                    w = w_next
                    w_s = w_s_next


                w_output = torch.cat(w_list, axis=0)
                w_s_output = torch.cat(w_s_list, axis=0)
                
                generated_video = G.synthesis(w_output.view(-1, 16, 512), noise_mode='const', force_fp32=True).unsqueeze(0)
                with torch.no_grad():
                    sound_image = G.synthesis(w_s_output.view(-1, 16, 512), noise_mode='const', force_fp32=True).unsqueeze(0)

                batchrecon.append(generated_video)
                batchrecon_s.append(sound_image)

            batchrecon = torch.cat(batchrecon, axis=0)
            batchrecon_s = torch.cat(batchrecon_s, axis=0)

            optimizer.zero_grad()
            
            loss = reg_loss + l2_loss + criterion_video(batchrecon_s, batchrecon.clone().detach()).mean() + l2_loss 
            loss.backward()
            
            optimizer.step()

            torch.distributed.barrier()
            
            cnt_iteration += 1
            
            print(f"[epoch : {epoch}] [{batchidx} / {len(traindataloader)}] total_g_loss : {loss.item():.3f} / min : {torch.min(batchrecon): .2f}, max : {torch.max(batchrecon) : .2f}")
            if cnt_iteration % 50 == 0 and rank==0:
                print("Model Save !")
                save_path = "./pretrained_models/coarseformer_.pth"
                torch.save(former_coarse.state_dict(), save_path)
                save_path = "./pretrained_models/midformer_.pth"
                torch.save(former_mid.state_dict(), save_path)
                save_path = "./pretrained_models/fineformer_.pth"
                torch.save(former_fine.state_dict(), save_path)
                save_path = f"./pretrained_models/audio_inversion_.pth"
                torch.save(sound_inversion.state_dict(), save_path)

            if cnt_iteration % 50 == 0 and rank==0:
                save_fake = batchrecon[0]

                save_s = batchrecon_s[0]

                for s in range(sequence_length):


                    pil_image = TF.to_pil_image(save_fake[s].add(1).div(2).clamp(0,1).cpu())
                    pil_image.save(f'samples/{timestring}/fake_000{s}.jpg')

                    pil_image = TF.to_pil_image(save_s[s].add(1).div(2).clamp(0,1).cpu())
                    pil_image.save(f'samples/{timestring}/sound_000{s}.jpg')

        scheduler.step()

if __name__ == '__main__': 
    timestring = "video-generation"
    os.makedirs(f'samples/{timestring}', exist_ok=True)

    random_seed = 32
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    num_gpus=2
    torch.multiprocessing.spawn(fn=subprocess_fn, args=(num_gpus, timestring), nprocs=num_gpus)