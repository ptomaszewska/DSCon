import torch
import torch.nn as nn
from math import floor
import os
import random
import numpy as np
import pdb
import time
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
from models.resnet_custom import resnet50_baseline
import argparse
from utils.utils import print_network, collate_features
from utils.file_utils import save_hdf5
from PIL import Image
import h5py
import openslide
import torchvision
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

import timm
import math
from functools import partial, reduce
from operator import mul

from timm.models.layers.helpers import to_2tuple
from timm.models.layers import PatchEmbed

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
random.seed(1)


# the parts of code regarding transformer-based models are inspired from https://github.com/KMnP/vpt


def Swin_choice(model_type):
	model_components=model_type.split('_')
	if "v2" in model_components[0]:
		from models.swin_transformerv2 import SwinTransformerV2 as SwinTransformer
	else:
		from models.swin_transformer import SwinTransformer 

	if "22k" in model_type:
		model_type=model_type.replace("_22k","")

	model_components=model_type.split('_')
	model_size=model_components[1]
	window_size=int(model_components[3].replace("window",""))
	img_size=int(model_components[-1])
	if model_size == "tiny":  
		model = SwinTransformer(
			img_size=(img_size, img_size),
			embed_dim=96,
			depths=[2, 2, 6, 2],
			num_heads=[3, 6, 12, 24],
			window_size=window_size,
			drop_path_rate=0.2,
			num_classes=-1,  # setting to a negative value will make head as identity
		)
		embed_dim = 96
		num_layers = 4

	elif model_size == "small":
		model = SwinTransformer(
			img_size=(img_size, img_size),
			embed_dim=96,
			depths=[2, 2, 18, 2],
			num_heads=[3, 6, 12, 24],
			window_size=window_size,
			drop_path_rate=0.3,
			num_classes=-1,
		)
		embed_dim = 96
		num_layers = 4
	elif model_size == "base":
		model = SwinTransformer(
			img_size=(img_size, img_size),
			embed_dim=128,
			depths=[2, 2, 18, 2],
			num_heads=[4, 8, 16, 32],
			window_size=window_size,
			drop_path_rate=0.5,
			num_classes=-1,
		)
		embed_dim = 128
		num_layers = 4
        
	feat_dim = int(embed_dim * 2 ** (num_layers - 1))

	return model, feat_dim

def load_ViT(args):   
	if args.model_type =='swin':
		model,feat_dim=Swin_choice(args.swin_model_name)	
		
		ckpt="./pretrained_models/"+args.swin_model_name+".pth" 
		checkpoint = torch.load(ckpt, map_location=device)
		if 'pcam' in args.swin_model_name:
			state_dict=checkpoint
			for k in list(state_dict.keys()):
				if k.startswith('enc.'):
					state_dict[k[len("enc."):]] = state_dict[k] 
				del state_dict[k]
		else:
			state_dict = checkpoint['model']
		
		if args.swin_model_name.endswith("ssl"):
			# rename moco pre-trained keys
			for k in list(state_dict.keys()):
				# retain only encoder_q up to before the embedding layer
				if k.startswith('encoder.'):
					# remove prefix
					state_dict[k[len("encoder."):]] = state_dict[k]
				# delete renamed or unused k
				del state_dict[k]
		 
		model.load_state_dict(state_dict, strict=False)

	elif args.model_type=='vit':
		from models.vit import VisionTransformer
		model= VisionTransformer(
			  "sup_vitb16_imagenet21k", 224, num_classes=-1, vis=False)
		model.load_from(np.load("./pretrained_models/ViT-B_16.npz"))

	model.to(device)
	return model


def compute_w_loader(file_path, output_path, wsi, model,
 	batch_size = 8, verbose = 0, print_every=20, pretrained=True, 
	custom_downsample=1, target_patch_size=-1, gray=False):
	"""
	args:
		file_path: directory of bag (.h5 file)
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		batch_size: batch_size for computing features in batches
		verbose: level of feedback
		pretrained: use weights pretrained on imagenet
		custom_downsample: custom defined downscale factor of image patches
		target_patch_size: custom defined, rescaled image size before embedding
   gray: if the input images should be transformed to grayscale
	"""
	model.eval()
	dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, 
		custom_downsample=custom_downsample, target_patch_size=target_patch_size, gray=gray)
	x, y = dataset[0]
	kwargs = {'num_workers': 1, 'pin_memory': True} if device.type == "cuda" else {} 
	loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'
	
	for count, (batch, coords) in enumerate(loader): 
		if count % print_every == 0:
			print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
		batch = batch.to(device, non_blocking=True)
		features = model(batch)
		features = features.detach().cpu().numpy()
		asset_dict = {'features': features, 'coords': coords}
		save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
		mode = 'a'
	
	return output_path


def none_or_str(value):
	if value == 'None':
		return None
	return value

parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--model_type', type=str, default=None)
parser.add_argument('--gray', default="False", help='grayscaled images') #action='store_true'
parser.add_argument('--swin_model_name', type=none_or_str, default=None, help='Name of pretrained swin')

args = parser.parse_args()



if __name__ == '__main__':

	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError

	bags_dataset = Dataset_All_Bags(csv_path)
	
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

	print('loading model checkpoint', args.feat_dir.split("/")[1].split("_")[-1])
	if "resnet" not in args.model_type:
		model = load_ViT(args) 
	elif "resnetCLAM" == args.model_type: 
		model=resnet50_baseline(pretrained=True) 

	model = model.to(device)
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		
	model.eval()
	total = len(bags_dataset)
	in_grayscale=args.gray

		
	for bag_candidate_idx in range(total):
		slide_id = bags_dataset[bag_candidate_idx].split(".")[0]
		bag_name = slide_id+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
		slide_file_path = os.path.join(args.data_slide_dir, bags_dataset[bag_candidate_idx])
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)

		if not args.no_auto_skip and slide_id+'.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue 

		 
		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		time_start = time.time()
		wsi = openslide.open_slide(slide_file_path)
		output_file_path = compute_w_loader(h5_file_path, output_path, wsi, 
			model = model, batch_size = args.batch_size, verbose = 1, print_every = 20, 
			custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size, gray=in_grayscale)
		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
		file = h5py.File(output_file_path, "r")

		features = file['features'][:]
		print('features size: ', features.shape)
		print('coordinates size: ', file['coords'].shape)
		features = torch.from_numpy(features)
		bag_base, _ = os.path.splitext(bag_name)
		torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))