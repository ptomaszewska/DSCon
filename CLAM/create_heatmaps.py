from __future__ import print_function

import numpy as np

import argparse

import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
from utils.eval_utils import initiate_model as initiate_model
from utils.core_utils import NAME_ARGS
from models.model_clam import CLAM_MB, CLAM_SB
from models.resnet_custom import resnet50_baseline
from types import SimpleNamespace
from collections import namedtuple
import h5py
import yaml
from wsi_core.batch_process_utils import initialize_df
from vis_utils.heatmap_utils import initialize_wsi, drawHeatmap, compute_from_patches
from wsi_core.wsi_utils import sample_rois
from utils.file_utils import save_hdf5

import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Heatmap inference script')
parser.add_argument('--save_exp_code', type=str, default=None,
					help='experiment code')
parser.add_argument('--overlap', type=float, default=None)
parser.add_argument('--config_file', type=str, default="heatmap_config_template.yaml")
args = parser.parse_args()

def infer_single_slide(model, features, label, reverse_label_dict, k=1):
	features = features.to(device)
	with torch.no_grad():
		if isinstance(model, (CLAM_SB, CLAM_MB)):
			model_results_dict = model(features)
			logits, Y_prob, Y_hat, A, _ = model(features)
			Y_hat = Y_hat.item()

			if isinstance(model, (CLAM_MB,)):
				A = A[Y_hat]

			A = A.view(-1, 1).cpu().numpy()
		else:
			raise NotImplementedError

		print('Y_hat: {}, Y: {}, Y_prob: {}'.format(reverse_label_dict[Y_hat], label, ["{:.4f}".format(p) for p in Y_prob.cpu().flatten()]))
		

	return Y_prob[-1], A


def parse_config_dict(args, config_dict):
	if args.save_exp_code is not None:
		config_dict['exp_arguments']['save_exp_code'] = args.save_exp_code
	if args.overlap is not None:
		config_dict['patching_arguments']['overlap'] = args.overlap
	return config_dict

if __name__ == '__main__':
	config_path = os.path.join('heatmaps/configs', args.config_file)
	config_dict = yaml.safe_load(open(config_path, 'r'))
	config_dict = parse_config_dict(args, config_dict)

	for key, value in config_dict.items():
		if isinstance(value, dict):
			print('\n'+key)
			for value_key, value_value in value.items():
				print (value_key + " : " + str(value_value))
		else:
			print ('\n'+key + " : " + str(value))


	args = config_dict
	patch_args = argparse.Namespace(**args['patching_arguments'])
	data_args = argparse.Namespace(**args['data_arguments'])
	model_args = args['model_arguments']
    
	# Z core_utils NAME_ARGS
	ckpt_args_names = NAME_ARGS[1:]

	if "resnet" in model_args['ckpt_path']:
		model_size="base"
	elif ("mae" in model_args['ckpt_path']) or ("moco" in model_args['ckpt_path']) or ("vit" in model_args['ckpt_path']):
		model_size="small"
	else:
		split_name = model_args['ckpt_path'].split('/')[-2].split('_')
		model_size = split_name[3]

	ENCODING_SIZE_MAPPING={"small":768, "tiny":768,  "base":1024, "large": 1535}
	features_size = ENCODING_SIZE_MAPPING[model_size]        
	split_name=model_args['ckpt_path'].split('/')[-2].split('_')
	split_name = split_name[split_name.index('weighted')+1:]
	ckpt_args_dict = {key:val for key, val in zip(ckpt_args_names, split_name)}
	update_model_args = {'n_classes': args['exp_arguments']['n_classes'], 'dropout_rate':ckpt_args_dict['dropout_rate'], 
        'layer1':ckpt_args_dict['layer1'], 'layer2':ckpt_args_dict['layer2'], 'features_size':features_size}
	model_args.update(update_model_args)
    
	model_args = argparse.Namespace(**model_args)
	exp_args = argparse.Namespace(**args['exp_arguments'])
	heatmap_args = argparse.Namespace(**args['heatmap_arguments'])
	sample_args = argparse.Namespace(**args['sample_arguments'])

	patch_size = tuple([patch_args.patch_size for i in range(2)])
	step_size = tuple((np.array(patch_size) * (1-patch_args.overlap)).astype(int))
	print('patch_size: {} x {}, with {:.2f} overlap, step size is {} x {}'.format(patch_size[0], patch_size[1], patch_args.overlap, step_size[0], step_size[1]))

	
	preset = data_args.preset
	def_seg_params = {'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 'close': 2, 'use_otsu': False, 
					  'keep_ids': 'none', 'exclude_ids':'none'}
	def_filter_params = {'a_t':50.0, 'a_h': 8.0, 'max_n_holes':10}
	def_vis_params = {'vis_level': -1, 'line_thickness': 250}
	def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}
 
	fold=model_args.ckpt_path.split('/')[-1][2]

	if preset is not None:
		preset_df = pd.read_csv(preset)
		for key in def_seg_params.keys():
			def_seg_params[key] = preset_df.loc[0, key]

		for key in def_filter_params.keys():
			def_filter_params[key] = preset_df.loc[0, key]

		for key in def_vis_params.keys():
			def_vis_params[key] = preset_df.loc[0, key]

		for key in def_patch_params.keys():
			def_patch_params[key] = preset_df.loc[0, key]


	if data_args.process_list is None:
		if isinstance(data_args.data_dir, list):
			slides = []
			for data_dir in data_args.data_dir:
				slides.extend(os.listdir(data_dir))
		else:
			slides = sorted(os.listdir(data_args.data_dir))        
		slides = [slide for slide in slides if data_args.slide_ext in slide]
		df = initialize_df(slides, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)
		
	else:
		df = pd.read_csv(os.path.join('heatmaps/process_lists', data_args.process_list))
		df = initialize_df(df, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)

	mask = df['process'] == 1
	process_stack = df[mask].reset_index(drop=True)
	total = len(process_stack)
	print('\nlist of slides to process: ')
	print(process_stack.head(len(process_stack)))

	print('\ninitializing model from checkpoint')
	ckpt_path = model_args.ckpt_path
	print('\nckpt path: {}'.format(ckpt_path))
	
	if model_args.initiate_fn == 'initiate_model':
		model =  initiate_model(model_args, ckpt_path, 'cpu')
	else:
		raise NotImplementedError

	reverse_label_dict={0:0, 1:1}   

	os.makedirs(exp_args.production_save_dir, exist_ok=True)
	os.makedirs(exp_args.raw_save_dir, exist_ok=True)
	blocky_wsi_kwargs = {'top_left': None, 'bot_right': None, 'patch_size': patch_size, 'step_size': patch_size, 
	'custom_downsample':patch_args.custom_downsample, 'level': patch_args.patch_level, 'use_center_shift': heatmap_args.use_center_shift}

	for i in range(len(process_stack)):
		slide_name = process_stack.loc[i, 'slide_id']
		print('\nprocessing: ', slide_name)	

		try:
			label = process_stack.loc[i, 'label']
		except KeyError:
			label = 'Unspecified'

		slide_id = slide_name
   

		if not isinstance(label, str):
			grouping = reverse_label_dict[label]
		else:
			grouping = label

		p_slide_save_dir = os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, str(grouping))
		os.makedirs(p_slide_save_dir, exist_ok=True)


		exp_variant=ckpt_path.split("/")[1]  
		r_slide_save_dir = os.path.join(exp_args.raw_save_dir, exp_variant,str(fold), str(grouping), slide_id) 
		os.makedirs(r_slide_save_dir, exist_ok=True)

		
		print('slide id: ', slide_id)


		if isinstance(data_args.data_dir, str):
			slide_path = os.path.join(data_args.data_dir, slide_name)
		elif isinstance(data_args.data_dir, dict):
			data_dir_key = process_stack.loc[i, data_args.data_dir_key]
			slide_path = os.path.join(data_args.data_dir[data_dir_key], slide_name)
		else:
			raise NotImplementedError


		mask_file = os.path.join(r_slide_save_dir, slide_id.split("/")[-1].split(".")[0]+'_mask.pkl')
		


		block_map_save_path = os.path.join(r_slide_save_dir, '{}_blockmap.h5'.format(slide_id))
   
		print(block_map_save_path, flush=True)
   
		features_path = os.path.join(exp_args.feature_dir,'pt_files', slide_id+'.pt')     
		h5_path = os.path.join(exp_args.feature_dir,'h5_files', slide_id+'.h5')
	

		# load features 
		features = torch.load(features_path)
		process_stack.loc[i, 'bag_size'] = len(features)
		
		Y_probs, A = infer_single_slide(model, features, label, reverse_label_dict, exp_args.n_classes)

		del features

		if True:  
			file = h5py.File(h5_path, "r")
			coords = file['coords'][:]
			file.close()
			asset_dict = {'attention_scores': A, 'coords': coords}
			block_map_save_path = save_hdf5(block_map_save_path, asset_dict, mode='w')
		
		for c in range(exp_args.n_classes):
			process_stack.loc[i, 'p_{}'.format(c)] = Y_probs[c].cpu().numpy()  
      
		os.makedirs('heatmaps/results/{}/{}'.format(model_args.ckpt_path.split('/')[1], str(fold)), exist_ok=True)

		if data_args.process_list is not None:
			process_stack.to_csv('heatmaps/results/{}/{}.csv'.format(model_args.ckpt_path.split('/')[1], str(fold)), index=False) 
		else:
			process_stack.to_csv('heatmaps/results/{}/{}.csv'.format(model_args.ckpt_path.split('/')[1], str(fold)), index=False) 

		file = h5py.File(block_map_save_path, 'r')
		dset = file['attention_scores']
		coord_dset = file['coords']
		scores = dset[:]
		coords = coord_dset[:]
		file.close()

		samples = sample_args.samples
