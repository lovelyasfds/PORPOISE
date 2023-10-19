from __future__ import print_function
import numpy as np
import torch_geometric

import argparse
import pdb
import os
import math
import sys
from timeit import default_timer as timer

import numpy as np
import pandas as pd

### Internal Imports
from datasets.dataset_survival import Generic_WSI_Survival_Dataset, Generic_MIL_Survival_Dataset
from utils.file_utils import save_pkl, load_pkl
from utils.core_utils import train
from utils.utils import get_custom_exp_code

### PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, sampler

# 进行基于交叉验证的模型训练和评估任务
def main(args):
	# 创建结果目录，先检查是否存在结果目录，如果不存在则创建。这个目录用于存储模型训练和评估的结果。
	if not os.path.isdir(args.results_dir):
		os.mkdir(args.results_dir)

	# 设置交叉验证起始和结束折数
	if args.k_start == -1:
		start = 0
	else:
		start = args.k_start
	if args.k_end == -1:
		end = args.k
	else:
		end = args.k_end

	# 初始化变量和循环遍历折数
	latest_val_cindex = [] # latest_val_cindex 用于存储每个折数的验证集 c-index（一个性能指标）
	folds = np.arange(start, end) # 使用 np.arange(start, end) 创建一个表示折数范围的数组，并依次遍历每个折数

	# 开始交叉验证
	for i in folds:
		start = timer()
		seed_torch(args.seed)
		results_pkl_path = os.path.join(args.results_dir, 'split_latest_val_{}_results.pkl'.format(i)) # 构建保存结果的文件路径
		if os.path.isfile(results_pkl_path) and (not args.overwrite): # 如果加过文件已经存在且不需要覆盖，则跳过当前折数处理
			print("Skipping Split %d" % i)
			continue

		# 获取训练集和验证集的数据集
		train_dataset, val_dataset = dataset.return_splits(from_id=False, 
				csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
		train_dataset.set_split_id(split_id=i)
		val_dataset.set_split_id(split_id=i)
		
		# 输出当前折数的训练集和验证集的样本数量，并将其存储在一个元组中
		print('training: {}, validation: {}'.format(len(train_dataset), len(val_dataset)))
		datasets = (train_dataset, val_dataset)
		
		# 根据模型来确定输入数据的维度
		if 'omic' in args.mode or args.mode == 'cluster' or args.mode == 'graph' or args.mode == 'pyramid':
			args.omic_input_dim = train_dataset.genomic_features.shape[1]
			print("Genomic Dimension", args.omic_input_dim)
		elif 'coattn' in args.mode:
			args.omic_sizes = train_dataset.omic_sizes
			print('Genomic Dimensions', args.omic_sizes)
		else:
			args.omic_input_dim = 0

		# 如果任务类型是生存分析，则进行模型训练和验证，并获取每折验证集的C指数
		if args.task_type == 'survival':
			val_latest, cindex_latest = train(datasets, i, args)
			latest_val_cindex.append(cindex_latest)

		# 将当前折数的验证结果保存到文件中，输出当前折数的处理时间
		save_pkl(results_pkl_path, val_latest)
		end = timer()
		print('Fold %d Time: %f seconds' % (i, end - start))

	# 整理交叉验证的结果，根据任务类型和折数情况选择不同的保存文件名，并将结果保存为CSV文件
	if args.task_type == 'survival':
		results_latest_df = pd.DataFrame({'folds': folds, 'val_cindex': latest_val_cindex})

	if len(folds) != args.k:
		save_name = 'summary_partial_{}_{}.csv'.format(start, end)
	else:
		save_name = 'summary.csv'

	results_latest_df.to_csv(os.path.join(args.results_dir, 'summary_latest.csv'))

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description='Configurations for Survival Analysis on TCGA Data.')

# 路径等参数
parser.add_argument('--data_root_dir',   type=str, default='path/to/data_root_dir', help='Data directory to WSI features (extracted via CLAM')
parser.add_argument('--seed', 			 type=int, default=1, help='Random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', 			     type=int, default=5, help='Number of folds (default: 5)')
parser.add_argument('--k_start',		 type=int, default=-1, help='Start fold (Default: -1, last fold)')
parser.add_argument('--k_end',			 type=int, default=-1, help='End fold (Default: -1, first fold)')
parser.add_argument('--results_dir',     type=str, default='./results_new', help='Results directory (Default: ./results)')
# 数据被分为验证和训练的文件夹
parser.add_argument('--which_splits',    type=str, default='1foldcv', help='Which splits folder to use in ./splits/ (Default: ./splits/5foldcv')
# 用于训练的癌症类型
parser.add_argument('--split_dir',       type=str, default='tcga_blca1', help='Which cancer type within ./splits/<which_splits> to use for training. Used synonymously for "task" (Default: tcga_blca_100)')
# 是否使用 TensorBoard 记录数据
parser.add_argument('--log_data',        action='store_true', default=True, help='Log data using tensorboard')
# 是否覆盖已有实验
parser.add_argument('--overwrite',     	 action='store_true', default=False, help='Whether or not to overwrite experiments (if already ran)')

# 模型参数
parser.add_argument('--model_type',      type=str, default='porpoise_mmf', help='Type of model (Default: mcat)')
# 这是指定使用哪些模态数据或在数据加载器中使用哪种合并函数
parser.add_argument('--mode',            type=str, choices=['omic', 'path', 'pathomic', 'pathomic_fast', 'cluster', 'coattn'], default='pathomic', help='Specifies which modalities to use / collate function in dataloader.')
# 数据融合类型
parser.add_argument('--fusion',          type=str, choices=['None', 'concat', 'bilinear'], default='bilinear', help='Type of fusion. (Default: concat).')
# 是否将基因组特征用于签名嵌入
parser.add_argument('--apply_sig',		 action='store_true', default=False, help='Use genomic features as signature embeddings.')
# 是否将基因组特征用作表格特征
parser.add_argument('--apply_sigfeats',  action='store_true', default=False, help='Use genomic features as tabular features.')
# 是否启用dropout
parser.add_argument('--drop_out',        action='store_true', default=True, help='Enable dropout (p=0.25)')
# WSI模型的网络大小
parser.add_argument('--model_size_wsi',  type=str, default='small', help='Network size of AMIL model')
# 基因组模型的网络大小
parser.add_argument('--model_size_omic', type=str, default='small', help='Network size of SNN model')
parser.add_argument('--n_classes', type=int, default=4)


# PORPOISE
# 是否使用突变签名嵌入
parser.add_argument('--apply_mutsig', action='store_true', default=False)
# 是否使用路径模态的门控
parser.add_argument('--gate_path', action='store_true', default=False)
# 是否使用基因组模态的门控
parser.add_argument('--gate_omic', action='store_true', default=False)
# 缩放维度1的值
parser.add_argument('--scale_dim1', type=int, default=8)
# 缩放维度2的值
parser.add_argument('--scale_dim2', type=int, default=8)
parser.add_argument('--skip', action='store_true', default=False)
# 输入dropout概率
parser.add_argument('--dropinput', type=float, default=0.0)
# 路径模态输入的维度
parser.add_argument('--path_input_dim', type=int, default=1024)
# 是否使用多层感知机
parser.add_argument('--use_mlp', action='store_true', default=False)


# 优化器参数+生存损失参数
# 优化器类型
parser.add_argument('--opt',             type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--batch_size',      type=int, default=1, help='Batch Size (Default: 1, due to varying bag sizes)')
# 梯度累积步数
parser.add_argument('--gc',              type=int, default=32, help='Gradient Accumulation Step.')
parser.add_argument('--max_epochs',      type=int, default=20, help='Maximum number of epochs to train (default: 20)')
parser.add_argument('--lr',				 type=float, default=2e-4, help='Learning rate (default: 0.0001)')
# 包级别的损失函数
parser.add_argument('--bag_loss',        type=str, choices=['svm', 'ce', 'ce_surv', 'nll_surv'], default='nll_surv', help='slide-level classification loss function (default: ce)')
# 训练标签的比例
parser.add_argument('--label_frac',      type=float, default=1.0, help='fraction of training labels (default: 1.0)')
#  L2 正则化的权重衰减
parser.add_argument('--reg', 			 type=float, default=1e-5, help='L2-regularization weight decay (default: 1e-5)')
# 未被删失患者的权重
parser.add_argument('--alpha_surv',      type=float, default=0.0, help='How much to weigh uncensored patients')
# 正则化类型
parser.add_argument('--reg_type',        type=str, choices=['None', 'omic', 'pathomic'], default='pathomic', help='Which network submodules to apply L1-Regularization (default: None)')
# L1 正则化强度
parser.add_argument('--lambda_reg',      type=float, default=1e-5, help='L1-Regularization Strength (Default 1e-4)')
# 是否启用加权采样
parser.add_argument('--weighted_sample', action='store_true', default=True, help='Enable weighted sampling')
# 是否早停
parser.add_argument('--early_stopping',  action='store_true', default=False, help='Enable early stopping')

# CLAM-Specific参数
# 包级别损失的权重系数
parser.add_argument('--bag_weight',      type=float, default=0.7, help='clam: weight coefficient for bag-level loss (default: 0.7)')
# 是否启用测试模式，用作调试工具
parser.add_argument('--testing', 	 	 action='store_true', default=False, help='debugging tool')

args = parser.parse_args()
# 判断是用CPU还是GPU运行
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 生成一个用于区分不同实验的代码
args = get_custom_exp_code(args)
# 根据文件夹名称创建任务名
args.task = '_'.join(args.split_dir.split('_')[:2]) + '_survival'
# 打印代码
print("Experiment Name:", args.exp_code)

# 设置随机种子，使实验结果可重复
def seed_torch(seed=7):
	import random
	# 这些代码用于设置不同库的随机数种子，包括random、os.environ 环境变量、NumPy 和 PyTorch，确保代码的不同部分收到相同的随机种子形象
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if device.type == 'cuda':
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 1024
settings = {'num_splits': args.k, 
			'k_start': args.k_start,
			'k_end': args.k_end,
			'task': args.task,
			'max_epochs': args.max_epochs, 
			'results_dir': args.results_dir, 
			'lr': args.lr,
			'experiment': args.exp_code,
			'reg': args.reg,
			'label_frac': args.label_frac,
			'bag_loss': args.bag_loss,
			#'bag_weight': args.bag_weight,
			'seed': args.seed,
			'model_type': args.model_type,
			'model_size_wsi': args.model_size_wsi,
			'model_size_omic': args.model_size_omic,
			"use_drop_out": args.drop_out,
			'weighted_sample': args.weighted_sample,
			'gc': args.gc,
			'opt': args.opt}
print('\nLoad Dataset')

# 检查任务类型是否包含关键字survival，如果有表示当前任务是一个生存分析任务，如果没有，则抛出NotImplementedError，表示当前任务类型尚未实现
if 'survival' in args.task:
	# 将原始的研究领域名称映射到更一般的合并研究领域，以便在后续的数据集构建中使用合适的研究领域目录
	study = '_'.join(args.task.split('_')[:2])
	# 判断是不是肾脏领域
	if study == 'tcga_kirc' or study == 'tcga_kirp':
		combined_study = 'tcga_kidney'
	# 判断是不是肺部领域
	elif study == 'tcga_luad' or study == 'tcga_lusc':
		combined_study = 'tcga_lung'
	else:
		combined_study = study
	# 根据合并的研究领域名称创建一个目录路径
	study_dir = '%s_20x_features' % combined_study

	dataset = Generic_MIL_Survival_Dataset(csv_path = './%s/%s_all_clean.csv.zip' % (args.dataset_path, study),
										   mode = args.mode,
										   apply_sig = args.apply_sig,
										   data_dir= os.path.join(args.data_root_dir, study_dir),
										   shuffle = False, 
										   seed = args.seed, 
										   print_info = True,
										   patient_strat= False,
										   n_bins=2,
										   label_col = 'survival_months',
										   ignore=[])
else:
	raise NotImplementedError

# 根据数据集的类型判断任务类型，如果数据集属于生存分析的任务类型，将args.task_type设置为'survival'，表示任务类型是生存分析
if isinstance(dataset, Generic_MIL_Survival_Dataset):
	args.task_type = 'survival'
else:
	raise NotImplementedError

# 检查结果目录 args.results_dir 是否存在，如果不存在则创建该目录
if not os.path.isdir(args.results_dir):
	os.mkdir(args.results_dir)

# 将折数，参数代码，用于区分不同实验的代码附加到结果目录的路径中，并确保最终目录存在
args.results_dir = os.path.join(args.results_dir, args.which_splits, args.param_code, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
	os.makedirs(args.results_dir)

# 检查是否存在具有相同的 用于区分不同实验的代码  的实验结果文件（就是看这个实验有没有做过），并根据是否存在summary_latest.csv判断是否终止实验
if ('summary_latest.csv' in os.listdir(args.results_dir)) and (not args.overwrite):
	print("Exp Code <%s> already exists! Exiting script." % args.exp_code)
	sys.exit()

# 设置args.split_dir的绝对路径，并更新settings中的相应路径
args.split_dir = os.path.join('./splits', args.which_splits, args.split_dir)
print("split_dir", args.split_dir)
assert os.path.isdir(args.split_dir)
settings.update({'split_dir': args.split_dir})

# # 将设置信息写入一个文本文件，以便记录实验的配置和参数
# with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
# 	print(settings, file=f)
# f.close()

# 打印输出实验的设置和参数，以便在运行过程中查看实验的配置
print("################# Settings ###################")
for key, val in settings.items():
	print("{}:  {}".format(key, val))        

if __name__ == "__main__":
	start = timer()
	results = main(args)
	end = timer()
	print("finished!")
	print("end script")
	print('Script Time: %f seconds' % (end - start))
