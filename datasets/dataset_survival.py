from __future__ import print_function, division
import math
import os
import pdb
import pickle
import re

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset

from utils.utils import generate_split, nth


class Generic_WSI_Survival_Dataset(Dataset):
    def __init__(self,
        csv_path = 'dataset_csv/tcga_blca1_all_clean.csv.zip', mode = 'omic', apply_sig = False,
        shuffle = False, seed = 7, print_info = True, n_bins = 4, ignore=[],
        patient_strat=False, label_col = None, filter_dict = {}, eps=1e-6):
        r"""
        Generic_WSI_Survival_Dataset 

        Args:
            csv_file (string): Path to the csv file with annotations.
            shuffle (boolean): Whether to shuffle 是否对数据随机打乱
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int 用于将字符串标签转换为整数标签的键值对
            ignore (list): List containing class labels to ignore
        """
        # 表示在初始化时还没有自定义的测试集标识
        self.custom_test_ids = None
        self.seed = seed
        self.print_info = print_info
        # 控制是否进行基于患者的分层采样
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
        self.data_dir = None

        # 读取 CSV 将其存储为 DataFrame
        slide_data = pd.read_csv(csv_path, low_memory=False)
        # 删除CSV文件中第一列
        #slide_data = slide_data.drop(['Unnamed: 0'], axis=1)

        # 如果shuffle参数为True，则对slide_data进行随机打乱操作
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)

        # 如果 DataFrame 中不存在名为 'case_id' 的列，就根据索引创建一个 'case_id' 列，并重置 DataFrame 的索引
        if 'case_id' not in slide_data:
            slide_data.index = slide_data.index.str[:12]
            slide_data['case_id'] = slide_data.index
            slide_data = slide_data.reset_index(drop=True)

        # 如果没有指定标签列，则默认使用生存时间作为标签列。如果指定了标签列，会检查它是否存在于 DataFrame 的列中，并将其设置为类的实例属性
        if not label_col:
            label_col = 'survival_months'
        else:
            assert label_col in slide_data.columns
        self.label_col = label_col

        # 如果 'oncotree_code' 包含值 "IDC"，则只保留这个类别的数据行
        if "IDC" in slide_data['oncotree_code']: # must be BRCA (and if so, use only IDCs)
            slide_data = slide_data[slide_data['oncotree_code'] == 'IDC']

        # 去除重复的病人信息，筛选出未删失的病人
        patients_df = slide_data.drop_duplicates(['case_id']).copy()
        uncensored_df = patients_df[patients_df['censorship'] < 1]

        # 将数值型标签列进行分位数划分，并计算分位数的边界
        disc_labels, q_bins = pd.qcut(uncensored_df[label_col], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = slide_data[label_col].max() + eps
        q_bins[0] = slide_data[label_col].min() - eps

        # 这段代码的作用是将病人的数值型标签列按照之前计算的分位数边界进行划分，并将划分结果作为离散标签插入到病人信息 DataFrame 中
        disc_labels, q_bins = pd.cut(patients_df[label_col], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        patients_df.insert(2, 'label', disc_labels.values.astype(int))

        # 创建一个病人信息字典，并将 slide_data 的索引列由生存时间改为 case_id
        patient_dict = {}
        slide_data = slide_data.set_index('case_id')

        # 把每个 slide_ids 都添加到 patient_dict 中
        for patient in patients_df['case_id']:
            slide_ids = slide_data.loc[patient, 'slide_id']
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
            else:
                slide_ids = slide_ids.values
            patient_dict.update({patient:slide_ids})

        self.patient_dict = patient_dict

        # 将处理后的数据定义为 slide_data ，删除原来索引，把 case_id 列的数据复制到 slide_id 列上
        slide_data = patients_df
        slide_data.reset_index(drop=True, inplace=True)
        slide_data = slide_data.assign(slide_id=slide_data['case_id'])

        # 对每一个分组的label和是否审查进行一个特定数字的编码，比如label0和未审查0对应0，label0和未审查0对应1.(0,0):0,(0,1):1.
        label_dict = {}
        key_count = 0
        for i in range(len(q_bins)-1):
            for c in [0, 1]:
                print('{} : {}'.format((i, c), key_count))
                label_dict.update({(i, c):key_count})
                key_count+=1

        # 重新给定label值，就是上一段的编码值变成现在的label值
        self.label_dict = label_dict
        for i in slide_data.index:
            key = slide_data.loc[i, 'label']
            slide_data.at[i, 'disc_label'] = key
            censorship = slide_data.loc[i, 'censorship']
            key = (key, int(censorship))
            slide_data.at[i, 'label'] = label_dict[key]

        # 将标签分割点、分类类别数、病人数据等信息进行了处理和准备
        self.bins = q_bins
        self.num_classes=len(self.label_dict)
        # 删除重复的病人信息
        patients_df = slide_data.drop_duplicates(['case_id'])
        self.patient_data = {'case_id':patients_df['case_id'].values, 'label':patients_df['label'].values}

        #new_cols = list(slide_data.columns[-2:]) + list(slide_data.columns[:-2]) ### ICCV
        # 对幻灯片数据集的列进行了重新排序，把原来的分组标签 disc_label 放到了第一列
        new_cols = list(slide_data.columns[-1:]) + list(slide_data.columns[:-1])  ### PORPOISE
        slide_data = slide_data[new_cols]
        self.slide_data = slide_data
        metadata = ['disc_label', 'case_id', 'slide_id', 'label', 'site', 'is_female', 'oncotree_code', 'age', 'survival_months', 'censorship', 'train']
        # metadata = ['disc_label', 'Unnamed: 0', 'case_id', 'label', 'slide_id', 'age', 'site', 'survival_months', 'censorship', 'is_female', 'oncotree_code', 'train']
        self.metadata = slide_data.columns[:11]

        # 查找并输出那些不包含指定字符串的基因名
        for col in slide_data.drop(self.metadata, axis=1).columns:
            if not pd.Series(col).str.contains('|_cnv|_rnaseq|_rna|_mut')[0]:
                print(col)
        #pdb.set_trace()

        # 进行元数据的断言检查
        assert self.metadata.equals(pd.Index(metadata))
        self.mode = mode
        self.cls_ids_prep()

        ### ICCV discrepancies
        # For BLCA, TPTEP1_rnaseq was accidentally appended to the metadata
        #pdb.set_trace()

        # print_info 参数为 True，打印输出信息
        if print_info:
            self.summarize()

        #  根据需要，选择是否使用基因组特征签名
        self.apply_sig = apply_sig
        if self.apply_sig:
            self.signatures = pd.read_csv('./datasets_csv_sig/signatures.csv')
        else:
            self.signatures = None

        if print_info:
            self.summarize()


    def cls_ids_prep(self):
        r"""

        """
        # 创建了个大列表里面包含每个组对应患者的索引，patient_data只包含患者和对应的索引，slide_data是整个数据
        self.patient_cls_ids = [[] for i in range(self.num_classes)]        
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

        #
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]


    def patient_data_prep(self):
        r"""
        
        """
        # 获取唯一的患者标识（case_id）
        patients = np.unique(np.array(self.slide_data['case_id'])) # get unique patients
        patient_labels = []

        # 为每个患者找到与其关联的所有幻灯片的位置（索引），"patient_labels" 列表将包含所有患者的标签
        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = self.slide_data['label'][locations[0]] # get patient label
            patient_labels.append(label)

        # "patient_data" 字典就将每个患者的标识与其对应的标签关联起来
        self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}


    @staticmethod
    def df_prep(data, n_bins, ignore, label_col):
        r"""
        
        """
        # 从数据中移除不需要的标签，并对剩余的数据进行分箱处理。最终返回处理后的数据和分箱信息
        mask = data[label_col].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        disc_labels, bins = pd.cut(data[label_col], bins=n_bins)
        return data, bins

    def __len__(self):
        # 在训练和验证循环中确定迭代的次数。如果采用患者层次的策略，每个患者被视为一个数据点；如果采用切片层次的策略，每个切片被视为一个数据点
        if self.patient_strat:
            return len(self.patient_data['case_id'])
        else:
            return len(self.slide_data)

    # 打印一些关键信息
    def summarize(self):
        print("label column: {}".format(self.label_col))
        print("label dictionary: {}".format(self.label_dict))
        print("number of classes: {}".format(self.num_classes))
        print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
        for i in range(self.num_classes):
            print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
            print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))


    def get_split_from_df(self, all_splits: dict, split_key: str='train', scaler=None):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split(df_slice, metadata=self.metadata, mode=self.mode, signatures=self.signatures, data_dir=self.data_dir, label_col=self.label_col, patient_dict=self.patient_dict, num_classes=self.num_classes)
        else:
            split = None
        
        return split


    def return_splits(self, from_id: bool=True, csv_path: str=None):
        if from_id:
            raise NotImplementedError
        else:
            assert csv_path 
            all_splits = pd.read_csv(csv_path)
            train_split = self.get_split_from_df(all_splits=all_splits, split_key='train')
            val_split = self.get_split_from_df(all_splits=all_splits, split_key='val')
            test_split = None #self.get_split_from_df(all_splits=all_splits, split_key='test')

            ### --> Normalizing Data
            print("****** Normalizing Data ******")
            scalers = train_split.get_scaler()
            train_split.apply_scaler(scalers=scalers)
            val_split.apply_scaler(scalers=scalers)
            #test_split.apply_scaler(scalers=scalers)
            ### <--
        return train_split, val_split#, test_split


    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def __getitem__(self, idx):
        return None

    def __getitem__(self, idx):
        return None


class Generic_MIL_Survival_Dataset(Generic_WSI_Survival_Dataset):
    def __init__(self, data_dir, mode: str='omic', **kwargs):
        # 这是调用父类 Generic_WSI_Survival_Dataset 的构造函数的方式。它使用 super() 函数来调用父类的构造函数，从而初始化继承自父类的属性和方法。
        # 通过 **kwargs，可以将所有关键字参数传递给父类的构造函数，以确保父类的属性正确初始化。
        super(Generic_MIL_Survival_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.mode = mode
        self.use_h5 = False

    # 控制是否从 HDF5 数据格式加载数据
    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def __getitem__(self, idx):
        case_id = self.slide_data['case_id'][idx]
        label = torch.Tensor([self.slide_data['disc_label'][idx]])
        event_time = torch.Tensor([self.slide_data[self.label_col][idx]])
        c = torch.Tensor([self.slide_data['censorship'][idx]])
        slide_ids = self.patient_dict[case_id]

        if type(self.data_dir) == dict:
            source = self.slide_data['oncotree_code'][idx]
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir
        
        if not self.use_h5:
            if self.data_dir:
                if self.mode == 'path':
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
                    return (path_features, torch.zeros((1,1)), label, event_time, c)

                elif self.mode == 'cluster':
                    path_features = []
                    cluster_ids = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                        cluster_ids.extend(self.fname2ids[slide_id[:-4]+'.pt'])
                    path_features = torch.cat(path_features, dim=0)
                    cluster_ids = torch.Tensor(cluster_ids)
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    return (path_features, cluster_ids, genomic_features, label, event_time, c)

                elif self.mode == 'omic':
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    return (torch.zeros((1,1)), genomic_features.unsqueeze(dim=0), label, event_time, c)

                elif self.mode == 'pathomic':
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    return (path_features, genomic_features.unsqueeze(dim=0), label, event_time, c)

                elif self.mode == 'pathomic_fast':
                    casefeat_path = os.path.join(data_dir, f'split_{self.split_id}_case_pt', f'{case_id}.pt')
                    path_features = torch.load(casefeat_path)
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    return (path_features, genomic_features.unsqueeze(dim=0), label, event_time, c)

                elif self.mode == 'coattn':
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
                    omic1 = torch.tensor(self.genomic_features[self.omic_names[0]].iloc[idx])
                    omic2 = torch.tensor(self.genomic_features[self.omic_names[1]].iloc[idx])
                    omic3 = torch.tensor(self.genomic_features[self.omic_names[2]].iloc[idx])
                    omic4 = torch.tensor(self.genomic_features[self.omic_names[3]].iloc[idx])
                    omic5 = torch.tensor(self.genomic_features[self.omic_names[4]].iloc[idx])
                    omic6 = torch.tensor(self.genomic_features[self.omic_names[5]].iloc[idx])
                    return (path_features, omic1, omic2, omic3, omic4, omic5, omic6, label, event_time, c)

                else:
                    raise NotImplementedError('Mode [%s] not implemented.' % self.mode)
            else:
                return slide_ids, label, event_time, c


class Generic_Split(Generic_MIL_Survival_Dataset):
    def __init__(self, slide_data, metadata, mode, 
        signatures=None, data_dir=None, label_col=None, patient_dict=None, num_classes=2):
        self.use_h5 = False
        self.slide_data = slide_data
        self.metadata = metadata
        self.mode = mode
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.label_col = label_col
        self.patient_dict = patient_dict
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

        ### --> Initializing genomic features in Generic Split
        self.genomic_features = self.slide_data.drop(self.metadata, axis=1)
        self.signatures = signatures

        if mode == 'cluster':
            with open(os.path.join(data_dir, 'fast_cluster_ids.pkl'), 'rb') as handle:
                self.fname2ids = pickle.load(handle)

        def series_intersection(s1, s2):
            return pd.Series(list(set(s1) & set(s2)))

        if self.signatures is not None:
            self.omic_names = []
            for col in self.signatures.columns:
                omic = self.signatures[col].dropna().unique()
                omic = np.concatenate([omic+mode for mode in ['_mut', '_cnv', '_rnaseq']])
                omic = sorted(series_intersection(omic, self.genomic_features.columns))
                self.omic_names.append(omic)
            self.omic_sizes = [len(omic) for omic in self.omic_names]
        print("Shape", self.genomic_features.shape)
        ### <--

    def __len__(self):
        return len(self.slide_data)

    ### --> Getting StandardScaler of self.genomic_features
    def get_scaler(self):
        scaler_omic = StandardScaler().fit(self.genomic_features)
        return (scaler_omic,)
    ### <--

    ### --> Applying StandardScaler to self.genomic_features
    def apply_scaler(self, scalers: tuple=None):
        transformed = pd.DataFrame(scalers[0].transform(self.genomic_features))
        transformed.columns = self.genomic_features.columns
        self.genomic_features = transformed
    ### <--

    def set_split_id(self, split_id):
        self.split_id = split_id