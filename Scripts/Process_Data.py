# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 16:17:43 2023

@author: orifl
"""
from Util_Functions import create_dir,string_to_bool
from Data_Functions import get_FPKM, get_CHR_idx, get_RNA_annotation
import pandas as pd

import argparse
parser = argparse.ArgumentParser('CellRBP')

parser.add_argument('--input_data_path',    type=str,   default='',                                 help='Input data path. Example: ../Data/clip_data/AARS_K562.tsv')
parser.add_argument('--output_data_dir',    type=str,   default="../Data/clip_data_processed",      help='Output data path. Default: ../Data/clip_data_processed')
parser.add_argument('--is_prismnet',        type=str,   default='True',                             help='Is the input data in PrismNet format? [True,False]')
parser.add_argument('--is_annotated',       type=str,   default='True',                             help='Is the input data in PrismNet format? [True,False]')

parser.add_argument('--train_frac',         type=float, default="0.7",                              help='Train set size [0-1]')
parser.add_argument('--valid_frac',         type=float, default="0.15",                             help='Validation set size [0-1]')
parser.add_argument('--test_frac',          type=float, default="0.15",                             help='Test set size [0-1]')

args = parser.parse_args()

input_data_path = args.input_data_path
output_data_dir = args.output_data_dir
is_prismnet = string_to_bool(args.is_prismnet)
is_annotated = string_to_bool(args.is_annotated)

train_frac = args.train_frac
valid_frac = args.valid_frac
test_frac = args.test_frac

#%%

# --------------------- Read data and create dataframe --------------------- 

def create_df (data_path, is_prismnet):
    if is_prismnet:
        names = ['Unk1','ENST_Indices','Sequence','SHAPE','Unk2']
        if is_annotated:
            names.append('eCLIP_Score')
        df_eclip = pd.read_csv(data_path,delimiter='\t',names=names)
        df_eclip = df_eclip.drop(columns =['Unk1','Unk2'])

    else:
        #Check if the structure of the data is correct
        df = pd.read_csv(data_path,delimiter='\t')
        if df.shape[1] != 3:
            print('Error: The eCLIP data file contains {} columns.\nThe eCLIP data file should contain 3 tab limited columns: ENST indices (ENST|start|stop),Sequence,icSHAPE')
        else:
            names = ['ENST_Indices','Sequence','SHAPE']
            if is_annotated:
                names.append('eCLIP_Score')
            df_eclip = pd.read_csv(data_path,delimiter='\t',names=names)

    return df_eclip

#%%
# --------------------- ADD FPKM --------------------- 

def add_FPKM (df_eclip, Cell_Type, FPKM_path):

    print('------ Retreiving FPKM data to {} {} ------'.format(Protein,Cell_Type))
    df_eclip_fpkm = get_FPKM(df_eclip,FPKM_path)
    
    return df_eclip_fpkm

#%%
# --------------------- ADD CHR info --------------------- 

def add_chr_pos (df_eclip):
    transcripts_Path = '../Features/df_transcripts.pkl'
    print('------ Retreiving genome positions for {} {} ------'.format(Protein,Cell_Type))
    df_eclip_Chr = get_CHR_idx (df_eclip,transcripts_Path)
    
    return df_eclip_Chr 


#%% 
# --------------------- ADD RNA Annotation info --------------------- 

def add_RNA_annotations (df_eclip_Chr,output_data_dir,Protein,Cell_Type,gtf_Path):
    # gtf_Path = '../Features/GTF/Homo_sapiens.GRCh38.89.gtf'
    print('------ Retreiving RNA annotations info for {} {} ------'.format(Protein,Cell_Type))
    df_eclip_fpkm_Chr_RNA_annotations = get_RNA_annotation (df_eclip_Chr,gtf_Path)
    
    path_csv = output_data_dir +'/{}_{}'.format(Protein,Cell_Type)
    create_dir(path_csv)
    
    df_eclip_fpkm_Chr_RNA_annotations.to_csv(path_csv+'/all.tsv',sep='\t',index=False)
    return df_eclip_fpkm_Chr_RNA_annotations

#%%
# --------------------- Filter, shuffle, split ---------------------

def filt_shuffle_split (df_eclip,letters_pool,train_frac,valid_frac,test_frac,Protein,Cell_Type,output_data_dir):

    print('------ Filtering, shuffling and splitting {} {} ------'.format(Protein,Cell_Type))
    
    #Delete rows containing illegal strings ['N','W','Y','R','M','K','Seq']
    raw_len = len(df_eclip)
    for letter in letters_pool:
        df_eclip = df_eclip[~df_eclip['Sequence'].str.contains(letter)]
    print('{} {}: {} Row(s) were deleted'.format(Protein,Cell_Type,raw_len-len(df_eclip)))
    df_eclip = df_eclip.reset_index(drop=True)
    
    #Divide into Training, Validation and Test
    df_eclip_shuff = df_eclip.sample(frac=1, random_state=42)
            
    #Train [70%]
    train_Samples = round(len(df_eclip_shuff)*train_frac)
    df_eclip_shuff_train = df_eclip_shuff[:train_Samples]
    
    #Validation [15%]
    validation_Samples = round(len(df_eclip_shuff)*valid_frac)
    df_eclip_shuff_valid = df_eclip_shuff[train_Samples:train_Samples+validation_Samples]
    
    #Test [15%]
    df_eclip_shuff_test = df_eclip_shuff[train_Samples+validation_Samples:]
    
    #Save Training, Validation and both Test data sets
    path_csv = output_data_dir + '/{}_{}'.format(Protein,Cell_Type)
    df_eclip_shuff_train.to_csv(path_csv+'/Train_{}.tsv'.format(int(train_frac*100)),sep='\t',header=True,index=False)
    df_eclip_shuff_valid.to_csv(path_csv+'/Validation_{}.tsv'.format(int(valid_frac*100)),sep='\t',header=True,index=False)
    df_eclip_shuff_test.to_csv(path_csv+'/Test_{}.tsv'.format(int(test_frac*100)),sep='\t',header=True,index=False)
    
    print('------ Data sets for {} {} were successfully created ------'.format(Protein,Cell_Type))
    
#%%
# --------------------- Functions caller ---------------------
if input_data_path == '':
    print('\nError: No input data path was inserted. Example: --input_data_path ../Data/clip_data/AARS_K562.tsv')
else:
    Protein = input_data_path.split('/')[-1].split('.')[0].split('_')[0]
    Cell_Type = input_data_path.split('/')[-1].split('.')[0].split('_')[1]

    print('------ Processing data details ------\nInput data path: \t\t{}\nIs in PrismNet format: \t\t{}\nIs annotated: \t\t\t{}\nOutput data dir: \t\t{} \
          \nProtein: \t\t\t{}\nCell type: \t\t\t{}\nTrain set fraction size: \t{}\nValidation set fraction size: \t{}'\
          '\nTest set fraction size: \t{}\n'.format(input_data_path,is_prismnet,is_annotated,output_data_dir,Protein,Cell_Type,train_frac,valid_frac,test_frac)) 

    df_eclip = create_df (input_data_path, is_prismnet)
    
    if Cell_Type == 'K562':
        FPKM_path = '../Features/FPKM/K562/ENCFF429YLC.tsv'
    if Cell_Type == 'HepG2':
        FPKM_path = '../Features/FPKM/HepG2/ENCFF878MHG.tsv'
    df_eclip_fpkm = add_FPKM (df_eclip, Cell_Type, FPKM_path)
    df_eclip_fpkm_chr = add_chr_pos (df_eclip_fpkm)
    
    gtf_Path = '../Features/GTF/Homo_sapiens.GRCh38.89.gtf'
    df_eclip_fpkm_chr_RNA_annotations = add_RNA_annotations (df_eclip_fpkm_chr,output_data_dir,Protein,Cell_Type,gtf_Path)
    
    letters_pool_to_filt = ['N','W','Y','R','M','K','Seq']
    filt_shuffle_split (df_eclip_fpkm_chr_RNA_annotations,letters_pool_to_filt,train_frac,valid_frac,test_frac,Protein,Cell_Type,output_data_dir)






    
