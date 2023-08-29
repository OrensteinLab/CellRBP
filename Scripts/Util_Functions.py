# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 10:40:02 2021

@author: Ori Feldman
"""
#%%
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing

import matplotlib.pyplot as plt
import logomaker
import os
import time
import sys


length_of_seq=101
Length_of_nbs = 20


#%% General functions

def create_dir(path, silence=False):
    isExist = os.path.exists(path)
    if not isExist:
       os.makedirs(path)
       if silence == False:
           print("New Directory Created in {}".format(path)) 

def check_path(path):
    isExist = os.path.exists(path)
    if not isExist:
        print("\nError: path doesn\'t exist: {}.\nExitting".format(path)) 
        sys.exit()
        
def fix_path(path_args):
    for i in range(len(path_args)):
        if path_args[i]:
            if '\\' in path_args[i]:
                path_args[i] = path_args[i].replace('\\','/')
    return path_args
            
def ETA(startTime,partDone,include_seconds = True):
    #Estimates the ETA of a function given the time the function has started and the completion ratio
    
    ElapsedTime = time.time() - startTime
    TimeLeft = (1/partDone)*ElapsedTime - ElapsedTime
    if include_seconds == False:
        print('Time Elapsed: {}h {}m --- ETA: {}h {}m'.format(int(ElapsedTime//3600),int((ElapsedTime%3600)//60),int(TimeLeft//3600),int((TimeLeft%3600)//60)))
    else:
        print('Time Elapsed: {}h {}m {}s --- ETA: {}h {}m {}s'.format(int(ElapsedTime//3600),int((ElapsedTime%3600)//60),int(ElapsedTime%60),int(TimeLeft//3600),int((TimeLeft%3600)//60),int(TimeLeft%60)))
        
def string_to_bool(args):
    #Convert a list of strings to list of booleans
    args_bool = []
    for arg in args:
        if arg.lower() == 'true':
            args_bool.append(True)
            # arg_bool = True
            # return arg_bool
    
        elif arg.lower() == 'false':
            args_bool.append(False)
            # arg_bool = False
            # return arg_bool
    
        else:
            print('Error, input should be \'False\' or \'True\'')
        
    return args_bool
        
#%% Preprocessing functions
def one_hot_encode_RNA(seq):
    mapping = dict(zip("ACGU", range(4)))
    seq2 = [mapping[i] for i in seq]
    return np.eye(4)[seq2].astype(int)

def one_hot_encode_DNA(seq):
    mapping = dict(zip("ACGT", range(4)))
    seq2 = [mapping[i] for i in seq]
    return np.eye(4)[seq2].astype(int)
    

def preprocess(Protein,Cell_Type,df_eclip,SHAPE_model_dir):
    '''
    Input: Name of protein, name of cell type, the eCLIP dataframe containing the sequence and the measured icSHAPE vectors
    Output: 
        *processed data (needed as the model's input):
            x (sequence and icSHAPE)
            y(eCLIP score)
            fpkm score
            RNA_annotations
        *label_binarizer
        *ENST_id
    '''
    
    RNA_annotations_unique_list = ['transcript', 'exon', 'CDS', 'start_codon', 'stop_codon', 'five_prime_utr', 'three_prime_utr','Selenocysteine']

    #OHE RNA annotations
    RNA_annotations_list = []
    for i in range(len(df_eclip)):
        array_counter = np.zeros(len(RNA_annotations_unique_list))
        
        RNA_annotation_arr = np.array(df_eclip['RNA_annotations'][i].split(','))
        
        if len(RNA_annotation_arr) == 1:
            RNA_annotations_list.append(RNA_annotation_arr[0].split(':')[0])
        else:
            for k in range(len(RNA_annotation_arr)):
                RNA_annotation_index = RNA_annotations_unique_list.index(RNA_annotation_arr[k].split(':')[0])
                window_size = int(float(RNA_annotation_arr[k].split(':')[2])) - int(float(RNA_annotation_arr[k].split(':')[1]))
                array_counter[RNA_annotation_index] += window_size
            max_index = np.argmax(array_counter)
            RNA_annotations_list.append(RNA_annotations_unique_list[max_index])
    
    df_eclip_RNA_annotations_np = np.array(RNA_annotations_list)
        
    label_binarizer = preprocessing.LabelBinarizer()
    label_binarizer.fit(RNA_annotations_unique_list)
    RNA_annotations_OHE = label_binarizer.transform(df_eclip_RNA_annotations_np)
    
    #Process FPKM
    df_eclip_fpkm_np = df_eclip['FPKM'].to_numpy()
    fpkmReplacement = np.median(df_eclip_fpkm_np[~np.isnan(df_eclip_fpkm_np.astype(float))].astype(float))
    df_eclip['FPKM'] = df_eclip['FPKM'].replace(np.nan,fpkmReplacement)
    df_eclip['logFPKM'] = np.log((df_eclip['FPKM'].astype(float))+1)
      
    #Scores Handling 
    df_scores = df_eclip['eCLIP_Score'].to_numpy()
    
    #Real SHAPE Handling
    df_realSHAPE_raw = df_eclip['SHAPE'].to_numpy()
    df_realSHAPE = np.zeros((len(df_realSHAPE_raw),length_of_seq))
    for i in range(len(df_realSHAPE_raw)):
        df_realSHAPE[i,:] = np.array(df_realSHAPE_raw[i].split(','))
    df_realSHAPE = df_realSHAPE.reshape(df_realSHAPE.shape[0],df_realSHAPE.shape[1],1)
    
    #Sequence Handling
    df_eclip_seq = df_eclip['Sequence'].to_numpy()
    fpkm_set = df_eclip['logFPKM'].to_numpy()

    #Predict SHAPE 
    print('Loading icSHAPE model of cell type: \t{}'.format(Cell_Type))
    SHAPE_Model = tf.keras.models.load_model(SHAPE_model_dir+'/InVivo_{}_PrismNet_20nbs'.format(Cell_Type))
    
    #Preparing data for prediction
    OHE_Seq_Ext = np.zeros((len(df_eclip_seq),(length_of_seq+Length_of_nbs*2)*4))
    
    #Padding Sequence Matrix with zeros from left and right
    for i in range(len(df_eclip_seq)):
        OHE_Seq_Ext[i,Length_of_nbs*4:Length_of_nbs*4+length_of_seq*4] = one_hot_encode_DNA(df_eclip_seq[i]).flatten()
        
    #Expanding matrix so each row contains data of 1 nucleotide
    dataToPredict = np.zeros((length_of_seq*len(df_eclip_seq),(Length_of_nbs*2+1)*4),dtype='float32')
    for j in range(len(df_eclip_seq)):
        for i in range(length_of_seq):
            dataToPredict[length_of_seq*j+i,:] = OHE_Seq_Ext[j,4*i:4*i+(Length_of_nbs*2+1)*4]
    
    print('------ Predicting missing icSHAPE values ------')

    predictedSHAPE = SHAPE_Model.predict(dataToPredict,batch_size=5000)
    predictedSHAPE_reshaped = predictedSHAPE.reshape(int(predictedSHAPE.shape[0]/length_of_seq),length_of_seq,1)
    
    eclip_OHE = OHE_Seq_Ext[:,Length_of_nbs*4:Length_of_nbs*4+length_of_seq*4]
    eclip_OHE_reshaped = eclip_OHE.reshape(eclip_OHE.shape[0],length_of_seq,4)
                 
    #-------------------------------------------------------------------------
    #Mix Real and predicted SHAPE into one vector
    ConcatenatedSet = np.concatenate((eclip_OHE_reshaped,predictedSHAPE_reshaped,df_realSHAPE),axis=2) #Add Predicted SHAPE, realSHAPE
    
    #Create vector indicating '1' where measured icSHAPE exists and '0' for 'None'
    isRealSHAPEvec = np.ones((len(eclip_OHE_reshaped),length_of_seq))
    for i in range(len(isRealSHAPEvec)):
        for j in range(length_of_seq):
            if ConcatenatedSet[i,j,5] == -1:
                isRealSHAPEvec[i,j] = 0
                ConcatenatedSet[i,j,5] = ConcatenatedSet[i,j,4]        # if no measurement exists replace with SHAPE prediction 
    
    ConcatenatedSet[:,:length_of_seq,4] = isRealSHAPEvec           #Replace rest of SHAPE predictions with binary vector indicator
    
    ConcatenatedSetSeq = ConcatenatedSet[:,:,:4]
    ConcatenatedSetSHAPE = ConcatenatedSet[:,:,5]
    ConcatenatedSetSHAPE = ConcatenatedSetSHAPE.reshape(ConcatenatedSetSHAPE.shape[0],ConcatenatedSetSHAPE.shape[1],1)
    ConcatenatedSet = np.concatenate((ConcatenatedSetSeq,ConcatenatedSetSHAPE),axis=2) #Add Predicted SHAPE, realSHAPE

    x = ConcatenatedSet.astype(float)
    y = df_scores.astype(float)
    fpkm = fpkm_set.astype(float)
    RNA_annotations = RNA_annotations_OHE.astype(float)
    ENST_id = df_eclip['ENST_Indices']
    
    return x,y,fpkm,RNA_annotations,label_binarizer, ENST_id
# --------------------------------

def inverseLabelBinarizerTransform(RNA_annotations,label_binarizer):
    RNA_annotations_labels = label_binarizer.inverse_transform(RNA_annotations)
    return RNA_annotations_labels

def one_hot_encode_SHAPE(seq,output_type = int):
    mapping = dict(zip("PU", range(2)))
    seq2 = [mapping[i] for i in seq]
    return np.eye(2)[seq2].astype(output_type)

#%% Interpretation functions
def create_DNA_logo(PWM, Protein, Cell_Type, Title, path_plot, figsize=(16, 4), show_score = False, labelpad=-1, ax=None, show_only_y = False, show_Title = True):
    
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if show_score == False:
        ax.axis('off')
    
    if show_only_y == True:
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.box(False)
        ax.tick_params(axis='both', which='major', labelsize=14)

    
    PWM_df = pd.DataFrame(PWM, columns=['A', 'C', 'G', 'U'])
    IG_logo = logomaker.Logo(PWM_df,
                             shade_below=.5,
                             fade_below=.5,
                             color_scheme='classic',
                             ax=ax,
                             figsize=figsize)

    if show_Title == True:
        IG_logo.ax.set_title(Title , loc='left', fontsize = 25)

    create_dir(path_plot,silence=True)
    
    if Cell_Type == '': #i.e. 2 cell types
        plt.savefig(path_plot+'{}.png'.format(Protein),bbox_inches='tight', dpi = 100)
    else:
        plt.savefig(path_plot+'{}_{}.png'.format(Protein,Cell_Type),bbox_inches='tight', dpi = 100)
    plt.close(fig)


def create_structure_logo(PWM, Protein, Cell_Type, Title, path_plot, figsize=(16, 3),show_score = False, labelpad=-1, ax=None, show_only_y = False):
    PWM_df = pd.DataFrame(PWM, columns=['P', 'U'])

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if show_score == False:
        ax.axis('off')
        
    if show_only_y == True:
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis='both', which='major', labelsize=14)

    IG_logo = logomaker.Logo(-PWM_df,
                             shade_below=0, #shade_below=.5
                             fade_below=0, #fade_below=.5
                             color_scheme={'U': 'purple',
                              'P': 'brown'},
                             ax=ax,
                             flip_below = False,
                             figsize=figsize)

    IG_logo.ax.xaxis.set_ticks_position('none')
    IG_logo.ax.xaxis.set_tick_params('both')
    IG_logo.ax.set_xticklabels([])

    create_dir(path_plot,silence=True)
      
    if Cell_Type == '': #i.e. 2 cell types
        plt.savefig(path_plot+'{}.png'.format(Protein),bbox_inches='tight', dpi = 100)
    else:
        plt.savefig(path_plot+'{}_{}.png'.format(Protein,Cell_Type),bbox_inches='tight', dpi = 100)
    plt.close(fig)


def create_SHAPE_plot(SHAPE_data, Protein, Cell_Type, path_plot, figsize=(16, 3),show_icSHAPE_median_line = False,show_score = False, show_only_y = False):
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if show_score == False:
        plt.yticks([])
        plt.xticks([])
        ax.spines['top'].set_visible(True)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

    if show_only_y == True:
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.box(False)
        ax.tick_params(axis='both', which='major', labelsize=14)
        
    ax.bar(np.arange(len(SHAPE_data)),SHAPE_data)
    if show_icSHAPE_median_line:
        ax.axhline(y=0.233,linestyle='--',color='gray')
    ax.set_xlim(-0.5, len(SHAPE_data)-0.5)

    create_dir(path_plot,silence=True)
    plt.savefig(path_plot+'{}_{}.png'.format(Protein,Cell_Type),bbox_inches='tight', dpi=100)
    plt.close(fig)
    
def get_gradients(model, sample_inputs, target_range=None, jacobian=False):
    """Computes the gradients of outputs w.r.t input.

    Args:
        sample_inputs (ndarray):: model sample inputs 
        target_rtarget_range (slice)ange: Range of target 

    Returns:
        Gradients of the predictions w.r.t input
    """
    if isinstance(sample_inputs, list):
        for i in range(len(sample_inputs)):
            sample_inputs [i] = tf.cast(sample_inputs[i], tf.float32)
    else:
        sample_inputs = tf.cast(sample_inputs, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(sample_inputs)
        preds = model(sample_inputs)
        if (target_range is None):
            target_preds = preds[:, :]
        else:
            target_preds = preds[:, target_range]

    if(jacobian):
        grads = tape.jacobian(target_preds, sample_inputs)
    else:
        grads = tape.gradient(target_preds, sample_inputs)
    return grads

def linearly_interpolate(model, sample_input, baseline=None, num_steps=50, multiple_samples=False):
    # If baseline is not provided, start with a zero baseline
    # having same size as the sample input.
    if baseline is None:
        baseline = np.zeros(sample_input.shape).astype(np.float32)
    else:
        baseline = baseline.astype(np.float32)

    #Do interpolation.
    sample_input = sample_input.astype(np.float32)
    interpolated_sample_input = [
        baseline + (step / num_steps) * (sample_input - baseline)
        for step in range(num_steps + 1)
    ]

    #Modified - so that SHAPE, FPKM and RNA annotations are constant
    

    interpolated_sample_input = np.array(interpolated_sample_input).astype(np.float32)
    if(multiple_samples):
        old_shape = interpolated_sample_input.shape
        new_transpose_form = (1,0)+tuple(range(interpolated_sample_input.ndim)[2:]) #switch the two first axises
        new_shape_form = (old_shape[0]*old_shape[1],) + old_shape[2:]
        interpolated_sample_input = interpolated_sample_input.transpose(new_transpose_form).reshape(new_shape_form)

    
    
    return interpolated_sample_input, sample_input, baseline 

def get_integrated_gradients(model, sample_inputs, target_range=None, baselines=None, num_steps=50, multiple_samples=False, return_hyp=False):
    """Computes Integrated Gradients for range of labels.

    Args:
        model (tensorflow model): Model
        sample_inputs (ndarray): Original sample input to the model 
        target_range (slice): Target range - grdient of Target range  with respect to the input
        baseline (ndarray): The baseline to start with for interpolation
        num_steps: Number of interpolation steps between the baseline
            and the input used in the computation of integrated gradients. These
            steps along determine the integral approximation error. By default,
            num_steps is set to 50.

    Returns:
        integrated_grads_list : Integrated gradients w.r.t input
        hypothetical_importance_list : hypothetical importance w.r.t input
    """
    #tf.compat.v1.enable_v2_behavior()
    if isinstance(sample_inputs, list):
        num_of_inputs_types = len(sample_inputs)
    else:
        #insert the inputs to a list to fit generalized code
        num_of_inputs_types = 1
        sample_inputs = [sample_inputs]
        if (baselines is not None):
            baselines = [baselines]
    
    # 1. Do interpolation.
    output = []
    if(baselines is None):
        for sample_input in sample_inputs:
            output.append(linearly_interpolate(model, sample_input, baselines, num_steps=num_steps, multiple_samples=multiple_samples))
    else:
        for sample_input, baseline in zip(sample_inputs, baselines):
            output.append(linearly_interpolate(model, sample_input, baseline, num_steps=num_steps, multiple_samples=multiple_samples))

    interpolated_samples_inputs = [x[0] for x in output]
    sample_inputs = [x[1] for x in output]
    baselines = [x[2] for x in output]

    # 2. Get the gradients
    if (num_of_inputs_types == 1):
        grads_values = get_gradients(model, interpolated_samples_inputs[0], target_range=target_range)
        grads_list = [tf.convert_to_tensor(grads_values, dtype=tf.float32)]
    else:
        grads_list = get_gradients(model, interpolated_samples_inputs, target_range=target_range)
        grads_list = [tf.convert_to_tensor(grads, dtype=tf.float32) for grads in grads_list]

    if(multiple_samples):
        num_of_samples = sample_inputs[0].shape[0]
        grads_list = [tf.reshape(grads, [num_of_samples, num_steps+1] + grads.shape.as_list()[1:]) for grads in grads_list]

    # 3. Approximate the integral using the trapezoidal rule
    if(multiple_samples):
        grads_list = [(grads[:, :-1] + grads[:,1:]) / 2.0 for grads in grads_list]
        avg_grads_list = [tf.reduce_mean(grads, axis=1) for grads in grads_list]
    else: 
        grads_list = [(grads[:-1] + grads[1:]) / 2.0 for grads in grads_list]
        avg_grads_list = [tf.reduce_mean(grads, axis=0) for grads in grads_list]  

    # 4. get hypothetical importance score - it's the average gradient
    hypothetical_importance_list = [avg_grads.numpy() for avg_grads in avg_grads_list]

    # 5. Calculate integrated gradients and return
    integrated_grads_list = [(sample_inputs[i] - baselines[i]) * avg_grads_list[i] for i in range(num_of_inputs_types)]
    integrated_grads_list = [integrated_grads.numpy() for integrated_grads in integrated_grads_list]
    if (num_of_inputs_types == 1):
        if(return_hyp):
          return integrated_grads_list[0], hypothetical_importance_list[0]
        else:
          return integrated_grads_list[0]
    
    if(return_hyp):
      return integrated_grads_list, hypothetical_importance_list
    else:
      return integrated_grads_list
  
def get_HAR_window(PWM_matrix, window_length = 20):
    
    #Recieves a PWM matrix with a shape of sequence length x 4 ('ACGT')
    #Returns HAR window with maximum absolute IG value sum
    
    max_sum = 0
    HAR_window = np.zeros((window_length,PWM_matrix.shape[-1]))
    for i in range(len(PWM_matrix)-window_length):
        curr_window_sum = np.sum(np.abs(PWM_matrix[i:i+window_length]))
        if curr_window_sum > max_sum:
            max_sum = curr_window_sum
            HAR_window = PWM_matrix[i:i+window_length]
    
    return HAR_window, max_sum

def kmer_to_string(kmer):
    #Recieves an Nx5 kmer (sequence, SHAPE) and returns a string vector for sequence over {ACGU} and a string vector for SHAPE {P,U}
    seq_string = ''
    SHAPE_string = ''
    
    nuc_dict = ['A','C','G','U']

    if len(kmer.shape) == 2:
        for i in range(len(kmer)):
            #Sequence handling
            curr_arg_max = np.argmax(kmer[i,:4])
            seq_string += nuc_dict[curr_arg_max]
            
            #SHAPE handling (<0.233 = paired)
            if kmer[i,4] < 0.233:
                SHAPE_string += 'P'
            else:
                SHAPE_string += 'U'
    else:
        print('Error - Input\'s shape is {} and should be 2'.format(len(kmer.shape)))
    
    return seq_string, SHAPE_string

def hamming_distance(kmer_seq_src, kmer_seq_dest, kmer_SHAPE_src, kmer_SHAPE_dst):
    seq_distance = 0
    SHAPE_distance = 0
    
    for i in range(len(kmer_seq_src)):
        if kmer_seq_src[i] != kmer_seq_dest[i]:
            seq_distance += 1
        if kmer_SHAPE_src[i] != kmer_SHAPE_dst[i]:
            SHAPE_distance +=1 
            
    return seq_distance, SHAPE_distance
    
def shift_distance (kmer_seq_src, kmer_seq_dest, kmer_SHAPE_src, kmer_SHAPE_dst, kmer_len):
    #returns if destination kmer is with 1 shift right (+1) or 1 shift left (-1)
    shift = 0
    if   (kmer_seq_src[0:kmer_len-1] == kmer_seq_dest[1:kmer_len]) & (kmer_SHAPE_src[0:kmer_len-1] == kmer_SHAPE_dst[1:kmer_len]):
        shift = 1
    elif (kmer_seq_src[1:kmer_len] == kmer_seq_dest[0:kmer_len-1]) & (kmer_SHAPE_src[1:kmer_len] == kmer_SHAPE_dst[0:kmer_len-1]):
        shift = -1
    elif (kmer_seq_src[0:kmer_len-2] == kmer_seq_dest[2:kmer_len]) & (kmer_SHAPE_src[0:kmer_len-2] == kmer_SHAPE_dst[2:kmer_len]):
        shift = 2
    elif (kmer_seq_src[2:kmer_len] == kmer_seq_dest[0:kmer_len-2]) & (kmer_SHAPE_src[2:kmer_len] == kmer_SHAPE_dst[0:kmer_len-2]):
        shift = -2
    elif (kmer_seq_src[0:kmer_len-3] == kmer_seq_dest[3:kmer_len]) & (kmer_SHAPE_src[0:kmer_len-3] == kmer_SHAPE_dst[3:kmer_len]):
        shift = 3
    elif (kmer_seq_src[3:kmer_len] == kmer_seq_dest[0:kmer_len-3]) & (kmer_SHAPE_src[3:kmer_len] == kmer_SHAPE_dst[0:kmer_len-3]):
        shift = -3

    return shift

def divide_into_kmers (HAR_windows, kmer_len = 6):
    #Recieves HAR windows PWM and returns the unique kmers and the sum of their scores
    kmer_unique_list = []
    kmer_scores = []
    
    for i in range(len(HAR_windows)):
        for j in range(HAR_windows.shape[1]-kmer_len):
            curr_kmer = kmer_to_string(HAR_windows[i][j:j+kmer_len])
            
            if curr_kmer in kmer_unique_list:
                existing_kmer_idx = kmer_unique_list.index(curr_kmer)
                kmer_scores[existing_kmer_idx] += HAR_windows[i][j:j+kmer_len]
            
            else:
                kmer_unique_list.append(curr_kmer)
                kmer_scores.append(HAR_windows[i][j:j+kmer_len])
            
    return kmer_unique_list, kmer_scores