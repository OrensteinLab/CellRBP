# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 11:39:36 2023

@author: orifl
"""

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

from Util_Functions import ETA
import time
import pickle

eCLIP_Seq_Len = 101

def Delete_Corrupted_Samples (df_eclip, letters_list):
    #Deletes rows with nucelotides that are not ACGT
    
    #Input: eCLIP dataframe, list of potential corruptions such as ['N','W','Y']
    #Output: Filtered eCLIP dataframe
    
    raw_len = len(df_eclip)
    for letter in letters_list:
        df_eclip = df_eclip[~df_eclip['Sequence'].str.contains(letter)]
        
    df_eclip = df_eclip.reset_index(drop=True)
    print('{} Rows were deleted'.format(raw_len-len(df_eclip)))
    return df_eclip 

def Shuffle_Divide_Dataset (df_eclip, train_part = 0.7, validation_part = 0.15, test_part = 0.15):
    #Shuffle and divide the eCLIP data into train, validation and test sets
    
    #Input: eCLIP dataframe, ratios of training, validation and test.
    #Ouput: Shuffled train, validation and test sets
    
    df_eclip_shuff = df_eclip.sample(frac=1, random_state=42)
    
    #Train [default 70%]
    train_Samples = int(len(df_eclip_shuff)*train_part)
    df_eclip_shuff_train = df_eclip_shuff[:train_Samples].copy()
    
    #Validation [default 15%]
    validation_Samples = int(len(df_eclip_shuff)*validation_part)
    df_eclip_shuff_valid = df_eclip_shuff[train_Samples:train_Samples+validation_Samples].copy()

    #Test [default 15%]
    df_eclip_shuff_test = df_eclip_shuff[train_Samples+validation_Samples:]
    
    return df_eclip_shuff_train, df_eclip_shuff_valid, df_eclip_shuff_test

def get_FPKM (df_eCLIP, path_FPKM):
    #Adds FPKM data column to the eCLIP dataframe
    
    #Input: eCLIP dataframe, FPKM file path, Cell_Type ('K562' or 'HepG2')
    #Output: eCLIP dataframe with an additional FPKM column
    
    df_FPKM = pd.read_csv(path_FPKM,delimiter='\t', low_memory=False)
    df_FPKM_filt = df_FPKM[['transcript_id','FPKM']].copy() #Keep only 2 cols: ENST index, FPKM
    df_FPKM_filt = df_FPKM_filt[df_FPKM_filt['transcript_id'].str.contains('ENST')].reset_index(drop=True)#Delete rows that don't contain 'ENST'
    df_FPKM_filt['transcript_id_int'] = df_FPKM_filt['transcript_id'].str.split('ENST',expand=True)[1].str.split('.',expand=True)[0].astype(int)
    
    df_eCLIP['ENST_int'] = df_eCLIP['ENST_Indices'].str.split('|',expand=True)[0].str.split('ENST',expand=True)[1].astype(int)
    FPKM_arr = np.zeros(len(df_eCLIP))*np.nan
    
    start_time = time.time()
    for i in range(len(df_eCLIP)):
        if sum(df_FPKM_filt['transcript_id_int']==df_eCLIP['ENST_int'][i]):
            idx_FPKM = df_FPKM_filt.index[df_FPKM_filt['transcript_id_int']==df_eCLIP['ENST_int'][i]][0]
            FPKM_arr[i] = df_FPKM_filt['FPKM'][idx_FPKM]
            
        if (i % 1000 == 0) or (i == len(df_eCLIP) - 1):
            print('Sample No.:{}/{} [{:.2f}%]'.format(i,len(df_eCLIP),(i*100/len(df_eCLIP))))  
            if i > 0:
                ETA(start_time,i/len(df_eCLIP))
    
    df_eCLIP['FPKM'] = FPKM_arr     #Add FPKM values
    df_eCLIP = df_eCLIP.drop(['ENST_int'],axis=1) #Delete additional ENST column
    return df_eCLIP

def get_CHR_idx (df_eclip,transcripts_Path):
    #Adds 7 columns: ENST no., ENST Start, ENST End, Chr, chr start (arr), chr end (arr), strand
    
    #Input: eCLIP dataframe, transcripts pickle file path
    #Output: eCLIP dataframe + the following data columns: ENST no., ENST Start, ENST End, Chr, chr start (arr), chr end (arr), strand
    
    with open(transcripts_Path, 'rb') as f:
        df_Transcripts = pickle.load(f)
    
    df_eclip['ENST'] = df_eclip['ENST_Indices'].str.split('|',expand=True)[0].str.split('ENST',expand=True)[1].astype(int)
    df_eclip['ENST_Start'] = df_eclip['ENST_Indices'].str.split('|',expand=True)[1].astype(int)
    df_eclip['ENST_End'] = df_eclip['ENST_Indices'].str.split('|',expand=True)[2].astype(int)
    
    df_eclip['chr'] = ''
    df_eclip['chr_start'] = ''
    df_eclip['chr_end'] = ''
    df_eclip['strand'] = ''

    start_time = time.time()
    for transcript_no in range(len(df_eclip)):
        
        if (transcript_no % 2000 == 0) or (transcript_no == len(df_eclip) - 1):
            if transcript_no>0:
                print('Transcript No.:{}/{} [{:.2f}%]'.format(transcript_no,len(df_eclip),(transcript_no*100/len(df_eclip))))  
                ETA(start_time,transcript_no/len(df_eclip))
            
        exons_info = df_Transcripts.loc[df_Transcripts['ENST'] == df_eclip['ENST'][transcript_no]]['exon_chr_info'].reset_index(drop=True)[0]
        exons_info['ENST_Start'] = ''
        exons_info['ENST_End'] = ''
        for i in range(len(exons_info)):
            if i==0:
                exons_info['ENST_Start'][i] = 1
                exons_info['ENST_End'][i] = exons_info['ENST_Start'][i] + (exons_info['chr_end'][i]   - exons_info['chr_start'][i])
            else:
                exons_info['ENST_Start'][i] = exons_info['ENST_End'][i-1] + 1
                exons_info['ENST_End'][i] = exons_info['ENST_Start'][i] + (exons_info['chr_end'][i]  - exons_info['chr_start'][i])
            
        specific_exons = exons_info.loc[~(exons_info['ENST_Start']>df_eclip['ENST_End'][transcript_no]) & ~(exons_info['ENST_End']<df_eclip['ENST_Start'][transcript_no])].reset_index(drop=True)
        chr_start = []
        chr_end = []
        #If strand is positive - go upstream
        if exons_info['strand'].to_numpy()[0] == '+':
            if len(specific_exons) == 1:
                chr_start.append((specific_exons['chr_start'] + df_eclip['ENST_Start'][transcript_no] - specific_exons['ENST_Start'] - 1).to_numpy()[0])
                chr_end.append((specific_exons['chr_start'] + df_eclip['ENST_End'][transcript_no] - specific_exons['ENST_Start'] ).to_numpy()[0])
            else:
                cumulative_offset = 0 
                for j in range(len(specific_exons)):
                    if j==0:
                        chr_start.append(specific_exons['chr_start'][0] + df_eclip['ENST_Start'][transcript_no] - specific_exons['ENST_Start'][0] - 1)
                        chr_end.append(chr_start[0] + (specific_exons['ENST_End'][0] + 1 - df_eclip['ENST_Start'][transcript_no]))
                        cumulative_offset = chr_end[0] - chr_start[0]
                    else:
                        if (specific_exons['ENST_End'][j] - specific_exons['ENST_Start'][j]) > (eCLIP_Seq_Len - cumulative_offset):
                            chr_start.append(specific_exons['chr_start'][j] - 1)
                            chr_end.append(specific_exons['chr_start'][j] + eCLIP_Seq_Len - cumulative_offset - 1)
                        else:
                            chr_start.append(specific_exons['chr_start'][j] - 1)
                            chr_end.append(specific_exons['chr_end'][j])
                            cumulative_offset += chr_end[j] - chr_start[j]
        
        #If strand is negative - go downstream
        if exons_info['strand'].to_numpy()[0] == '-':
            if len(specific_exons) == 1:
                chr_start.append((specific_exons['chr_end'] - df_eclip['ENST_End'][transcript_no] + specific_exons['ENST_Start'] - 1).to_numpy()[0])
                chr_end.append(chr_start[0]+eCLIP_Seq_Len)
            else:
                cumulative_offset = 0 
                for j in range(len(specific_exons)):
                    if j==0:
                        chr_start.append(specific_exons['chr_start'][0] - 1)
                        chr_end.append(specific_exons['chr_end'][0] - (df_eclip['ENST_Start'][transcript_no] - specific_exons['ENST_Start'][0]))
                        cumulative_offset = chr_end[0] - chr_start[0]
                    else:
                        if (specific_exons['ENST_End'][j] - specific_exons['ENST_Start'][j]) > (eCLIP_Seq_Len - cumulative_offset):
                            chr_start.append(specific_exons['chr_end'][j] - eCLIP_Seq_Len + cumulative_offset)
                            chr_end.append(specific_exons['chr_end'][j])
                        else:
                            chr_start.append(specific_exons['chr_start'][j] - 1)
                            chr_end.append(specific_exons['chr_end'][j])
                            cumulative_offset += chr_end[j] - chr_start[j]
        
        df_eclip['chr'][transcript_no] = specific_exons['chr'][0]
        df_eclip['chr_start'][transcript_no] = chr_start
        df_eclip['chr_end'][transcript_no] = chr_end
        df_eclip['strand'][transcript_no] = specific_exons['strand'].to_numpy()[0]
        
    for i in range(len(df_eclip)):
        chr_start_list = df_eclip['chr_start'][i]
        chr_end_list = df_eclip['chr_end'][i]
 
        for j in range(len(chr_start_list)):
            if j==0:
                chr_start_str = str(int(chr_start_list[j]))
                chr_end_str = str(int(chr_end_list[j]))
            else:
                chr_start_str += ',{}'.format(int(chr_start_list[j]))
                chr_end_str += ',{}'.format(int(chr_end_list[j]))
        
        df_eclip['chr_start'][i] = chr_start_str
        df_eclip['chr_end'][i] = chr_end_str
    
    df_eclip_CHR_idxs = df_eclip
    return df_eclip_CHR_idxs
    
def get_RNA_annotation (df_eclip,GTF_path):
    #Given eCLIP dataframe and GTF path, returns eCLIP dataframe with an additional column with the major RNA annotations
    
    #Input: eCLIP dataframe, GTF file path
    #Output: eCLIP dataframe + Major RNA annotations column
    
    print('Reading GTF File...')
    column_names = ['chr','source','annotation','chr_start','chr_end','dc1','strand','dc2','additional_info']
    df_gtf = pd.read_csv(GTF_path, delimiter='\t', names = column_names, low_memory=False)
    df_gtf = df_gtf[5:]
    df_gtf = df_gtf[df_gtf['additional_info'].str.contains('ENST')].reset_index(drop=True) #Drop rows that don't contain gene ids
    additional_info_cols = df_gtf['additional_info'].str.split('"',expand=True)
    df_gtf['ENST'] = additional_info_cols[5].str.split('ENST',expand=True)[1].str.split('.',expand=True)[0].astype(int)
    df_gtf = df_gtf.drop(columns=['source','dc1','dc2','additional_info'])
    
    #eCLIP
    df_eclip['RNA_annotations'] = ''
    start_time=time.time()
    for i in range(len(df_eclip)): 
        
        if (i % 1000 == 0) or (i == len(df_eclip) - 1):
            if i>0:
                print('Sample No.:{}/{} [{:.2f}%]'.format(i,len(df_eclip),(i*100/len(df_eclip))))  
                ETA(start_time,i/len(df_eclip))
                
        df_eclip['chr_start'][i] = np.array(df_eclip['chr_start'][i].split(',')).astype(int)
        df_eclip['chr_end'][i] = np.array(df_eclip['chr_end'][i].split(',')).astype(int)
        
        RNA_annotations_list = []
        offset = 0
        for j in range(len(df_eclip['chr_start'][i])):
            ENST_specific = df_gtf.loc[df_gtf['ENST'] == df_eclip['ENST'][i]]
            chr_window_specific = ENST_specific.loc[~(ENST_specific['chr_start']>df_eclip['chr_end'][i][j]) & ~(ENST_specific['chr_end']<df_eclip['chr_start'][i][j])]
            #Debug
            #print(chr_window_specific)
            
            if (len(chr_window_specific)>=1) & (len(chr_window_specific)<=3):       #if only 1 RNA annotation: take the last one
                RNA_annotations_list.append(chr_window_specific['annotation'].to_numpy()[-1]+':{}:{}'.format(offset,offset+df_eclip['chr_end'][i][j] -1 - df_eclip['chr_start'][i][j]))
                offset += df_eclip['chr_end'][i][j] -1 -df_eclip['chr_start'][i][j] + 1
            else:                                                                   #if more than 1 RNA annotation
                sequence_window_length_iter = df_eclip['chr_end'][i][j] -1 -df_eclip['chr_start'][i][j]
                chr_window_specific = chr_window_specific.loc[~(chr_window_specific['annotation']=='transcript') & ~(chr_window_specific['annotation']=='exon')]
                
                if chr_window_specific['strand'].to_numpy()[0] == '+':
                    chr_window_specific = chr_window_specific.sort_values('chr_start',ascending=False)
                    chr_window_specific_np = chr_window_specific.to_numpy()
                    eCLIP_chr_start_iter = df_eclip['chr_start'][i][j] + 1
                    while(sequence_window_length_iter > 0):
                        if len(chr_window_specific_np)>0: #if CDS/3UTR/5UTR
                            annotation_window = chr_window_specific_np[-1][3] - eCLIP_chr_start_iter  #chr_end - chr_start
                            if (annotation_window) < (sequence_window_length_iter): #In case annotation window is smaller than requested window - take full window length
                                RNA_annotations_list.append(chr_window_specific_np[-1][1]+':{}:{}'.format(int(offset),int(offset+annotation_window)))
                                offset += annotation_window + 1
                                eCLIP_chr_start_iter = chr_window_specific_np[-1][3] + 1
                                #Continue from here - turn df to np array
                                chr_window_specific_np = chr_window_specific_np[:-1]
                                sequence_window_length_iter = sequence_window_length_iter - annotation_window - 1
                            else: #in case annotation window is bigger than requested window 
                                RNA_annotations_list.append(chr_window_specific_np[-1][1]+':{}:{}'.format(int(offset),int(offset + sequence_window_length_iter)))
                                offset += sequence_window_length_iter + 1
                                sequence_window_length_iter = 0
                                
                        else:                           #if transcript/exon
                            chr_window_specific_np = ENST_specific.loc[~(ENST_specific['chr_start']>df_eclip['chr_end'][i][j]) & ~(ENST_specific['chr_end']<eCLIP_chr_start_iter)].to_numpy()
                            RNA_annotations_list.append(chr_window_specific_np[-1][1]+':{}:{}'.format(int(offset),int(offset + sequence_window_length_iter)))
                            offset += sequence_window_length_iter + 1
                            sequence_window_length_iter = 0
                            
                if chr_window_specific['strand'].to_numpy()[0] == '-':
                    chr_window_specific = chr_window_specific.sort_values('chr_start',ascending=True)
                    chr_window_specific_np = chr_window_specific.to_numpy()
                    eCLIP_chr_end_iter = df_eclip['chr_end'][i][j] - 1
                    while(sequence_window_length_iter > 0):
                        if len(chr_window_specific_np)>0: #if CDS/3UTR/5UTR
                            annotation_window = eCLIP_chr_end_iter - chr_window_specific_np[-1][2] #chr_end - chr_start
                            if (annotation_window) < (sequence_window_length_iter): #In case annotation window is smaller than requested window - take full window length
                                RNA_annotations_list.append(chr_window_specific_np[-1][1]+':{}:{}'.format(int(offset),int(offset+annotation_window)))
                                offset += annotation_window  + 1
                                eCLIP_chr_end_iter = chr_window_specific_np[-1][2] - 1
                                #Continue from here - turn df to np array
                                chr_window_specific_np = chr_window_specific_np[:-1]
                                sequence_window_length_iter = sequence_window_length_iter - annotation_window - 1
                            else: #in case annotation window is bigger than requested window 
                                RNA_annotations_list.append(chr_window_specific_np[-1][1]+':{}:{}'.format(int(offset),int(offset + sequence_window_length_iter)))
                                offset += sequence_window_length_iter + 1
                                sequence_window_length_iter = 0
                                
                        else:                           #if transcript/exon
                            chr_window_specific_np = ENST_specific.loc[~(ENST_specific['chr_start']>eCLIP_chr_end_iter) & ~(ENST_specific['chr_end']<df_eclip['chr_start'][i][j])].to_numpy()
                            RNA_annotations_list.append(chr_window_specific_np[-1][1]+':{}:{}'.format(int(offset),int(offset + sequence_window_length_iter)))
                            offset += sequence_window_length_iter + 1
                            sequence_window_length_iter = 0
                    
        df_eclip['RNA_annotations'][i] = RNA_annotations_list

    #transform RNA annotations, chr start and chr end to str for the eclip tsv file
    for i in range(len(df_eclip)):
        RNA_annotations_list = df_eclip['RNA_annotations'][i]
 
        for j in range(len(RNA_annotations_list)):
            if j==0:
                RNA_annotations_str = str(RNA_annotations_list[j])
            else:
                RNA_annotations_str += ',{}'.format(RNA_annotations_list[j])
        df_eclip['RNA_annotations'][i] = RNA_annotations_str
        
        chr_start_arr = df_eclip['chr_start'][i]
        chr_end_arr = df_eclip['chr_end'][i]
 
        for k in range(len(chr_start_arr)):
            if k==0:
                chr_start_str = str(int(chr_start_arr[k]))
                chr_end_str = str(int(chr_end_arr[k]))
            else:
                chr_start_str += ',{}'.format(int(chr_start_arr[k]))
                chr_end_str += ',{}'.format(int(chr_end_arr[k]))
        
        df_eclip['chr_start'][i] = chr_start_str
        df_eclip['chr_end'][i] = chr_end_str
    
    return df_eclip

def create_transcripts_func(transcripts_dir,FASTA_path,GTF_path):
    
    print('--- Collecting transcripts information [This may take a while] ---')
    # transcripts_dir = '../Features/'
    # GTF_path = '../Features/GTF/gencode.v26.annotation.gtf'
    # FASTA_path = '../Features/FASTA/gencode.v26.transcripts.fa'
    
    debug = False
    # import GTF file
    print('--- Reading and processing GTF file ---')
    
    column_names = ['chr','source','annotation','chr_start','chr_end','dc1','strand','dc2','additional_info']
    df_gtf_gencode = pd.read_csv(GTF_path, delimiter='\t', names = column_names, skiprows= 5)
    df_gtf_gencode = df_gtf_gencode[df_gtf_gencode['additional_info'].str.contains('ENST')].reset_index(drop=True) #Drop rows that don't contain gene ids
    additional_info_cols = df_gtf_gencode['additional_info'].str.split('"',expand=True)
    df_gtf_gencode['ENST'] = additional_info_cols[3].str.split('ENST',expand=True)[1].str.split('.',expand=True)[0].astype(int)
    df_gtf_gencode = df_gtf_gencode.drop(columns=['source','dc1','dc2','additional_info'])
    
    #Creates a new dataframe 'df_Transcripts' containing: Transcript Name, Start line, end line, sequence columns and sequence length 
    print('--- Reading and processing FASTA file ---')
    df = pd.read_csv(FASTA_path, sep='/t', names=['Transcript_Name'], engine='python')
    df_Transcripts = pd.DataFrame(columns = ['Transcript_Name','Line_Start','Line_End','Sequence','Seq_Length'])
    df_Transcripts ['Transcript_Name'] = df.loc[df['Transcript_Name'].str.contains('>ENST')]['Transcript_Name'].str.split('|').str[0]
    df_Transcripts['Line_Start'] = df_Transcripts.index                     #Add column - Index of transcript (in lines)
    df_Transcripts = df_Transcripts.reset_index(drop=True)
    
    df_Transcripts['Line_End'][:-1] = df_Transcripts['Line_Start'][1:]
    df_Transcripts['Line_End'][-1:] =  len(df)
    
    #
    #Insert sequences from each transcript to its relevant place in the dataframe
    print('--- Collecting transcript sequences and genome positions ---')
    start = time.time()
    for transcript_no in range(len(df_Transcripts)):
        transcript = df[df_Transcripts['Line_Start'][transcript_no]+1:df_Transcripts['Line_End'][transcript_no]].stack().str.cat()
        df_Transcripts.loc[transcript_no,'Sequence'] = transcript #[transcript_no] = transcript
        df_Transcripts.loc[transcript_no,'Seq_Length'] = len(transcript)
    
        if transcript_no % 10000 == 0:
            if transcript_no:
                ETA(start,transcript_no/len(df_Transcripts))
    
    # df_Transcripts = df_Transcripts.reset_index(drop=True)
    df_Transcripts['ENST'] = df_Transcripts['Transcript_Name'].str.split('ENST00000',expand=True)[1].str.split('.',expand=True)[0].astype(int)
    
    # Calculate length of seq by summing all exon lengths in transcript (for debugging)
    if debug:
        df_Transcripts['GTF_Seq_Length_(Debug)'] = ''
        for transcript_no in range((1000)):
            gtf_transcript_info = df_gtf_gencode.loc[df_gtf_gencode['ENST'] == df_Transcripts['ENST'][transcript_no]]
            gtf_transcript_info = gtf_transcript_info.reset_index(drop=True)
            total_exon_length = 0
            for i in range(len(gtf_transcript_info)):
                if gtf_transcript_info['annotation'][i] == 'exon':
                    total_exon_length += gtf_transcript_info['chr_end'][i] + 1 - gtf_transcript_info['chr_start'][i]
            
            df_Transcripts['GTF_Seq_Length_(Debug)'][transcript_no] = total_exon_length
        
    # get exon chr info
    print('--- Filtering Exons ---')
    df_Transcripts['exon_chr_info'] = ''
    start = time.time()
    
    for transcript_no in range(len(df_Transcripts)): #TEMP
        gtf_transcript_info = df_gtf_gencode.loc[df_gtf_gencode['ENST'] == df_Transcripts['ENST'][transcript_no]]
        df_Transcripts['exon_chr_info'][transcript_no] = gtf_transcript_info.loc[gtf_transcript_info['annotation']=='exon'].reset_index(drop=True)
        
        if transcript_no % 5000 == 0 and transcript_no!=0:
            ETA(start,transcript_no/len(df_Transcripts))
            
    #dump into pickle file
    with open(transcripts_dir+'df_transcripts.pkl', 'wb') as f:
        pickle.dump(df_Transcripts, f)
    print('transcripts information file was successfully created in: {}'.format(transcripts_dir+'df_transcripts.pkl'))
            
