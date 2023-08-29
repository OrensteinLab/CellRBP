#%%
import numpy as np
import os
import tensorflow as tf
import cv2
import pandas as pd
import time

from Util_Functions import create_dir,preprocess, create_DNA_logo, create_structure_logo, create_SHAPE_plot,get_integrated_gradients
from Util_Functions import inverseLabelBinarizerTransform, ETA, fix_path, string_to_bool

Sequence_len = 101
import argparse
parser = argparse.ArgumentParser('CellRBP')

parser.add_argument('--input_data_path',        type=str,       default='',                                         help='Input data path. Default: \'\'. Example: Data/clip_data_processed/AGGF1_HepG2/all.tsv')
parser.add_argument('--output_data_dir',        type=str,       default="Interpretation/Local_Interpretation/",  help='Output data path. Default: Interpretation/Local_Interpretation/')
parser.add_argument('--model_data_dir',         type=str,       default="",                                         help='Predictive model dir. Example: Models/CellRBP/AGGF1/HepG2/')
parser.add_argument('--icSHAPE_model_dir',      type=str,       default='Models/icSHAPE/',                       help='icSHAPE model\'s directory path. Default: Models/icSHAPE/') 

#IG local attributions 
parser.add_argument('--LOCAL_INTERPRETATION',   type=str,       default='False',                                    help='Create local interpretations for a given RBP [True,False]. Default: False')
parser.add_argument('--pred_score_thr',         type=float,     default="0.8",                                      help='Prediction score threshold. Samples with lower prediction score will not be processed. Default: 0.8')
parser.add_argument('--IG_num_steps',           type=int,       default="25",                                       help='Number of steps for Integrated Gradients. Default: 25')
parser.add_argument('--seq_baseline',           type=float,     default="0",                                        help='Sequence baseline for IG. Default: 0')
parser.add_argument('--str_baseline',           type=float,     default="0.233",                                    help='Structure baseline for IG. Default: 0.233 (Global SHAPE median score)')

#RNA annotations global attributions
parser.add_argument('--REGION_TYPES_INTERPRETATION',   type=str,     default='False',                               help='Create global RNA region types interpretation [True,False]. Default: False')

#Integrative Motifs
parser.add_argument('--INTEGRATIVE_MOTIFS',     type=str,       default='False',                                    help='Create sequence and structure integrative motifs [True,False]. Default: False')


args = parser.parse_args()
[input_data_path,output_data_dir,model_data_dir,icSHAPE_model_dir] = fix_path([args.input_data_path,args.output_data_dir,args.model_data_dir,args.icSHAPE_model_dir])
[LOCAL_INTERPRETATION, REGION_TYPES_INTERPRETATION, INTEGRATIVE_MOTIFS] = string_to_bool([args.LOCAL_INTERPRETATION, args.REGION_TYPES_INTERPRETATION, args.INTEGRATIVE_MOTIFS])

print('\nLocal Interpretation:\t\t\t{} \nRegion Types Interpretation:\t\t\t{} \nIntegrative Motifs:\t\t\t{} \n'.format(LOCAL_INTERPRETATION,REGION_TYPES_INTERPRETATION,INTEGRATIVE_MOTIFS))
print(input_data_path)
#%%
def IG_Seq_Str (data_path,path_Model,output_dir,icSHAPE_model_dir,pred_score_thr=0.8,IG_num_steps = 25,seq_baseline=0,str_baseline=0.233,figure_height = 2):
    
# data_dir = 'D:/1.Research/PrismNetGit_V2/PrismNet/data/clip_data_fpkm_chr_positions_RNA_annotations'
    
    Protein = data_path.split('/')[-2].split('.')[0].split('_')[0]
    Cell_Type = data_path.split('/')[-2].split('.')[0].split('_')[1]
    
    print('\nprediction score threshold: \t\t{}\nSequence baseline: \t\t\t{}\nStructure baseline: \t\t\t{}'.format(pred_score_thr,seq_baseline,str_baseline))
    print('Protein: \t\t\t\t{} \nCell Type: \t\t\t\t{} \nIG number of steps: \t\t\t{}\n'.format(Protein,Cell_Type,IG_num_steps))
    
    names = ['ENST_Indices','Sequence','SHAPE','eCLIP_Score','FPKM','ENST','ENST_Start','ENST_End','chr','chr_start','chr_end','strand','RNA_annotations']
    df_eclip = pd.read_csv(data_path,delimiter='\t',names=names, skiprows=1)
    
    x,y,fpkm,RNA_annotations,label_binarizer,ENST_id = preprocess(Protein,Cell_Type,df_eclip,icSHAPE_model_dir)
    
    list_dirs = ['Plots_Sequence/','Plots_Structure/','Plots_SHAPE/','Plots_Combined/']
    for directory in list_dirs:
        create_dir(output_dir+directory)
    
    Title = 'IG_Sequence_Structure_{}'.format(Protein)

    #Load model
    
    loaded_model = tf.keras.models.load_model(path_Model)
    
    y_predicted = loaded_model.predict([x,fpkm,RNA_annotations])
    
    for sample_idx in range(len(x)): 

        y_pred = y_predicted[sample_idx:sample_idx+1]
        if y_pred > pred_score_thr:
            print('Sample index: \t{} \nPrediction Score: \t{:.3f}'.format(sample_idx,y_pred[0][0]))
            
            x_sample = x[sample_idx:sample_idx+1]
            FPKM_sample = fpkm[sample_idx:sample_idx+1]
            RNA_Annotations_sample = RNA_annotations[sample_idx:sample_idx+1]
            
            #%%Get Sequence Integrated Gradients         
            RNA_Annotation_baseline = RNA_Annotations_sample #Keep RNA annotations const
            FPKM_baseline = FPKM_sample #Keep FPKM const
            
            x_baseline =  np.ones(x_sample.shape)*seq_baseline
            x_baseline[0,:,4] = x_sample[0,:,4] #SHAPE const
            
            baselines = [x_baseline, FPKM_baseline, RNA_Annotation_baseline]
            
            (seq_features_ig_scores, FPKM_ig_scores, RNA_Annotations_ig_scores) = get_integrated_gradients(model=loaded_model, sample_inputs=[x_sample,FPKM_sample,RNA_Annotations_sample], baselines=baselines, num_steps=IG_num_steps, multiple_samples=True)

            #%%Get icSHAPE Integrated Gradients            
            x_baseline =  np.ones(x_sample.shape)*str_baseline
            x_baseline[0,:,:4] = x_sample[0,:,:4] #Sequence const
            
            baselines = [x_baseline, FPKM_baseline, RNA_Annotation_baseline]
            
            (SHAPE_features_ig_scores, FPKM_ig_scores, RNA_Annotations_ig_scores) = get_integrated_gradients(model=loaded_model, sample_inputs=[x_sample,FPKM_sample,RNA_Annotations_sample], baselines=baselines, num_steps=IG_num_steps, multiple_samples=True)
            #%%
            Seq_SHAPE_features_ig_scores_combined = seq_features_ig_scores + SHAPE_features_ig_scores

            #-----------------------  Seq IG  -----------------------

            ENST_id_split = ENST_id[sample_idx].split('|')
            Fig_Title = '{} {}   {}: {}-{}   Binding score: {:.3f}'.format(Protein,Cell_Type,ENST_id_split[0],ENST_id_split[1],ENST_id_split[2],y_pred[0][0])
            create_DNA_logo(Seq_SHAPE_features_ig_scores_combined[0,:,:4], Protein, Cell_Type, Title= Fig_Title, path_plot=output_dir+'Plots_Sequence/{}/'.format(sample_idx), figsize=(30, figure_height+0.5),show_score = False, show_only_y = False, show_Title=True)
            
            #----------------------- SHAPE IG -----------------------

            str_PWM = np.zeros((Sequence_len,2))
            for i in range(len(str_PWM)):
                if x[0,:,4][i] < 0.233:
                    str_PWM[i,0] = Seq_SHAPE_features_ig_scores_combined[0,:,4][i]
                else:
                    str_PWM[i,1] = Seq_SHAPE_features_ig_scores_combined[0,:,4][i]
                    
            create_structure_logo(str_PWM, Protein, Cell_Type, Title=Title+'/{}'.format(sample_idx), path_plot=output_dir+'Plots_Structure/{}/'.format(sample_idx), figsize=(30, figure_height-0.5),show_score = False, show_only_y = False)

            #----------------------- SHAPE Vals -----------------------
            create_SHAPE_plot(x[0,:,4], Protein, Cell_Type, path_plot=output_dir+'Plots_SHAPE/{}/'.format(sample_idx), figsize=(30, figure_height-0.5),show_score = False, show_only_y=False)


        
            img_seq = cv2.imread(output_dir+'Plots_Sequence/{}/{}_{}.png'.format(sample_idx,Protein,Cell_Type))
            img_str = cv2.imread(output_dir+'Plots_Structure/{}/{}_{}.png'.format(sample_idx,Protein,Cell_Type))
            img_SHAPE = cv2.imread(output_dir+'Plots_SHAPE/{}/{}_{}.png'.format(sample_idx,Protein,Cell_Type))
            
            img_str_resize = cv2.resize(img_str,(img_seq.shape[1],img_str.shape[0]))
            img_SHAPE_resize = cv2.resize(img_SHAPE,(img_seq.shape[1],img_SHAPE.shape[0]))

            img_v = cv2.vconcat([img_seq, img_str_resize, img_SHAPE_resize])
                
            fig_name = '{}_{}_{}_{:.4f}_{}_{}_{}'.format(Protein,Cell_Type,sample_idx,y_pred[0][0],ENST_id_split[0],ENST_id_split[1],ENST_id_split[2])
            cv2.imwrite(output_dir+'Plots_Combined/{}.png'.format(fig_name),img_v)


    
#%%
def IG_RNA_Annotations_Get_AVG (model_path,data_path,batch_size=200,num_steps_IG=25,pred_score_thr=0.9,RNA_annotations_baseline = 0):

    #Get List of Proteins
    values_list = []
    protein_list = []
    cell_type_list = []
    labels_list = []
    # window_size = 200 #Max num of IG to calculate at one without the PC crashing

    Protein = data_path.split('/')[-2].split('.')[0].split('_')[0]
    Cell_Type = data_path.split('/')[-2].split('.')[0].split('_')[1]
    print('Protein: \t\t\t{} \nCell Type: \t\t\t{}'.format(Protein,Cell_Type))
    
    names = ['ENST_Indices','Sequence','SHAPE','eCLIP_Score','FPKM','ENST','ENST_Start','ENST_End','chr','chr_start','chr_end','strand','RNA_annotations']
    df_data = pd.read_csv(data_path,delimiter='\t',names=names, skiprows=1)
       
    x,y,fpkm,RNA_annotations,label_binarizer,_ = preprocess(Protein,Cell_Type,df_data)
    
    #One hot encoding RNA region types
    RNA_annotations_all_opts = np.zeros((8,8))
    for j in range(len(RNA_annotations_all_opts)):
        RNA_annotations_all_opts[j,j] = 1
    RNA_annotations_labels = inverseLabelBinarizerTransform(RNA_annotations_all_opts,label_binarizer)
    RNA_annotations_labels = RNA_annotations_labels.reshape(RNA_annotations_labels.shape[0],1)

    sum_attributions = np.zeros((1,8))
    
    #Load model
    loaded_model = tf.keras.models.load_model(model_path)
    y_predicted = loaded_model.predict([x,fpkm,RNA_annotations], batch_size=5000)
    
    start_time = time.time()

    #Filter samples with prediction score above threshold
    y_predicted = y_predicted[:,0]
    x = x[y_predicted >pred_score_thr]
    fpkm = fpkm[y_predicted >pred_score_thr]
    RNA_annotations = RNA_annotations[y_predicted>pred_score_thr]
    
    #Get RNA region types attribution scores from all samples
    for sample_idx in range((len(x)//batch_size)+1):
        idx = sample_idx*batch_size

        if idx % 1000 == 0:
            if sample_idx:
                ETA(start_time,idx/len(x))          
        
        if (idx + batch_size) < len(x):  
            x_i = x[idx:idx+batch_size]
            FPKM_i = fpkm[idx:idx+batch_size]
            RNA_Annotations_i = RNA_annotations[idx:idx+batch_size]
            print('Samples: [{}:{}]'.format(idx,idx+batch_size))
        else:
            x_i = x[idx:]
            FPKM_i = fpkm[idx:]
            RNA_Annotations_i = RNA_annotations[idx:]
            print('Samples: [{}:{}]'.format(idx,idx+x_i.shape[0]))

        x_baseline =  x_i #Keep SHAPE and sequence const
        FPKM_baseline = FPKM_i #Keep FPKM const
        RNA_Annotation_baseline = np.ones(RNA_Annotations_i.shape[1:])*RNA_annotations_baseline 

        baselines = [x_baseline, FPKM_baseline, RNA_Annotation_baseline]
        
        (seq_features_ig_scores, FPKM_ig_scores, RNA_Annotations_ig_scores) = get_integrated_gradients(model=loaded_model, sample_inputs=[x_i,FPKM_i,RNA_Annotations_i], baselines=baselines, num_steps=num_steps_IG, multiple_samples=True)
        sum_attributions += sum(RNA_Annotations_ig_scores)
    
    #%%
    labels = []
    values = []
    for j in range(len(RNA_annotations_labels)):
        if RNA_annotations_labels[j][0] == 'five_prime_utr':
            labels.append('5\' UTR')
            values.append(sum_attributions[0][j]/len(x))
            
        elif RNA_annotations_labels[j][0] == 'three_prime_utr':
            labels.append('3\' UTR')
            values.append(sum_attributions[0][j]/len(x))
            
        elif RNA_annotations_labels[j][0] == 'exon':
            labels.append('Exon')
            values.append(sum_attributions[0][j]/len(x))
            
        elif RNA_annotations_labels[j][0] == 'CDS':
            labels.append('CDS')
            values.append(sum_attributions[0][j]/len(x))   
    
    values_list.append(values)
    protein_list.append(Protein)
    cell_type_list.append(Cell_Type)
    labels_list.append(labels)
    
    Protein_cell_type = np.array([protein_list,cell_type_list]).T
    
    df_IG = pd.DataFrame(np.concatenate((Protein_cell_type,np.array(values_list)),axis=1),
                         columns = ['Protein','Cell_Type',labels_list[0][0],labels_list[0][1],labels_list[0][2],labels_list[0][3]])
    
    df_IG.to_csv('Average_IG_RNA_Region_Types_{}_{}.csv'.format(Protein,Cell_Type), index=False, header = False)


#%% Functions caller
if LOCAL_INTERPRETATION:
    print('\n------ Creating Sequence and Structure Local Interpretations ------\n')
    print('Input data path: \t\t\t{} \nOutput data directory: \t\t\t{} \nPredictive model directory: \t\t{} \nicSHAPE model directory: \t\t{}'\
          .format(input_data_path,output_data_dir,model_data_dir,icSHAPE_model_dir))
        
    model_path = model_data_dir +'Best_Model.h5'
    IG_Seq_Str(data_path=input_data_path,path_Model=model_path,output_dir=output_data_dir,icSHAPE_model_dir = icSHAPE_model_dir,pred_score_thr=args.pred_score_thr\
               ,IG_num_steps = args.IG_num_steps, seq_baseline=args.seq_baseline,str_baseline=args.str_baseline)     
    
    
    
