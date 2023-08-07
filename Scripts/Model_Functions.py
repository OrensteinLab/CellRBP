# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 10:40:02 2021

@author: Ori Feldman
"""
#%%
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Concatenate, Input, Dropout
from tensorflow.keras.callbacks import Callback
from sklearn import metrics
import argparse

from Util_Functions import preprocess, create_dir,string_to_bool

#%%
#PARAMETERS
parser = argparse.ArgumentParser('CellRBP')

parser.add_argument('--TRAIN',                      type=str,   default='False',      help='Train a model [True,False]. Default: False')
parser.add_argument('--EVALUATE',                   type=str,   default='False',      help='Evaluate using an existing model [True,False]. Default: False')
parser.add_argument('--PREDICT',                    type=str,   default='False',      help='Predict using an existing model [True,False]. Default: False')


parser.add_argument('--input_data_dir',             type=str,   default="../Data/clip_data_processed/AARS_K562/",  help='Input data directory path. For example: ../Data/clip_data_fpkm_chr_positions_RNA_annotations/AARS_K562/')
parser.add_argument('--output_data_dir',            type=str,   default="../Data/clip_data_processed/AARS_K562/",  help='Output Data directory path. For example: ../Data/clip_data_fpkm_chr_positions_RNA_annotations/AARS_K562/')
parser.add_argument('--output_model_dir',           type=str,   default="../Models/",                                                      help='Model\'s output directory path. Default: ../Models/')
parser.add_argument('--output_model_type',          type=str,   default="CellRBP",                                                         help='Output Model\'s name. Default: CellRBP')
parser.add_argument('--icSHAPE_model_dir',          type=str,   default='../Models/icSHAPE/',                                              help='icSHAPE model\'s directory path. Default: ../Models/icSHAPE/') 
 
parser.add_argument('--train_set_file_name',        type=str,   default="Train_70.tsv",         help='Input train set file name. Default: Train_70.tsv')
parser.add_argument('--valid_set_file_name',        type=str,   default="Validation_15.tsv",    help='Input validation set file name. Default: Validation_15.tsv')
parser.add_argument('--test_set_file_name',         type=str,   default="Test_15.tsv",          help='Input test set file name. Default: Test_15.tsv')


#Training parameters
parser.add_argument('--LEARNING_RATE',              type=float, default="0.000693433",          help='Learning rate. Default: 0.000693433')
parser.add_argument('--DECAY_RATE',                 type=float, default="0.000334327",          help='Decay rate. Default: 0.000334327')
parser.add_argument('--BATCH_SIZE',                 type=int,   default="32",                   help='Batch size. Default: 32 ')
parser.add_argument('--MAX_EPOCHS',                 type=int,   default="50",                   help='Max epochs. Default: 50')
parser.add_argument('--N_KERNELS_1',                type=int,   default="1024",                 help='Number of kernels in the first set. Default: 1024')
parser.add_argument('--KERNEL_SIZE_1',              type=int,   default='13',                   help='Size of a kernel in the first set. Default: 13')
parser.add_argument('--DROP_OUT_CNN_1',             type=float, default="0.3",                  help='Drop out rate for the first set of kernels. Default: 0.3')
parser.add_argument('--N_KERNELS_2',                type=int,   default="512",                  help='Number of Kernels in the second set. Default: 512')
parser.add_argument('--KERNEL_SIZE_2',              type=int,   default="3",                    help='Size of a kernel in the second set. Default: 3')
parser.add_argument('--DROP_OUT_CNN_2',             type=float, default="0.0",                  help='Drop out rate for the second set of kernels. Default: 0.0')
parser.add_argument('--DENSE_FILTERS_1',            type=int,   default="512",                  help='Number of neurons in the fully connected layer. Default: 512')
parser.add_argument('--LOSS',                       type=str,   default="mse",                  help='Loss function [mse, bce]. Default: mse')
parser.add_argument('--FINAL_ACTIVATION_FUNCTION',  type=str,   default="linear",               help='Final layer\'s activation function [linear, sigmoid]. Default: linear')
parser.add_argument('--VERBOSE',                    type=int,   default ='0',                   help='Model verbose [0,1,2]. Default: 0')

parser.add_argument('--predict_model_dir',          type=str,                                   help='Prediction model directory path. Example: ../Models/CellRBP/AARS/K562/')

args = parser.parse_args()

# [TRAIN,EVALUATE,predict_model_dir,data_dir,output_model_dir,ModelType,icSHAPE_model_dir,train_fn,valid_fn,test_fn] = \
#     [args.train,args.evaluate,args.predict_model_dir,args.data_dir,args.output_model_dir,args.output_model_type,args.icSHAPE_model_dir,args.train_set_file_name,args.valid_set_file_name,args.test_set_file_name]

# [LEARNING_RATE,DECAY,MAX_EPOCHS,BATCH_SIZE,N_KERNELS_1,KERNEL_SIZE_1,DROP_OUT_CNN_1,N_KERNELS_2,KERNEL_SIZE_2,DROP_OUT_CNN_2,DENSE_FILTERS_1,LOSS,FINAL_ACTIVATION_FUNCTION,VERBOSE] = \
#     [args.LEARNING_RATE,args.DECAY_RATE,args.MAX_EPOCHS,args.BATCH_SIZE,args.N_KERNELS_1,args.KERNEL_SIZE_1,args.DROP_OUT_CNN_1, \
#      args.N_KERNELS_2,args.KERNEL_SIZE_2,args.DROP_OUT_CNN_2,args.DENSE_FILTERS_1,args.LOSS,args.FINAL_ACTIVATION_FUNCTION,args.VERBOSE]

args.TRAIN = string_to_bool(args.TRAIN)
args.EVALUATE = string_to_bool(args.EVALUATE)
args.PREDICT = string_to_bool(args.PREDICT)
args.input_data_dir = args.input_data_dir.replace('\\','/')

length_of_seq=101
input_Width = 5

if args.TRAIN:
    print('\nData Directory: \t\t{}\nOutput model directory: \t{}\nModel type: \t\t\t{}\nicSHAPE model directory: \t{}'\
          .format(args.input_data_dir,args.output_model_dir,args.output_model_type,args.icSHAPE_model_dir))
else:
    print('Data Directory: \t\t{}\nicSHAPE model directory: \t{}'.format(args.input_data_dir,args.icSHAPE_model_dir))

print('Train set filename: \t\t{} \nValidation set filename: \t{} \nTest set filename: \t\t{}'.format(args.train_set_file_name,args.valid_set_file_name,args.test_set_file_name))

Protein_Cell_Type = (args.input_data_dir.split('/'))[-2]   
Cell_Type = Protein_Cell_Type.split('_')[1]
Protein = Protein_Cell_Type.split('_')[0]

print('Protein: \t\t\t{}\nCell Type: \t\t\t{}'.format(Protein,Cell_Type))
if (Cell_Type == 'K562' ) or (Cell_Type == 'HepG2'):
    if args.TRAIN:
        print('\n------ Hyperparamters ------ \nLearning Rate: \t\t\t\t{:.5f} \nDecay: \t\t\t\t\t{:.5f} \nMax Epochs: \t\t\t\t{} \nBatch Size: \t\t\t\t{} \nNo. Kernels (1): \t\t\t{} '\
              .format(args.LEARNING_RATE,args.DECAY_RATE,args.MAX_EPOCHS,args.BATCH_SIZE,args.N_KERNELS_1))
        print('Kernel Size (1): \t\t\t{} \nDropout CNN (1): \t\t\t{} \nNo. Kernels (2): \t\t\t{} \nKernel Size (2): \t\t\t{} \nDropout CNN (2): \t\t\t{} \n Dense Filters: \t\t\t{}'\
              .format(args.KERNEL_SIZE_1,args.DROP_OUT_CNN_1,args.N_KERNELS_2,args.KERNEL_SIZE_2,args.DROP_OUT_CNN_2,args.DENSE_FILTERS_1))
        print('Loss function: \t\t\t\t{}\nFinal layer\'s activation function: \t{}\nVerbose: \t\t\t\t{}'.format(args.LOSS,args.FINAL_ACTIVATION_FUNCTION,args.VERBOSE))
        
        if args.LOSS == 'bce':
            args.LOSS = tf.keras.losses.BinaryCrossentropy();
    
        if Cell_Type == 'K562' or Cell_Type=='HepG2':    
            print('Protein: \t\t\t\t{}\nCell Type: \t\t\t\t{}'.format(Protein,Cell_Type))
            path_Model = args.output_model_dir + '/{}/{}/{}'.format(args.output_model_type,Protein,Cell_Type)
            create_dir(path_Model)
        
            names = ['ENST_Indices','Sequence','SHAPE','eCLIP_Score','FPKM','ENST','ENST_Start','ENST_End','chr','chr_start','chr_end','strand','RNA_annotations']
            df_eclip_train = pd.read_csv(args.input_data_dir + args.train_set_file_name, delimiter='\t',names=names, skiprows=1)
            df_eclip_val = pd.read_csv(args.input_data_dir + args.valid_set_file_name, delimiter='\t',names=names, skiprows=1)
        
            x_train,y_train,fpkm_train,RNA_annotations_train,_,_ = preprocess(Protein,Cell_Type,df_eclip_train,args.icSHAPE_model_dir)
            x_valid,y_valid,fpkm_valid,RNA_annotations_valid,_,_ = preprocess(Protein,Cell_Type,df_eclip_val,args.icSHAPE_model_dir)
        
            #---------------- Build Model ----------------
            
            print('------ Building Model ------')
            input_Seq_icSHAPE = Input(shape=(length_of_seq,input_Width),name='input_Seq_icSHAPE')    #All the rest
            input_FPKM = Input(shape=(1,),name='input_FPKM') #FPKM
            input_RNA_annotations = Input(shape=(RNA_annotations_train.shape[1],),name='input_RNA_annotations') #RNA Annotations
        
            A2 = Conv1D(args.N_KERNELS_1, kernel_size = args.KERNEL_SIZE_1, activation='relu', input_shape=(length_of_seq,input_Width), use_bias=True, name='A2')(input_Seq_icSHAPE)
            A3 = MaxPooling1D(pool_size=length_of_seq-args.KERNEL_SIZE_1+1, name='A3')(A2)
            A4 = Flatten()(A3)
            A5 = Dropout(args.DROP_OUT_CNN_1)(A4)
            
            B2 = Conv1D(args.N_KERNELS_2, kernel_size = args.KERNEL_SIZE_2, activation='relu', input_shape=(length_of_seq,input_Width), use_bias=True, name='B2')(input_Seq_icSHAPE)
            B3 = MaxPooling1D(pool_size=length_of_seq-args.KERNEL_SIZE_2+1, name='B3')(B2)
            B4 = Flatten()(B3)
            B5 = Dropout(args.DROP_OUT_CNN_2)(B4)
            
            C = Concatenate(axis=1)([A5, B5, input_FPKM, input_RNA_annotations])
            M1 = Dense(args.DENSE_FILTERS_1, activation='relu', name = 'M1')(C)
            M2 = Dense(1, activation='linear', name = 'M2')(M1)
            
            finalModel = Model(inputs=[input_Seq_icSHAPE, input_FPKM, input_RNA_annotations], outputs=M2)
            finalModel.summary()
            
            #Compile      
            opt = tf.keras.optimizers.legacy.Adam(learning_rate = args.LEARNING_RATE, decay = args.DECAY_RATE)
            finalModel.compile(loss = args.LOSS, optimizer=opt)
            
            max_auc = 0
            
            class IntervalEvaluation(Callback):
                def __init__(self, model_dir,ModelType,Protein,Cell_Type, validation_data=(), max_epochs = args.MAX_EPOCHS, interval=10,max_auc=0,best_epoch=1):
                    super(Callback, self).__init__()
            
                    self.interval = interval
                    self.max_auc = max_auc
                    self.X_val, self.y_val = validation_data
                    self.best_epoch = best_epoch
                    self.Protein = Protein
                    self.Cell_Type = Cell_Type
                    self.ModelType = ModelType
                    self.model_dir = model_dir
                    self.max_epochs = max_epochs
                    
                def on_epoch_end(self, epoch, logs={}):
                    if epoch % self.interval == 0:
                        y_pred = self.model.predict(self.X_val, verbose=0)
                        score = metrics.roc_auc_score(self.y_val, y_pred)
                        self.best_epoch = epoch
                        
                        if score>self.max_auc:
                            self.max_auc = score    
                            print('Epoch No. [{}/{}]: \tValidation AUC: {:.4f} \tNew high AUC score'.format(self.best_epoch+1,self.max_epochs,score))
                            self.model.save('{}/{}/{}/{}/Best_Model.h5'.format(self.model_dir,self.ModelType,self.Protein,self.Cell_Type))
                        else:
                            print('Epoch No. [{}/{}]: \tValidation AUC: {:.4f} \tNo Improvement'.format(self.best_epoch+1,self.max_epochs,score))
                
                def get_auc(self):
                    return [self.max_auc, self.best_epoch]
                        
            ival  = IntervalEvaluation(args.output_model_dir,args.output_model_type,Protein,Cell_Type,validation_data=([x_valid, fpkm_valid, RNA_annotations_valid], y_valid), \
                                       max_epochs = args.MAX_EPOCHS, interval=1, max_auc=0, best_epoch = 1)
        
            print('\n------ Training Model: {} {} ------'.format(Protein,Cell_Type))
            finalModel.fit([x_train,fpkm_train,RNA_annotations_train], y_train, epochs = args.MAX_EPOCHS, batch_size = args.BATCH_SIZE, callbacks=[ival], verbose = args.VERBOSE)
        
            [max_auc,best_epoch] = ival.get_auc()
            print('Highest AUC: {:.4f}'.format(max_auc))
            
        else:
            print('The model currently supports only HepG2 and K562 cell types')
        
    if args.EVALUATE or args.PREDICT:
        if args.predict_model_dir != None:
            args.predict_model_dir = args.predict_model_dir.replace('\\','/')

            model_protein = args.predict_model_dir.split('/')[-3]
            model_cell_type = args.predict_model_dir.split('/')[-2]
            print('\n------ Data prediction ------')
            print('Data protein: \t\t\t{} \nData cell type: \t\t{}'.format(Protein,Cell_Type))
            print('Model protein: \t\t\t{} \nModel cell type: \t\t{}'.format(model_protein,model_cell_type))
            if model_protein == Protein:
                if (model_cell_type == 'K562') or (model_cell_type=='HepG2'):    
                    df_eclip_test = pd.read_csv(args.input_data_dir + args.test_set_file_name, delimiter='\t')
                    x_test,y_test,fpkm_test,RNA_annotations_test,_,_ = preprocess(Protein,Cell_Type,df_eclip_test,args.icSHAPE_model_dir)
                    loaded_model = load_model(args.predict_model_dir + '/Best_Model.h5')
                    y_pred = loaded_model.predict([x_test, fpkm_test, RNA_annotations_test], verbose=0)
                    if args.EVALUATE:
                        score = metrics.roc_auc_score(y_test, y_pred)
                        print("AUC score for {} {} using model {} {}: \t\t{}".format(Protein,Cell_Type,model_protein,model_cell_type,score))
        
                    if args.PREDICT:
                        if 'eCLIP_Score' in df_eclip_test:      #Move the real eCLIP scores to the end if they exist
                            real_eCLIP_scores = df_eclip_test['eCLIP_Score']
                            df_eclip_test = df_eclip_test.drop(columns=['eCLIP_Score'])
                            df_eclip_test['eCLIP_Score'] = real_eCLIP_scores
                            
                        df_eclip_test['Predicted_eCLIP_Score'] = y_pred
                        df_eclip_test.to_csv(args.output_data_dir+'/{}_{}_Test_predicted_eCLIP_Scores.tsv'.format(Protein,Cell_Type),sep='\t',index=False)
                else:
                    print('Error: Wrong Cell Type. The model currently supports only HepG2 and K562 cell types')
            else:
                print('Error: The model\'s protein does not match the data\'s protein')
        else:
            print('Error: Please provide a path for the predictive model. \nUtilize the --predict_model_dir to specify the location of the predictive model')
else:        
    print('Error: Wrong Cell Type. CellRBP currently supports HepG2 and K562 cell types only')
        
    
