# CellRBP

## Requirements

Python 3.9.15
TensorFlow version: 2.11.0
Pandas version: 1.5.2
NumPy version: 1.24.1
Matplotlib 3.6.3
Scikit-learn 1.2.0
logomaker

## Installation

Install packages and clone repository :

```bash
pip install -r requirements.txt

cd CellRBP
git clone https://github.com/kuixu/PrismNet.git
tar zxvf ./Features/df_transcripts.pkl.gz
tar zxvf ./Features/GTF/Homo_sapiens.GRCh38.89.gtf.gz
```

## Preprocess eCLIP data 

add RNA abundance,RNA region types and predict missing icSHAPE values
```
cd Scripts
python Process_Data.py --process_eclip True --input_data_path ../Data/clip_data/RBFOX2_HepG2.tsv
```

# Training and evaluation
```
python Model_Functions.py --TRAIN True --input_data_dir ../Data/clip_data_processed/RBFOX2_HepG2/
python Model_Functions.py --PREDICT True --predict_data_path ../Data/clip_data_processed/RBFOX2_HepG2/Test_15.tsv --predict_model_dir ../Models/CellRBP/RBFOX2/HepG2/
python Model_Functions.py --EVALUATE True --input_data_dir ../Data/clip_data_processed/RBFOX2_HepG2/ --predict_model_dir ../Models/CellRBP/RBFOX2/HepG2/
```






