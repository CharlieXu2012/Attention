# Subjective Annotations for Vision-Based Attention Level Estimation

# Pre-requisites:
  - Python 3.5.4
  - CUDA 9.0
  - CudNN 7
  - Keras 2.1.6
  - Tensorflow 1.8.0

# Installation:
1) Install all pre-requisites.
2) Clone the repository.
3) Run the default OpenPose keypoint extraction (JSON Output with No Visualization https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/demo_overview.md) on the selected Pandora RGB images. The list of selected image names can be found in ./PANDORA_ATTENTION_LABELS.xlsx.

IF you want to run depth extraction:

4) Copy and extract the PANDORA depth image pairs, and run ./padcrop.py with output folder ./PANDORA_depth/.

IF not:

4) Move the file from ./PANDORA_depth/Pre/ to ./PANDORA_features/

# To run feature generation:
$python main_feature_parse --dataset 1 --depth False --oversampling False --method 2

Arguments: 

  --dataset: Wether to use PANDORA (0) or GRANADE (1)., type=int, default=0
  
  --depth: Wether to perform depth feature extraction or load depth .npy array., type=bool, default=False
  
  --oversampling: Wether or not to perform oversampling of minority clases., type=bool, default=False
  
  --method: Method for oversampling: (1)None (2)SMOTE (3)ADASYN., type=int, default=2
  
 Outputs:
 
  - Training split (geometric features, depth, labels) in ./PANDORA_features/
  
  - Validation split (geometric features, depth, labels) in ./PANDORA_features/
  
 # To train the model:
 $python main_train_model.py --oversample False --model 3 --fusiontype 1 --type 0 --bs 32 --ep 175
 
 Arguments:
 
  --oversample: Wether or not to use oversampled data for training., type=bool, default=False
  
  --model: Two (2) or three (3) stream DNN model.,  type=int, default=3
  
  --fusiontype: Use early (0), fully connected (1) or late (2) fusion., type=int, default=1
  
  --type: Use average (0), max (1) or WSLF-LW (2) fusion., type=int, default=0

  --bs: Batch size., type=int, default=32

  --ep: Epochs., type=int, default=175
  
  
 Outputs:
 
   - Confusion matrices
   
   - Trained model in ./models/
