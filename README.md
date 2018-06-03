# masters-repo

# Pre-requisites:
  - Python 3.5.4
  - CUDA 9.0
  - CudNN 7
  - Keras 2.1.6
  - Tensorflow 1.8.0

# Installation:
1) Install all pre-requisites.
2) Clone the repository.
3) Copy the PANDORA .json keypoint files from the server, located in /srv/data/shared/PANDORA/KEYPOINTS/ to ./PANDORA_keypoints/

IF you want to run depth extraction:

4) Copy the PANDORA depth images from the server, located in /srv/data/shared/PANDORA/DEPTH/ to ./PANDORA_depth/

IF not:

4) Move the file from ./PANDORA_depth/Pre/ to ./PANDORA_features/


# To run feature generation:
$python main_feature_parse --dataset 1 --depth False --oversampling False --method 2

Arguments: 

  --dataset: Wether to use PANDORA (0) or GRANADE (1)., type=int, default=0
  
  --depth: Wether to perform depth feature extraction or load dataset., type=bool, default=False
  
  --oversampling: Wether or not to perform oversampling of minority clases., type=bool, default=False
  
  --method: Method for oversampling: (1)None (2)SMOTE (3)ADASYN., type=int, default=2
  
 Outputs:
 
  - Training split (geometric features, depth, labels) in ./PANDORA_features/
  
  - Validation split (geometric features, depth, labels) in ./PANDORA_features/
  
 # To train the model:
 $python main_train_model.py --oversample False --model 3 --fusiontype 1 --type 0
 
 Arguments:
 
  --oversample: Wether or not to use oversampled data for training., type=bool, default=False
  
  --model: Two (2) or three (3) stream DNN model.,  type=int, default=3
  
  --fusiontype: Use early (0), fully connected (1) or late (2) fusion., type=int, default=1
  
  --type: Use average (0), max (1) or WSLF-LW (2) fusion., type=int, default=0
  
  
 Outputs:
 
   - Confusion matrices
   
   - Trained model in ./models/
