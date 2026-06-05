# Introduction
Tooth Segmentation and Landmark Localization in 3D Dental Mesh，Pytorch
1. "The first category relies on segmentation, localization, or detection techniques to isolate local tooth regions, followed by localized landmark localization or fine-grained segmentation within these individual areas."

2. "The second category directly performs global tooth landmark localization on the dental data, achieving end-to-end predictions without relying on any prior segmentation or detection stages."


# Dataset  3DTeethSeg22 and Teeth3DS

1. Large-Scale Segmentation Benchmark: Following the official split protocol of 3DTeethSeg'22 [37], the 1,800 cases are divided into 1,200 training samples and 600 independent test samples. This split is employed to validate the model's performance on the standard tooth segmentation task.

2. Fine-Grained Landmark Subset: This study further incorporates the annotation information from the 3DTeethLand [45] 2024 Challenge. This subset comprises two groups: The first group (Training Set) contains 240 scan cases derived from the Teeth3DS dataset, supplemented with six categories of landmark labels by professional dentists, serving as the training set for the landmark localization task. The second group (Hidden Private Test Set) contains 100 cases. This portion is independent of the parent dataset, and we obtained this data to test the algorithm proposed in this paper.

# Model 

   Model architecture diagram. To be added after Papers 1–3 are accepted.

   
# Train 
two_stage: 

Use the trained tooth detection model to crop local regions of individual teeth from both the training and test sets, which are then fed to the tooth landmark regression model for training.
"牙齿分割20250425.docx" This is the original manuscript for tooth detection and segmentation, which has been accepted and published.  The mesh_tooth_crop.py script cannot be run directly. Due to historical reasons, this paper was not open-sourced, and the code is too messy to clean up now.


python ./data/data_processing.py  #process data:

python  ./main_seg_landmarks.py   

one_stage:

Obtain the training data for model training in Paper 2 and Paper 3.

python ./data/data_process.py

paper 1 train：

python ./main.py

paper 2 train：

python ./main_cls.py    #A pre-trained model is obtained after training.

python ./main_reg.py    #Load the pre-trained model obtained from training with main_cls.py, and then train it further to obtain the final model.

# Environment

# Model structure
























# 📄 License & Commercial Use Restriction
This copy of the code is intended strictly for academic research purposes only.

Academic Use: You are free to modify and use this code for academic research, benchmarking, and paper publications. Please cite our paper if you find this work useful.

Commercial Use: Any commercial usage (including but not limited to integrating into commercial orthodontic/dental software, cloud services, or profit-making products) is STRICTLY PROHIBITED without prior written enforcement/licensing from the authors.
