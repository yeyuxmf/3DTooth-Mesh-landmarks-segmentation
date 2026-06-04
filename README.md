# Introduction
Tooth Segmentation and Landmark Localization in 3D Dental Mesh，Pytorch
1. "The first category relies on segmentation, localization, or detection techniques to isolate local tooth regions, followed by localized landmark localization or fine-grained segmentation within these individual areas."

2. "The second category directly performs global tooth landmark localization on the dental data, achieving end-to-end predictions without relying on any prior segmentation or detection stages."


# 一、Train 
two_stage:
python  main_seg_landmarks.py

one_stage:
paper 1 train：
python ./main.py

paper 2 train：
python ./main_cls.py    A pre-trained model is obtained after training.
python ./main_reg.py    Load the pre-trained model obtained from training with main_cls.py, and then train it further to obtain the final model.

# Environment

# Model structure
























# 📄 License & Commercial Use Restriction
This copy of the code is intended strictly for academic research purposes only.

Academic Use: You are free to modify and use this code for academic research, benchmarking, and paper publications. Please cite our paper if you find this work useful.

Commercial Use: Any commercial usage (including but not limited to integrating into commercial orthodontic/dental software, cloud services, or profit-making products) is STRICTLY PROHIBITED without prior written enforcement/licensing from the authors.
