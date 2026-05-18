
#point_order = ["MCP","DCP","CEP","FA","IEP","DeCP","BCP","LCP","MBCP","DBCP","MLCP","DLCP","CFP","WALA"] #private data
# "MCP", "DCP", "FA", "CEP","IEP"
# "MCP", "DCP", "FA", "DeCP"
# "MCP", "DCP", "FA", "CFP", "BCP", "LCP"
# "MCP", "DCP", "FA", "CFP", "MLCP", "DLCP", "MBCP", "DBCP"

fxiedPorder = ['Mesial', 'Distal', 'InnerPoint', 'OuterPoint', 'FacialPoint']

matchPorder = ['Cusp']


#tooth_landmark_nums = [8, 8, 8, 6, 6, 4, 5, 5, 5, 5, 4, 6, 6, 8, 8, 8]  #private data
fxid_tnums = 5   #teethMICCAI2022 data
match_tnums = 6

maxlandnums = 11
tooth_nums = 16
sam_points = 1024#256\512\1024\2048\4096
Tpoints = 512
knn_nums = 32#32\48\64
neark = 20
max_remove_nums = 4
Angles = [i for i in range(0, 16, 1)]
dAngles = [i for i in range(0, 30, 1)]

LOSSNUMS =20





leaky_relu_slope = 0.2
dropout_rate = 0.1

noise_scale = 1.0
prediction_dim = 7
grad_clip_norm = 10.0



