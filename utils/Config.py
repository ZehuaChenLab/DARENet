# Model and Dataset
LOSS = 'dice_bce_loss'  # dice_bce_loss  # gf_loss
MODEL_NAME = 'MobileRoadNet'  # Baseline # Baseline_Mixer_Dskip
                # UNet # DLinkNet # NLlinkNet # DeepLabv3_plus # CARNet # RCFSNet # CMLFormer  #FuseExperiment  #CMTFNet #CARENet #L2KDNet
                # MobileRoadNet # LID2Mamba # LID2MambaMobile #Samba # ConvRoad
DATASET = "./dataset/ROAD"  # ./dataset/ROAD # ./dataset/Mas / CHN6-CUG /spacenet
WEIGHT = ''
# Resume
RESUME = False # 断点恢复训练
LOSS_UPDATE = False  # 加载训练后权重时，是否还原对应损失
# Test
THRESHOLD =None# None寻找最佳阈值
# Train
TOTAL_EPOCH = 500
BATCHSIZE_PER_CARD = 4
# Loss
LOSS_GAP = 0.01
INIT_MID_LOSS = 0.1
# Optimizer and Scheduler
SCHEDULER = 'Plateau'  # 'Cosine'  # 'Plateau'
WARMUP_T = 5
PATIENCE_T = 5
DECAY_RATE = 0.25
NUM_EARLY_STOP = 8
WEIGHT_DECAY = 1e-2
COOLDOWN_EPOCH = 150
WARMUP_LR_INIT = 1e-6
MIN_LEARNING_RATE = 5e-7
INIT_LEARNING_RATE = 2e-4

# Email
SENDER_HOST = "smtp.qq.com" # 邮箱服务器
SENDER = "1097550996@qq.com"
SENDER_PW = "guwulnvuooypicdd"  # 授权码
RECEIVER = "1097550996@qq.com"

# SENDER_HOST = "smtp.163.com" # 邮箱服务器
# SENDER = "limingzhe111341@163.com"
# SENDER_PW = "FAbqyzEkwUmw3dYH"  # 授权码
# RECEIVER = "limingzhe111341@163.com"
