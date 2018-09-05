import logging
import sys
import tensorflow as tf
import datetime

#  GPU
GPU_WHEN_TRAINING = "4"

#  Model Environment
DIMENSIONALITY = 2
# 1 PHPPHPHHHPHHPHHHHH -9
# 2 HHPPHPPHPPHPPHPPHPPHPPHH  -9
# 3 HHPHPHPHPHHHHPHPPPHPPPHPPPPHPPPHPPPHPHHHHPHPHPHPHH  -21 -30
# 4 PHHPHHHPHHHHPPHHHPPPPPPHPHHPPHHPHPPPPHHPHPHPHHPPP       -31
# 5 HPHHPPHHHHPHHHPPHHPPHPHHHPHPHHPPHHPPPHPPPPPPPPHH        -30
# 6 PHPHHPHHHHHHPPHPHPPHPHHPHPHPPPHPPHHPPHHPPHPHPPHP        -31
# 7 PHPHHPPHPHHHPPHHPHHPPPHHHHHPPHPHHPHPHPPPPHPPHPHP        -30
# 8 PPHPPPHPHHHHPPHHHHPHHPHHHPPHPHPHPPHPPPPPPHHPHHPH        -30
# 9 PHHPPPPPPHHPPPHHHPHPPHPHHPPHPPHPPHHPPHHHHHHHPPHH        -30
import global_variables
SEQ_NAME = "SEQ_3"
HP_SEQ = global_variables.SEQ[SEQ_NAME]
print(len(HP_SEQ))
SEQ_LENGTH = len(HP_SEQ)


#  Network
if DIMENSIONALITY == 2:
    VECTOR_LENGTH = 3
elif DIMENSIONALITY == 3:
    VECTOR_LENGTH = 4

N_FILTER = 64
L2_REG = 0.0001
N_BLOCKS = 7

#  MCTS
C_PUCT = 15
DIRICHLET = 1.5
EPSILON = 0.25

#  Training
LEARNING_RATE = 1e-3
TEMP = 0.0001
N_PLAYOUT = 6
BUFFER_SIZE = 1000*SEQ_LENGTH
BATCH_SIZE = 23
KL_TARG = 0.0002
GAME_BATCH_NUM = 2500
EPOCH = 1
N_POLICY_UPDATE = 5


#  Session
SESS_CONFIG = tf.ConfigProto()
SESS_CONFIG.gpu_options.allow_growth = True

#  Path
# WORK_PATH = "/nfshome/hangyu/HP-model_uct/"
WORK_PATH = "./"
MODEL_NAME = "ResNet_"+SEQ_NAME+"_"+str(C_PUCT)+"_"+str(N_PLAYOUT)+str(datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))
MODEL_PATH = WORK_PATH + 'models/' + MODEL_NAME + "/"
