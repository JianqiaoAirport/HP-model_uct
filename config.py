import logging
import sys
import tensorflow as tf
import datetime

#  GPU
GPU_WHEN_TRAINING = "4"

#  Model Environment
DIMENSIONALITY = 2
def string_to_list(str):
    l = []
    for s in str:
        if s == "H":
            l.append(-1)
        elif s == "P":
            l.append(1)
    return l

# HP_SEQ = [-1, -1, -1, 1, -1, 1, 1, -1, -1]

# HHPHPPPHHHPP -3

# PHPPHPHHHPHHPHHHHH -9

# HHPPHPPHPPHPPHPPHPPHPPHH  -9

HP_SEQ = string_to_list("HHPPHPPHPPHPPHPPHPPHPPHH")

ACTION_VECTOR_DICT = {}
INT_ACTION_DICT = {}
ACTION_INT_DICT = {}

if DIMENSIONALITY == 2:
    ACTION_VECTOR_DICT["START_TOKEN"] = (0, 0)
    ACTION_VECTOR_DICT["UP"] = (0, 1)
    ACTION_VECTOR_DICT["DOWN"] = (0, -1)
    ACTION_VECTOR_DICT["LEFT"] = (-1, 0)
    ACTION_VECTOR_DICT["RIGHT"] = (1, 0)

    INT_ACTION_DICT[4] = "START_TOKEN"
    INT_ACTION_DICT[0] = "UP"
    INT_ACTION_DICT[1] = "DOWN"
    INT_ACTION_DICT[2] = "LEFT"
    INT_ACTION_DICT[3] = "RIGHT"

    ACTION_INT_DICT["START_TOKEN"] = 4
    ACTION_INT_DICT["UP"] = 0
    ACTION_INT_DICT["DOWN"] = 1
    ACTION_INT_DICT["LEFT"] = 2
    ACTION_INT_DICT["RIGHT"] = 3
elif DIMENSIONALITY == 3:
    ACTION_VECTOR_DICT["START_TOKEN"] = (0, 0, 0)
    ACTION_VECTOR_DICT["UP"] = (0, 0, 1)
    ACTION_VECTOR_DICT["DOWN"] = (0, 0, -1)
    ACTION_VECTOR_DICT["LEFT"] = (0, -1, 0)
    ACTION_VECTOR_DICT["RIGHT"] = (0, 1, 0)
    ACTION_VECTOR_DICT["FORWARD"] = (1, 0, 0)
    ACTION_VECTOR_DICT["BACKWARD"] = (-1, 0, 0)

    INT_ACTION_DICT[6] = "START_TOKEN"
    INT_ACTION_DICT[0] = "UP"
    INT_ACTION_DICT[1] = "DOWN"
    INT_ACTION_DICT[2] = "LEFT"
    INT_ACTION_DICT[3] = "RIGHT"
    INT_ACTION_DICT[4] = "FORWARD"
    INT_ACTION_DICT[5] = "BACKWARD"

    ACTION_INT_DICT["START_TOKEN"] = 6
    ACTION_INT_DICT["UP"] = 0
    ACTION_INT_DICT["DOWN"] = 1
    ACTION_INT_DICT["LEFT"] = 2
    ACTION_INT_DICT["RIGHT"] = 3
    ACTION_INT_DICT["FORWARD"] = 4
    ACTION_INT_DICT["BACKWARD"] = 5
else:
    print("Dimensionality error")
    logging.error("Dimensionality error")
    sys.exit(1)



SEQ_LENGTH = len(HP_SEQ)

if DIMENSIONALITY == 2:
    VECTOR_LENGTH = 3
elif DIMENSIONALITY == 3:
    VECTOR_LENGTH = 4

# FIRST_LAYER_DIM = 128
# HIDDEN_SIZE = 128
# N_LAYER = 1
# FC_HIDDEN_SIZE = 128

N_FILTER = 64
L2_REG = 0.0001
N_BLOCKS = 7

#  MCTS
C_PUCT = 100
DIRICHLET = 1.5
EPSILON = 0.25

#  Training
LEARNING_RATE = 1e-3
TEMP = 0.0001
N_PLAYOUT = 6000
BUFFER_SIZE = 1000*SEQ_LENGTH
BATCH_SIZE = 256
KL_TARG = 0.0002
GAME_BATCH_NUM = 2500
EPOCH = 1
N_POLICY_UPDATE = 5


#  Session
SESS_CONFIG = tf.ConfigProto()
SESS_CONFIG.gpu_options.allow_growth = True

#  Global varibles
BEST_ENERGY = 0
BEST_ACTION_LIST = []

#  Path
# WORK_PATH = "/nfshome/hangyu/HP-model_uct/"
WORK_PATH = "./"
MODEL_NAME = "UCT_"+str(len(HP_SEQ))+"_"+str(datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))
MODEL_PATH = WORK_PATH + 'models/' + MODEL_NAME + "/"
