import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average

import config
import global_variables


class ResNet:
    """
    title: inputs->ResNet->output
    """

    def __init__(self, ):
        self.model_name = config.MODEL_NAME
        self.input_len = config.SEQ_LENGTH  # number of the residues
        self.vector_len = config.VECTOR_LENGTH
        self.n_class = len(global_variables.ACTION_VECTOR_DICT.keys()) - 1
        self._global_step = tf.Variable(0, trainable=False, name='Global_Step')
        # placeholders
        self._tst = tf.placeholder(tf.bool, name='tst')
        self._keep_prob = tf.placeholder(tf.float32, [], name='keep_prob')
        self._batch_size = tf.placeholder(tf.int32, [], name='batch_size')

        with tf.name_scope('Inputs'):
            self._X_inputs = tf.placeholder(tf.float32, [None, self.input_len, self.vector_len], name='X_inputs')
            self._label_p = tf.placeholder(tf.float32, [None, self.n_class], name='policy_label')
            self._label_v = tf.placeholder(tf.float32, [None, 1], name="value_label")

        self.conv1 = tf.layers.conv1d(inputs=self._X_inputs, name="conv1",  kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2, mode='FAN_IN', uniform=False),
                                      filters=config.N_FILTER, kernel_size=3,
                                      padding="same", activation=None)
        self.conv1_bn = self.batch_norm(self.conv1, is_testing=self._tst)
        self.conv1_act = tf.nn.relu(self.conv1_bn)
        conv1_w = tf.get_collection(tf.GraphKeys.VARIABLES, 'conv1/kernel')[0]
        conv1_b = tf.get_collection(tf.GraphKeys.VARIABLES, 'conv1/bias')[0]
        tf.summary.histogram("conv1_w", conv1_w)
        tf.summary.histogram("conv1_b", conv1_b)

        with tf.name_scope('res_blocks'):
            x = self.conv1_act
            for i in range(config.N_BLOCKS):
                x = self._build_residual_block(x, i + 1)
            self.res_out = x


        with tf.variable_scope('policy_head'):
            # 3-1 Action Networks
            self.action_conv = tf.layers.conv1d(inputs=self.res_out, filters=4,  kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1., mode='FAN_IN', uniform=False),
                                                kernel_size=1, padding="same",
                                                activation=None, name="convp")
            self.action_conv_bn = self.batch_norm(self.action_conv, is_testing=self._tst)
            self.action_conv_act = tf.nn.relu(self.action_conv_bn)
            convp_w = tf.get_collection(tf.GraphKeys.VARIABLES, 'policy_head/convp/kernel')[0]
            convp_b = tf.get_collection(tf.GraphKeys.VARIABLES, 'policy_head/convp/bias')[0]
            tf.summary.histogram("convp_w", convp_w)
            tf.summary.histogram("convp_b", convp_b)
            # Flatten the tensor
            self.action_conv_flat = tf.reshape(
                self.action_conv_act, [-1, 4 * self.input_len])
            # 3-2 Full connected layer, the output is the log probability of moves
            # on each slot on the board
            self._policy_out = tf.layers.dense(inputs=self.action_conv_flat, units=self.n_class, activation=tf.nn.log_softmax, name="dense_p")
            self._policy_out_softmax = tf.exp(self._policy_out)

        with tf.variable_scope('value_head'):
            # 4 Evaluation Networks
            self.evaluation_conv = tf.layers.conv1d(inputs=self.res_out, filters=2,  kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1, mode='FAN_IN', uniform=False),
                                                    kernel_size=1,
                                                    padding="same",
                                                    activation=None, name="convv")
            self.evaluation_conv_bn = self.batch_norm(self.evaluation_conv, is_testing=self._tst)
            self.evaluation_conv_act = tf.nn.relu(self.evaluation_conv_bn)
            convv_w = tf.get_collection(tf.GraphKeys.VARIABLES, 'value_head/convv/kernel')[0]
            convv_b = tf.get_collection(tf.GraphKeys.VARIABLES, 'value_head/convv/bias')[0]
            tf.summary.histogram("convv_w", convv_w)
            tf.summary.histogram("convv_b", convv_b)
            self.evaluation_conv_flat = tf.reshape(
                self.evaluation_conv_act, [-1, 2 * self.input_len])
            self.evaluation_fc1 = tf.layers.dense(inputs=self.evaluation_conv_flat,
                                                  units=64, activation=tf.nn.relu, name="dense1_v")
            # output the score of evaluation on current state
            self._value_out = tf.layers.dense(inputs=self.evaluation_fc1, units=1, activation=tf.nn.tanh, name="dense2_v")
            self._value_out_mean = tf.reduce_mean(self._value_out)

        with tf.variable_scope('loss'):
            # Define the Loss function
            # 1. Label: the array containing if the game wins or not for each state
            # 2. Predictions: the array containing the evaluation score of each state
            # which is self.evaluation_fc2
            # 3-1. Value Loss function
            self._value_loss = tf.losses.mean_squared_error(self._label_v, self._value_out)
            tf.summary.scalar("value_loss", self.value_loss)
            # 3-2. Policy Loss function
            self._policy_loss = tf.negative(tf.reduce_mean(tf.reduce_sum(tf.multiply(self._label_p, self._policy_out), 1)))
            tf.summary.scalar("policy_loss", self.policy_loss)
            # 3-3. L2 penalty (regularization)
            l2_penalty_beta = config.L2_REG
            vars = tf.trainable_variables()
            self.l2_penalty = l2_penalty_beta * tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])
            # 3-4 Add up to be the Loss function
            self._loss = self.value_loss + self.policy_loss + self.l2_penalty
            tf.summary.scalar("loss", self.loss)

            # calc policy entropy, for monitoring only
            self._entropy = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.log(self._policy_out_softmax) * self._policy_out_softmax, 1)))
            tf.summary.scalar("entropy", self.entropy)

        with tf.name_scope('train_option'):
            self._learning_rate = tf.placeholder(tf.float32, name="learning_rate")
            tf.summary.scalar("learning_rate", self._learning_rate)
            self._optimizer = tf.train.MomentumOptimizer(learning_rate=self._learning_rate, momentum=0.9)
            #  when training, fetch train_op and update_op at the same time
            self.train_op = self._optimizer.minimize(self._loss)

        self.sess = tf.Session(config=config.SESS_CONFIG)

        self.train_writer = tf.summary.FileWriter(config.WORK_PATH + "summary/" + config.MODEL_NAME + '/train/',
                                                  self.sess.graph)

        writer = tf.summary.FileWriter(config.WORK_PATH + "summary/" + config.MODEL_NAME, self.sess.graph)
        writer.add_graph(self.sess.graph)

        self.merged = tf.summary.merge_all()

        # Initialize variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.saver = tf.train.Saver(max_to_keep=10)

    @property
    def tst(self):
        return self._tst

    @property
    def keep_prob(self):
        return self._keep_prob

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def global_step(self):
        return self._global_step

    @property
    def X_inputs(self):
        return self._X_inputs

    @property
    def label_p(self):
        return self._label_p

    @property
    def label_v(self):
        return self._label_v

    @property
    def policy_out(self):
        return self._policy_out

    @property
    def policy_out_softmax(self):
        return self._policy_out_softmax

    @property
    def value_out(self):
        return self._value_out

    @property
    def value_out_mean(self):
        return self._value_out_mean

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def policy_loss(self):
        return self._policy_loss

    @property
    def value_loss(self):
        return self._value_loss

    @property
    def loss(self):
        return self._loss

    @property
    def entropy(self):
        return self._entropy

    def batch_norm(self, x, is_testing, eps=1e-05, decay=0.9, affine=True, name=None):
        with tf.variable_scope(name, default_name='BatchNorm2d'):
            params_shape = [x.shape[-1]]
            moving_mean = tf.get_variable('mean', shape=params_shape, initializer=tf.zeros_initializer, trainable=False)
            moving_variance = tf.get_variable('variance', shape=params_shape, initializer=tf.ones_initializer, trainable=False)
            def mean_var_with_update():
                mean, variance = tf.nn.moments(x, list(range(len(x.shape)-1)), name='moments')
                with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),
                                              assign_moving_average(moving_variance, variance, decay)]):
                    return tf.identity(mean), tf.identity(variance)

            mean, variance = tf.cond(is_testing, lambda: (moving_mean, moving_variance), mean_var_with_update)
            if affine:
                beta = tf.get_variable('beta', params_shape,
                                       initializer=tf.zeros_initializer)
                gamma = tf.get_variable('gamma', params_shape,
                                        initializer=tf.ones_initializer)
                x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
            else:
                x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
            return x

    def _build_residual_block(self, x, index):
        in_x = x
        res_name = "res_block_" + str(index)
        with tf.name_scope(res_name):
            x = tf.layers.conv1d(inputs=x, name=res_name+"_conv1", filters=config.N_FILTER, kernel_size=3, padding="same", kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=0.5, mode='FAN_IN', uniform=False), activation=None)
            x_bn = self.batch_norm(x, is_testing=self._tst)
            x_act = tf.nn.relu(x_bn)
            conv1_w = tf.get_collection(tf.GraphKeys.VARIABLES, res_name+'_conv1/kernel')[0]
            conv1_b = tf.get_collection(tf.GraphKeys.VARIABLES, res_name+'_conv1/bias')[0]
            tf.summary.histogram(res_name+"_conv1_w", conv1_w)
            tf.summary.histogram(res_name+"_conv1_b", conv1_b)
            x = tf.layers.conv1d(x_act, name=res_name+"_conv2", filters=config.N_FILTER, kernel_size=3, padding="same", kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=0.5, mode='FAN_IN', uniform=False), activation=None)
            conv2_w = tf.get_collection(tf.GraphKeys.VARIABLES, res_name+'_conv2/kernel')[0]
            conv2_b = tf.get_collection(tf.GraphKeys.VARIABLES, res_name+'_conv2/bias')[0]
            tf.summary.histogram(res_name + "_conv1_w", conv2_w)
            tf.summary.histogram(res_name + "_conv1_b", conv2_b)
            x = self.batch_norm(x, is_testing=self._tst)
            x = tf.add(in_x, x)
            x_act = tf.nn.relu(x)

        return x_act

    def policy_value(self, encoded_state):
        """
        input: an encoded state
        output: a batch of action probabilities and state values
        """
        act_probs, value = self.sess.run([self._policy_out_softmax, self._value_out],
                                             feed_dict={self._X_inputs: encoded_state,
                                                        self._tst: True,
                                                        self._keep_prob: 1.0,
                                                        self.batch_size: encoded_state.shape[0]}
                                             )
        return act_probs, value

    def policy_value_fn(self, state):
        """
        input: a state (env)
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_actions = state.legal_action_list
        legal_actions_int = []
        for action in legal_actions:
            legal_actions_int.append(global_variables.ACTION_INT_DICT[action])

        encoded_state = state.encoded_state

        act_probs, value = self.policy_value(encoded_state)
        act_probs = zip(legal_actions_int, act_probs[0][legal_actions_int])
        return act_probs, value

    def train_step(self, state_batch, mcts_probs_batch, value_batch, learning_rate, step):
        fetch = [self.merged, self.loss, self.policy_loss, self.value_loss, self.policy_out_softmax, self.value_out_mean, self.entropy, self.train_op]
        summary, loss, policy_loss, value_loss, policy_out_softmax, value_out_mean, entropy, train_op \
            = self.sess.run(fetch, feed_dict={self.X_inputs: state_batch,
                                              self.label_p: mcts_probs_batch,
                                              self.label_v: value_batch,
                                              self.learning_rate: learning_rate,
                                              self.tst: False,
                                              self.keep_prob: 0.5,
                                              self.batch_size: config.BATCH_SIZE})
        self.train_writer.add_summary(summary, step)
        return loss, policy_loss, value_loss, value_out_mean, entropy

    def save_model(self, i):
        self.saver.save(self.sess, config.MODEL_PATH+'model-' + str(i) + '.ckpt')

    def restore_model(self, model_path):
        self.saver.restore(self.sess, model_path)

# test the model


if __name__ == '__main__':
    pass