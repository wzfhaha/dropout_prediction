import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from time import time
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from sklearn.metrics import f1_score


class CFIN():
    def __init__(self, a_feat_size, u_feat_size, c_feat_size, a_field_size, u_field_size, c_field_size,
                 embedding_size=8,
                 conv_size = 32,
                 context_size = 16,
                 deep_layers=[32, 32], dropout_deep=[0.5, 0.5, 0.5], dropout_attn = [0.5, 0.5],
                 activation=tf.nn.relu,
                 attn_size = 16,
                 epoch=10, batch_size=256,
                 
                 learning_rate=0.001, optimizer_type="adam",
                 batch_norm=0, batch_norm_decay=0.995,
                 verbose=False, random_seed=2016,
                 loss_type="logloss", eval_metric=roc_auc_score,
                 l2_reg=0.0, attn=True):

        self.a_feat_size = a_feat_size      
        self.u_feat_size = u_feat_size
        self.c_feat_size = c_feat_size
        
        self.a_field_size = a_field_size
        self.u_field_size = u_field_size
        self.c_field_size = c_field_size
        self.conv_size = conv_size
        self.context_size = context_size
        self.embedding_size = embedding_size
         
        self.deep_layers = deep_layers
        self.dropout_deep = dropout_deep
        self.dropout_attn = dropout_attn
        self.activation = activation
        self.l2_reg = l2_reg
        self.attn_enable = attn
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.attn_size = attn_size
        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay
        self.greater_is_better = True 
        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.train_result, self.valid_result = [], []
        self._init_graph()
        

    def _init_graph(self):
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            
            self.u_feat_index = tf.placeholder(tf.int32, shape=[None, None],
                                                 name="u_feat_index")  # None * F
            self.c_feat_index = tf.placeholder(tf.int32, shape=[None, None], name='c_feat_index')
            self.a_feat_index = tf.placeholder(tf.int32, shape=[None, None], name='a_feat_index')
            self.dropout_keep_attn = tf.placeholder(tf.float32, shape=[None], name = 'dropout_attn_uc')
            
            self.u_feat_value = tf.placeholder(tf.float32, shape=[None, None], name="u_feat_value")
            self.c_feat_value = tf.placeholder(tf.float32, shape=[None, None], name="c_feat_value")
            self.a_feat_value = tf.placeholder(tf.float32, shape=[None, None], name="a_feat_value")
            
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")  # None * 1
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_deep")
            self.train_phase = tf.placeholder(tf.bool, name="train_phase")
            
            self.weights = self._initialize_weights()
             
            # model
            self.a_embeddings = tf.nn.embedding_lookup(self.weights["a_feat_embeddings"], self.a_feat_index)
            self.u_embeddings = tf.nn.embedding_lookup(self.weights['u_feat_embeddings'], self.u_feat_index)
            self.c_embeddings = tf.nn.embedding_lookup(self.weights['c_feat_embeddings'], self.c_feat_index)
                       
            u_feat_value = tf.reshape(self.u_feat_value, shape=[-1, self.u_field_size, 1])
            c_feat_value = tf.reshape(self.c_feat_value, shape=[-1, self.c_field_size, 1])
            a_feat_value = tf.reshape(self.a_feat_value, shape=[-1, self.a_field_size, 1])
            self.c_embeddings = tf.multiply(self.c_embeddings, c_feat_value)
            self.u_embeddings = tf.multiply(self.u_embeddings, u_feat_value)
            self.a_embeddings = tf.multiply(self.a_embeddings, a_feat_value)
            if self.batch_norm:
                self.a_embeddings = self.batch_norm_layer(self.a_embeddings, train_phase=self.train_phase, scope_bn='bn_conv')
            self.a_embeddings = tf.nn.conv1d(self.a_embeddings, self.weights['a_conv_filter'], stride=5, padding='VALID',data_format='NWC') + self.weights['a_conv_bias']
            """
            if self.batch_norm:
                self.a_embeddings = self.batch_norm_layer(self.a_embeddings, train_phase=self.train_phase, scope_bn='bn_conv')
            """
            self.a_embeddings = tf.nn.relu(self.a_embeddings)
            self.uc_embeddings = tf.concat([self.u_embeddings, self.c_embeddings], axis=1)
            
            self.uc_inter = tf.nn.relu(tf.matmul(tf.reshape(self.uc_embeddings, shape=[-1, (self.u_field_size+self.c_field_size)*self.embedding_size]), self.weights['ctx_pool_weight'])+self.weights['ctx_pool_bias'])

            self.uca_inter = tf.concat([tf.tile(tf.expand_dims(self.uc_inter, 1), [1,self.a_field_size//5,1]), self.a_embeddings], 2)
            
            self.attn_logit = tf.nn.relu(tf.matmul(tf.reshape(self.uca_inter, shape=[-1, self.conv_size + self.context_size]), self.weights['attn_out_1']) + self.weights['attn_bias_1'])
            self.attn_w = tf.nn.softmax(tf.reshape(tf.matmul(self.attn_logit, self.weights['attn_out']), shape=[-1, self.a_field_size//5]))
            if self.attn_enable:
                self.a_weight_emb = tf.multiply(tf.expand_dims(self.attn_w,2), self.a_embeddings)
            else:
                self.a_weight_emb = self.a_embeddings

            deep_input = tf.reduce_sum(self.a_weight_emb, axis=1)
            deep_input = tf.concat([deep_input, self.uc_inter], axis=1) 
            
            #self.y_deep = tf.reshape(deep_input, shape=[-1,  self.conv_size])
            self.y_deep = deep_input
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])
            for i in range(0, len(self.deep_layers)):
                self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" %i]), self.weights["bias_%d"%i])
                if self.batch_norm:
                    self.y_deep = self.batch_norm_layer(self.y_deep, train_phase=self.train_phase, scope_bn="bn_%d" %i) 
                self.y_deep = self.activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[1+i]) # dropout at each Deep layer
            
            self.out = tf.add(tf.matmul(self.y_deep, self.weights["logistic_weight"]), self.weights["logistic_bias"])

            # loss
            self.out = tf.nn.sigmoid(self.out)
            self.loss = tf.losses.log_loss(self.label, self.out)
            # l2 regularization on weights
            if self.l2_reg > 0:
                for k in self.weights.keys():
                    self.loss += tf.contrib.layers.l2_regularizer(
                            self.l2_reg)(self.weights[k])
            # optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)

            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print("#params: %d" % total_parameters)


    def _init_session(self):
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        return tf.Session(config=config)

    def _init_layer_weight(self, input_size, output_size):
        glorot = np.sqrt(2.0 / (input_size + output_size))
        
        return np.random.normal(loc=0, scale=glorot, size=(input_size, output_size)), np.random.normal(loc=0, scale=glorot, size=(1, output_size))
 
    def _initialize_weights(self, c_node_embedding=None, u_node_embedding=None):
        weights = dict()

        # embeddings
        weights["a_feat_embeddings"] = tf.Variable(
            tf.random_normal([self.a_feat_size, self.embedding_size], 0.0, 0.1),
            name="a_feature_embeddings")  # feature_size * K

        weights["u_feat_embeddings"] = tf.Variable(
            tf.random_normal([self.u_feat_size, self.embedding_size], 0.0, 0.1),
            name="u_feature_embeddings")  # feature_size * K
        
        weights["c_feat_embeddings"] = tf.Variable(
            tf.random_normal([self.c_feat_size, self.embedding_size], 0.0, 0.1),
            name="c_feature_embeddings")  # feature_size * K
         
        u_pool_w, u_pool_b = self._init_layer_weight((self.u_field_size+self.c_field_size)*self.embedding_size, self.context_size)
        
        weights['ctx_pool_weight'] = tf.Variable(u_pool_w, name='ctx_pool_weight', dtype=np.float32)
        
        weights["ctx_pool_bias"] = tf.Variable(u_pool_b, name='ctx_pool_bias',dtype=np.float32)  # 1 * layers[0]
        
        a_conv_w, a_conv_b = self._init_layer_weight(5*self.embedding_size, self.conv_size)
        weights['a_conv_filter'] = tf.Variable(np.reshape(a_conv_w, [5, self.embedding_size, self.conv_size]), name='a_conv_filter', dtype=np.float32)
        weights['a_conv_bias'] = tf.Variable(np.reshape(a_conv_b, [1,1,self.conv_size]), name='a_conv_bias', dtype=np.float32)
        attn_out_1, attn_bias_1 = self._init_layer_weight(self.conv_size+self.context_size, self.attn_size)
        weights['attn_out_1'] = tf.Variable(attn_out_1, name='attn_out_1', dtype='float32')
        weights['attn_bias_1'] = tf.Variable(attn_bias_1, name='attn_bias_1', dtype='float32')
        attn_out, attn_bias = self._init_layer_weight(self.attn_size, 1)
        weights['attn_out'] = tf.Variable(attn_out, name='attn_out', dtype='float32')
        
        num_layer = len(self.deep_layers)
        input_size = self.conv_size + self.context_size
        
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))
        weights["layer_0"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32)
        weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])), dtype=np.float32)  # 1 * layers[0]
        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i-1] + self.deep_layers[i]))
            weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i-1], self.deep_layers[i])),
                dtype=np.float32)  # layers[i-1] * layers[i]
            weights["bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                dtype=np.float32)  # 1 * layer[i]
        
        input_size = self.deep_layers[-1]
        glorot = np.sqrt(2.0 / (input_size + 1))
        weights["logistic_weight"] = tf.Variable(
                        np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
                        dtype=np.float32)  # layers[i-1]*layers[i]
        weights["logistic_bias"] = tf.Variable(tf.constant(0.01), dtype=np.float32)
        return weights


    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z


    def get_batch(self, ui, uv, ci, cv, ai, av, y, batch_size, index):
        start = index * batch_size
        end = (index+1) * batch_size
        end = end if end < len(y) else len(y)
        return ui[start:end], uv[start:end], ci[start:end], cv[start:end], ai[start:end], av[start:end], [[y_] for y_ in y[start:end]]

    def shuffle_in_unison_scary(self, a, b, c, d,e,f,g):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)
        np.random.set_state(rng_state)
        np.random.shuffle(d)
        np.random.set_state(rng_state)
        np.random.shuffle(e)
        np.random.set_state(rng_state)
        np.random.shuffle(f)
        np.random.set_state(rng_state)
        np.random.shuffle(g)

    def fit_on_batch(self, ui, uv, ci, cv, ai, av, y):
        feed_dict = {self.u_feat_index: ui,
                     self.u_feat_value: uv,
                     self.c_feat_index: ci,
                     self.c_feat_value: cv,
                     self.a_feat_index: ai,
                     self.a_feat_value: av,
                     self.dropout_keep_deep: self.dropout_deep,
                     self.dropout_keep_attn: self.dropout_attn,
                     self.label: y,
                     self.train_phase: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def fit(self, ui_train, uv_train, ci_train, cv_train, ai_train, av_train, y_train,
            ui_valid=None, uv_valid=None, ci_valid = None, cv_valid = None, ai_valid=None, av_valid=None,y_valid=None,
            early_stopping = True, early_stopping_round = 5, max_epoch =None):
        has_valid = uv_valid is not None
        best_epoch = 0
        if max_epoch:
            self.epoch = max_epoch
        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(ui_train, uv_train, ci_train, cv_train, ai_train, av_train, y_train)
            total_batch = int(len(y_train) / self.batch_size)
            for i in range(total_batch):
                ui_batch, uv_batch, ci_batch, cv_batch, ai_batch, av_batch, y_batch = self.get_batch(ui_train, uv_train, ci_train, cv_train, ai_train, av_train, y_train, self.batch_size, i)
                self.fit_on_batch(ui_batch, uv_batch, ci_batch, cv_batch, ai_batch, av_batch, y_batch)

            # evaluate training and validation datasets
            train_result, train_deep, train_f1 = self.evaluate(ui_train, uv_train, ci_train, cv_train, ai_train, av_train, y_train)
            self.train_result.append(train_result)
            if has_valid:
                valid_result, valid_deep, valid_f1= self.evaluate(ui_valid, uv_valid, ci_valid, cv_valid, ai_valid, av_valid, y_valid)
                self.valid_result.append(valid_result)
            if self.verbose > 0 and epoch % self.verbose == 0:
                if has_valid:
                    print("[%d] train-result=%.4f, valid-result=%.4f, valid-f1=%.4f [%.1f s]"
                        % (epoch + 1, train_result, valid_result, valid_f1, time() - t1))
                else:
                    print("[%d] train-result=%.4f [%.1f s]"
                        % (epoch + 1, train_result, time() - t1))
            
            if has_valid and early_stopping and self.training_termination(self.valid_result, early_stopping_round):
                best_epoch = epoch - early_stopping_round + 1
                print("best epoch: ", best_epoch)
                return best_epoch
                
        """         
        if has_valid and refit:
            if self.greater_is_better:
                best_valid_score = max(self.valid_result)
            else:
                best_valid_score = min(self.valid_result)
            best_epoch = self.valid_result.index(best_valid_score)
            best_train_score = self.train_result[best_epoch]
            ui_train = np.concatenate((ui_train, ui_valid), axis=0)
            uv_train = np.concatenate((uv_train, uv_valid), axis=0)
            ci_train = np.concatenate((ci_train, ci_valid), axis=0)
            cv_train = np.concatenate((cv_train, cv_valid), axis=0)
            ai_train = np.concatenate((ai_train, ai_valid), axis=0)
            av_train = np.concatenate((av_train, av_valid), axis=0)
            y_train = np.concatenate((y_train, y_valid), axis=0)

            for epoch in range(100):
                self.shuffle_in_unison_scary(ui_train, uv_train, ci_train, cv_train, ai_train, av_train, y_train)
                total_batch = int(len(y_train) / self.batch_size)
                for i in range(total_batch):
                    ui_batch, uv_batch, ci_batch, cv_batch, ai_batch, av_batch, y_batch = self.get_batch(ui_train, uv_train, ci_train, cv_train, ai_train, av_train, y_train, self.batch_size, i)
                    self.fit_on_batch(ui_batch, uv_batch, ci_batch, cv_batch, ai_batch, av_batch, y_batch)
                train_result, train_deep, train_f1 = self.evaluate(ui_train, uv_train, ci_train, cv_train, ai_train, av_train, y_train)
                if abs(train_result - best_train_score) < 0.001 or \
                    (self.greater_is_better and train_result > best_train_score) or \
                    ((not self.greater_is_better) and train_result < best_train_score):
                    break
        """
        save_path = self.saver.save(sess, "my_model/CFIN")
        print("Save to path: ", save_path)
        return save_path
        
    def training_termination(self, valid_result, early_stopping_round):
        if len(valid_result) > early_stopping_round:
            if self.greater_is_better:
                if max(valid_result[-early_stopping_round:]) > valid_result[-1-early_stopping_round]:
                    return False
                else:
                    return True
            else:
                if min(valid_result[-early_stopping_round:]) < valid_result[-1-early_stopping_round]:
                    return False
                else:
                    return True    
        return False

    def get_feats(self, ui,uv,ci,cv, ai, av, fname):
        dummy_y = [[1]] *len(ui)
        feed_dict = {self.u_feat_index: ui,
                     self.u_feat_value: uv,
                     self.c_feat_index: ci,
                     self.c_feat_value: cv,
                     self.a_feat_index: ai,
                     self.a_feat_value: av,
                     self.dropout_keep_deep: [1.0] * len(self.dropout_deep),
                     self.dropout_keep_attn: [1.0] * len(self.dropout_attn),
                     self.label: dummy_y,
                     self.train_phase: False}
        
        y_deep, y_rate= self.sess.run([self.y_deep, self.out], feed_dict = feed_dict)
        return y_deep,y_rate
    def get_attn(self, ui, uv, ci, cv, ai, av):
        dummy_y = [1] * len(ui)
        batch_index = 0
        ui_batch, uv_batch, ci_batch, cv_batch, ai_batch, av_batch ,y_batch = self.get_batch(ui, uv, ci, cv, ai, av, dummy_y, self.batch_size, batch_index)
        attn_weight = None
        uca_weight = None
        while len(ui_batch) > 0:
            num_batch = len(y_batch)
            feed_dict = {self.u_feat_index: ui_batch,
                         self.u_feat_value: uv_batch,
                         self.c_feat_index: ci_batch,
                         self.c_feat_value: cv_batch,
                         self.a_feat_index: ai_batch,
                         self.a_feat_value: av_batch,
                         self.dropout_keep_deep: [1.0] * len(self.dropout_deep),
                         self.dropout_keep_attn: [1.0] * len(self.dropout_attn),
                         self.label: y_batch,
                         self.train_phase: False}
            attn, uca_inter = self.sess.run([self.attn_w,self.uca_inter], feed_dict=feed_dict)
            if batch_index == 0:
                attn_weight = np.reshape(attn, (num_batch,-1))
                uca_weight = np.reshape(uca_inter, (num_batch, -1, self.embedding_size))
            else:
                attn_weight = np.concatenate((attn_weight, np.reshape(attn, (num_batch, -1))))
                uca_weight = np.concatenate((uca_weight, np.reshape(uca_inter, (num_batch, -1, self.embedding_size))))
            batch_index += 1
            
            ui_batch, uv_batch, ci_batch, cv_batch, ai_batch, av_batch ,y_batch = self.get_batch(ui, uv, ci, cv, ai, av, dummy_y, self.batch_size, batch_index)
        return attn_weight, uca_weight

   
    def predict(self, ui, uv, ci, cv, ai, av):
        dummy_y = [1] * len(ui)
        batch_index = 0
        ui_batch, uv_batch, ci_batch, cv_batch, ai_batch, av_batch ,y_batch = self.get_batch(ui, uv, ci, cv, ai, av, dummy_y, self.batch_size, batch_index)
        y_pred = None
        while len(ui_batch) > 0:
            num_batch = len(y_batch)
            feed_dict = {self.u_feat_index: ui_batch,
                         self.u_feat_value: uv_batch,
                         self.c_feat_index: ci_batch,
                         self.c_feat_value: cv_batch,
                         self.a_feat_index: ai_batch,
                         self.a_feat_value: av_batch,
                         self.dropout_keep_deep: [1.0] * len(self.dropout_deep),
                         self.dropout_keep_attn: [1.0] * len(self.dropout_attn),
                         self.label: y_batch,
                         self.train_phase: False}
            y_deep,batch_out = self.sess.run([self.y_deep,self.out], feed_dict=feed_dict)
            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

            batch_index += 1
            
            ui_batch, uv_batch, ci_batch, cv_batch, ai_batch, av_batch ,y_batch = self.get_batch(ui, uv, ci, cv, ai, av, dummy_y, self.batch_size, batch_index)
        return y_deep,y_pred


    def evaluate(self, ui, uv, ci, cv, ai, av, y):
        y_deep, y_pred = self.predict(ui, uv, ci, cv, ai, av)  
        return self.eval_metric(y, y_pred), y_deep, f1_score(y, [1 if x>0.5 else 0 for x in y_pred])

