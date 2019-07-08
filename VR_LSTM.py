"""
   Written by Pengfei Sun. The proposed VRLSTM incoporates variational approximation 
   and rewarded reinforment with long short term sequence model. It is naturally inspired 
   by the brain speech mechanism. 
"""
class VRLSTM(object):       
        def __init__(self, D, M, K,  dura, bs=1, LR=0.1, 
                     ckpt_path="models/lstmr/", model_name="lstmr"):
            """
            D: dimensionality of inputs channel
            M: size of hidden layer
            V: size of vocabulary
            K: num of output classes
            LR: learning rate
            """
            self.D, self.M, self.K, self.bs, self.LR = D, M, K, bs, LR
            self.dura = dura
            self.ckpt_path = ckpt_path
            self.model_name= model_name
            
            def __graph__():    
                tf.reset_default_graph()
                x_inputs = tf.placeholder(shape=[self.bs, self.dura, self.D], dtype=tf.float32)
                ys_ = tf.placeholder(shape=[self.bs], dtype=tf.int32)
            
                xav_init = tf.contrib.layers.xavier_initializer
            
                with tf.name_scope("Variable"):
                    self.Wxf = tf.get_variable("Wxf", shape=[self.D, self.M], initializer=xav_init())
                    self.Whf = tf.get_variable("Whf", shape=[self.M, self.M], initializer=xav_init())
            
                    self.Wxi = tf.get_variable("Wxi", shape=[self.D, self.M], initializer=xav_init())
                    self.Whi = tf.get_variable("Whi", shape=[self.M, self.M], initializer=xav_init())
                
                    self.Wxc = tf.get_variable("Wxc", shape=[self.D, self.M], initializer=xav_init())
                    self.Whc = tf.get_variable("Whc", shape=[self.M, self.M], initializer=xav_init())
                    # residual compoent c to instantaneous component h
                    self.Wcr = tf.get_variable("Wcr", shape=[self.M, self.M], initializer=xav_init())
                    
                    self.Wco = tf.get_variable("Wxo", shape=[self.M, self.D], initializer=xav_init())
                    self.Who = tf.get_variable("Who", shape=[self.M, self.D], initializer=xav_init())
                
                with tf.name_scope("biases"):
                    self.bf  =  tf.get_variable("bf", shape=[self.M], 
                                                dtype=tf.float32, initializer=tf.constant_initializer(0))
                    self.bi  =  tf.get_variable("bi", shape=[self.M], 
                                                dtype=tf.float32, initializer=tf.constant_initializer(0))
                    self.bc  =  tf.get_variable("bc", shape=[self.M], 
                                                dtype=tf.float32, initializer=tf.constant_initializer(0))
                    self.bh  =  tf.get_variable("bh", shape=[self.M], 
                                                dtype=tf.float32, initializer=tf.constant_initializer(0))
                    self.bo  =  tf.get_variable("bo", shape=[self.D],  # output the bais
                                                dtype=tf.float32, initializer=tf.constant_initializer(0))
            
                init_state = tf.placeholder(shape=[2, None, self.M], dtype=tf.float32, name='initial_state')    
             
                def _recurrence(prev, x_t):
                    h_t_minus_1, c_t_minus_1 = tf.unstack(prev)
            
                    f_t = tf.nn.sigmoid(
                        tf.matmul(x_t, self.Wxf) + tf.matmul(h_t_minus_1, self.Whf) + self.bf)
                      
                    i_t = tf.nn.sigmoid(
                        tf.matmul(x_t, self.Wxi) + tf.matmul(h_t_minus_1, self.Whi) + self.bi)
                    
            
                    c_hat_t = tf.nn.tanh(
                        tf.matmul(x_t, self.Wxc) + tf.matmul(h_t_minus_1, self.Whc) + self.bc)
            
                    c_t = (f_t * c_t_minus_1) + (i_t * c_hat_t)
            
                    h_t = tf.nn.tanh(
                          tf.matmul(c_t, self.Wcr) + self.bh)
                
#                     h_t = tf.nn.relu(
#                           tf.matmul(c_t, self.Wcr) + self.bh)
                
                    return tf.stack([h_t, c_t])
 
                x_inputs_hat = tf.transpose(x_inputs, [1,0,2])
                states = tf.scan(_recurrence, 
                         x_inputs_hat,
                         initializer=init_state)
    
                V = tf.get_variable('V', shape=[x_inputs.shape[1]*self.M, 29], initializer=xav_init())
                b = tf.get_variable('b', shape=[29], initializer=tf.constant_initializer(0.))
                last_state = states[-1]
            
                states = tf.transpose(states,[1,2,0,3])
            
                h_st = states[0]
                c_ht = states[1]
            
                # encodering decoding variational layer
#                 Out_put = tf.nn.tanh(tf.tensordot(c_ht, self.Wco, axes=[[-1], [0]]) + \
#                                     tf.tensordot(h_st, self.Who, axes=[[-1], [0]]) + self.bo)
                Out_put = tf.nn.tanh(tf.tensordot(c_ht+h_st, self.Wco, axes=[[-1], [0]]) + self.bo)
        
                states_reshaped = tf.reshape(h_st, [self.bs, -1])
                logits = tf.matmul(states_reshaped, V) + b
                predictions = tf.nn.softmax(logits)
                
                predictions = tf.argmax(predictions, 1)
                predictions = predictions[:]

                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ys_, logits=logits)
                
                
                dist_  = tf.losses.mean_squared_error(Out_put, x_inputs)
#                 ce     = x_inputs*tf.log(1e-10+Out_put) + (1.0-x_inputs)*tf.log(1e-10+1.0-Out_put)
#                 dist_  = -tf.reduce_sum(tf.boolean_mask(ce, tf.is_finite(ce)), axis=-1)
#                 dist_  = tf.reduce_mean(dist_) 
                
                loss   = 100*tf.reduce_mean(losses) + dist_
#                 loss   = 100*tf.reduce_mean(losses)
            
                train_op = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(loss)
            
                self.x_inputs = x_inputs
                self.ys_      = ys_
                self.loss     = loss
                self.train_op = train_op
                self.predictions= predictions
                self.last_state= last_state
                self.init_state=init_state
            
            __graph__()
            
        def train(self, train_set, epochs=1000):
            if os.path.exists(self.ckpt_path + 'checkpoint'):
                files = glob.glob(self.ckpt_path+'*')
                for file_ in files:
                    os.remove(file_)
            
            with tf.Session() as sess:
                 saver = tf.train.Saver()
                 if os.path.exists(self.ckpt_path + 'checkpoint'):
                        path_ck = tf.train.latest_checkpoint(self.ckpt_path)
                        saver.restore(sess, path_ck)
                 else:
                        sess.run(tf.global_variables_initializer())
                 train_loss=0
                 for i in range(epochs):
                        for j in range(len(train_set)):
                            xs, ys = train_set[j] # train_set data input: data & label 
                            batch_size = xs.shape[0]
                            _, train_loss_ = sess.run([self.train_op, self.loss], feed_dict = {
                                self.x_inputs : xs,
                                self.ys_ : ys,
                                self.init_state : np.zeros([2, batch_size, self.M], dtype=np.float32)
                            })
                            train_loss += train_loss_
                        if i%10==0:
                            print('[{}] loss : {}'.format(i,train_loss/100))
                        train_loss = 0
                 saver.save(sess, self.ckpt_path + self.model_name, global_step=i)
        
        def evaluation(self, vali_set):
            with tf.Session() as sess:
                 saver = tf.train.Saver()
                 path_ck = tf.train.latest_checkpoint(self.ckpt_path)
                 saver.restore(sess, path_ck) 
                 xs, ys = vali_set
                 batch_size = xs.shape[0]
                 predict_labels = sess.run(self.predictions, feed_dict={
                     self.x_inputs: xs, 
                     self.ys_ : ys, 
                     self.init_state: np.zeros([2, batch_size, self.M], dtype=np.float32)
                 })
                 print(np.shape(predict_labels))
                 acc = 0
                 for i in range(len(predict_labels)):
                     if predict_labels[i] == ys[i]:
                            acc += 1
                 print(acc/(i+1))
                                  
