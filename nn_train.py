import pickle
import tensorflow as tf 
import numpy as np
import pandas as pd
from sklearn.metrics import auc,roc_curve
import seaborn as sns
from sklearn.model_selection import train_test_split
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'


class NN():
    def __init__(self,LEARNING_RATE,sess):
        
        self.LEARNING_RATE = LEARNING_RATE
        self.tf_x = tf.placeholder(tf.float32,[None,28],name = 'x_input')
        self.tf_y = tf.placeholder(tf.int32,[None,],name = 'y_input')
        self.output = self.build_net()
        self.loss = tf.losses.sparse_softmax_cross_entropy(labels = self.tf_y,logits = self.output)
        # self.accuracy = tf.reduce_mean(
        #         tf.cast(
        #         tf.equal(tf.argmax(self.tf_y,1),tf.argmax(self.output,1)),tf.float32))
        self.accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.output,self.tf_y,1),tf.float32))
        self.train = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(self.loss)
        tf.summary.scalar('loss',self.loss)
        tf.summary.scalar('accuracy',self.accuracy)
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('logs/',sess.graph)
    def build_net(self):
        # 搭建神经网络
        with tf.name_scope('layers'):
            init = tf.contrib.layers.xavier_initializer()
            layer1 = tf.layers.dense(self.tf_x,
                                     128,
                                     tf.nn.relu,
                                     kernel_initializer=init,
                                     bias_initializer=tf.initializers.truncated_normal,
                                     name = 'layer_1')
            layer2 = tf.layers.dense(layer1,
                                     64,
                                     tf.nn.relu,
                                     kernel_initializer=init,
                                     bias_initializer=tf.initializers.truncated_normal,
                                     name = 'layer_2')
            layer3 = tf.layers.dense(layer2,
                                     64,
                                     tf.nn.relu,
                                     kernel_initializer=init,
                                     bias_initializer=tf.initializers.truncated_normal,
                                     name = 'layer_3')                         
            output = tf.layers.dense(layer3,
                                     2,
                                     None,
                                     kernel_initializer=init,
                                     bias_initializer=tf.initializers.truncated_normal,
                                     name = 'output_')
        return output





def main(xtrain,ytrain):
    # gpu配置允许内存增长
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        nn = NN(LEARNING_RATE,sess)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        for epoch in range(MAX_EPOCH):
            print('epoch:',epoch)
            loss_lst = []
            accuracy_lst = []
            for i in range(TRAIN_LEN//BATCH_SIZE): # 舍弃最后一个batch
                feed_dict={
                        nn.tf_x: xtrain[i * BATCH_SIZE:(i + 1) * BATCH_SIZE],
                        nn.tf_y: ytrain[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]}
                sess.run(nn.train,feed_dict)
                loss,accuracy = sess.run([nn.loss,nn.accuracy],feed_dict)
                loss_lst.append(loss)
                accuracy_lst.append(accuracy)
                if i%20 == 0:
                    result = sess.run(nn.merged,feed_dict)
                    nn.writer.add_summary(result,i*BATCH_SIZE+epoch*TRAIN_LEN+1)
            print("loss_lst___size:",len(loss_lst))
            print("accuracy_lst___size:",len(accuracy_lst))
            print("average training loss:", sum(loss_lst) / len(loss_lst))
            print("average accuracy:", sum(accuracy_lst) / len(accuracy_lst))
            
            
            # 保存网络
            if epoch % 10 == 0 and epoch >1:
                saver.save(sess,'Models/nnmodel_'+str(epoch)+'.ckpt')
                print('nnmodel_'+str(epoch)+'.ckpt is saved!')
        print(str(MAX_EPOCH)+' rounds trainning is over! ')
        

  
if __name__ == '__main__':

    with open('df_final.pickle','rb') as file:
        df_final = pickle.load(file)
    with open('label.pickle','rb') as file:
        label = pickle.load(file)  

    xtrain,xtest,ytrain,ytest = train_test_split(df_final,label,test_size = 0.2,random_state = 2018)
    xtrain = xtrain.reset_index(drop = True)
    xtest = xtest.reset_index(drop = True)
    ytrain = ytrain.reset_index(drop = True)
    # ytrain = ytrain[:,np.newaxis]
    ytest = ytest.reset_index(drop = True)
    # ytest = ytest[:,np.newaxis]

    # 对标签进行独热编码
    # 如果使用了独热处理数据,loss函数就使用softmax_cross_entropy
    # ytrain = pd.get_dummies(ytrain)
    # ytest = pd.get_dummies(ytest)

    # 定义一下超参数
    BATCH_SIZE = 128
    MAX_EPOCH = 1000
    TRAIN_LEN = 1869824 # xtrain.shape[0]
    LEARNING_RATE = 0.005
    # LEARNING_RATE = 0.01
    
    main(xtrain,ytrain)
