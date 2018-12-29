import pickle
import tensorflow as tf 
import numpy as np
import pandas as pd
from sklearn.metrics import auc,roc_curve,f1_score,accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

class NN():
    def __init__(self,LEARNING_RATE,sess):
        
        self.LEARNING_RATE = LEARNING_RATE
        self.tf_x = tf.placeholder(tf.float32,[None,28],name = 'x_input')
        self.tf_y = tf.placeholder(tf.int32,[None,],name = 'y_input')
        self.output = self.build_net()
        self.loss = tf.losses.sparse_softmax_cross_entropy(labels = self.tf_y,logits = self.output)
        # self.accuracy = tf.reduce_mean(
        #         tf.cast(
        #         tf.equal(self.tf_y,tf.argmax(self.output,1)),tf.float32))
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


def main(xtest,ytest):

    global predict_lst 
    global loss_lst
    global accuracy_lst
    global label_lst

    with tf.Session() as sess:
        nn = NN(LEARNING_RATE,sess)
        saver = tf.train.Saver()
        saver.restore(sess,NN_MODEL)
        print('model ' + NN_MODEL + ' restored!')
        
        feed_dict={
                nn.tf_x: xtest,
                nn.tf_y: ytest}
        loss,accuracy = sess.run([nn.loss,nn.accuracy],feed_dict)

        predict = sess.run(tf.argmax(nn.output,1),feed_dict) # (len(ytest),)

        loss_lst.append(loss)
        accuracy_lst.append(accuracy)

        predict_lst.append(predict)
            
        # print('predict_lst___size:',len(predict_lst))
        # print('predict_lst___shape:',predict_lst[0].shape)
        # print('predict_lst[0]:',predict_lst[0])
        print("average training loss:", sum(loss_lst) / len(loss_lst))
        print("average accuracy:", sum(accuracy_lst) / len(accuracy_lst))

    
            

if __name__ == '__main__':

    with open('df_final.pickle','rb') as file:
        df_final = pickle.load(file)
    with open('label.pickle','rb') as file:
        label = pickle.load(file)  

    xtrain,xtest,ytrain,ytest = train_test_split(df_final,label,test_size = 0.2,random_state = 2018)
    xtrain = xtrain.reset_index(drop = True)
    xtest = xtest.reset_index(drop = True)
    ytrain = ytrain.reset_index(drop = True) # shape (None,)
    # ytrain_ = ytrain[:,np.newaxis] # shape (None,1)
    ytest = ytest.reset_index(drop = True)
    # ytest_ = ytest[:,np.newaxis]

    # 对标签进行独热编码
    # 如果使用了独热处理数据,loss函数就使用softmax_cross_entropy
    # ytrain = pd.get_dummies(ytrain)
    # ytest = pd.get_dummies(ytest)

    # 定义一下超参数
    
    LEARNING_RATE = 0.01
    NN_MODEL = 'Models/nnmodel_30.ckpt'
    predict_lst = []
    loss_lst = []
    accuracy_lst = []
    label_lst = []

    main(xtest,ytest)


    # 绘制ROC曲线和AUC评分         
    fpr,tpr,thresholds = roc_curve(ytest,np.array(predict_lst[0])) # 要求shape = [n_samples]
    auc_score = auc(fpr,tpr)
    f1_score = f1_score(ytest,np.array(predict_lst[0]))
    print('AUC for test dataset :',auc_score)
    print('F1-score for test dataset :',f1_score)
    plt.plot(fpr,tpr,'b-')
    plt.xlabel('FPT')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.show()

   
    
