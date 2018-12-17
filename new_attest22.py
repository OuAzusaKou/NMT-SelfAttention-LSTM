# -*- coding: utf-8 -*-
#from gensim.models.keyedvectors import KeyedVectors
#from gensim.models import word2vec
import numpy as np
#from keras.layers import Embedding
import tensorflow as tf
import os  
from tensorflow.python import debug as tf_debug
##限制该代码在哪个GPU上跑，可以多个GPU
os.environ['CUDA_VISIBLE_DEVICES']='3'  
tf.device('/GPU:3') 
###隐藏结点数LSTM
num_hidden=128
##未知单词符号
UNK = "UNK"
###句子开始词
SOS = "<s>"
##句子结尾词
EOS = "</s>"
##上述符号的在字典中的ID（目标语言）
UNK_ID = 0
EOS_ID=2
SOS_ID=1
##Batch_size
batch_size=256
##金字塔结构时这个可以大于1,类比于卷积
stride=1
###同理
size=1
###是否重新训练标识
Trainnew_Flag=False
##这个是Embedding在Encoder，Decoder端的大小。
dem_num=128
enc_num=128
##学习率
learning_rate=0.1
##会对训练集做数据预处理，截取长度为MaxLenghth，不要超过150,会大大降低训练速度。
max_length=50
###源语言训练集地址
train_set_en='nmt_data/train.vi.cliped'
###目标语言训练集地址
train_set_zh='nmt_data/train.en.cliped'
##模型参数存放地址
Model_add='./parameterTT22save_8'
###使用Tensorboard进行可视化时数据地址
graph_add='./parameterTT22graph_8'
##源语言和目标语言字典地址
vocab_source='nmt_data/vocab.vi'
vocab_target='nmt_data/vocab.en'
###如果用预训练好的Wordembedding的地址
source_word_embedding='embedding_matrix.npy'
###测试数据的地址
test_set_source='nmt_data/tst2013.vi.cliped'
test_set_target='nmt_data/tst2013.en.cliped'
###这个是使用TF的dataset数据管线的Buff,依据显存容量配置。
buffer_size=batch_size*10
##训练时的Dropout
TrainDropout=0.2
###测试时的Dropout 一般都是1 。
DevDropout=1

def Std_Att_Layer(inputs,name,K,V,K_seqlen,num_hidden,atc_num,seq_len,state,dropout):
    '''
    该函数是将一个两层LSTM+Attention 的CELL做成一个按时间展开的计算图。
    其中这个CELL可以进行修改.
    '''
    time_max=tf.shape(inputs)[1]
    
    time_step=0
    layer_step=0
    '''
    将Cell按时间展开的整体思路都是先做一个TensorArray，之后用一个while块
    body中放Cell的计算图，之后通过filter过滤一下（使用where函数）
    过滤后写入TensorArray 之后TensorArray展开，进行维数变换出结果。
    这里有个可以调整的内容是，是拿前一时刻的AttentionVector送给下一个时刻，还是拿Context，目前是前者，后者可以进行测试。
    '''
    
    initcontext=array_ops.zeros([tf.shape(inputs)[0],atc_num],tf.float32)
    
    with tf.variable_scope("Create_Tensor_Array",reuse=tf.AUTO_REUSE):
            output_ta = tensor_array_ops.TensorArray(  
                          dtype=tf.float32, size=time_max)
            
            
    def body(inputs,atc,state,time_step,time_max,output_ta_t):
        tf.get_variable_scope().reuse_variables()
        #构建一个LSTM内核 输入tensor是 input的time_step 那一维度的矩阵，是一batch数据 ，和之前的状态，
        #输出一个H 和 （C，H）
        (flat_context,new_state,attention_vector) = Std_Attention_cel2layer(inputs[:,time_step,:],atc,state,num_hidden,atc_num,name,K,V,K_seqlen,dropout)
        
        print(new_state[0])
        
        with tf.variable_scope("filter",reuse=tf.AUTO_REUSE):
            copy_cond = (time_step >= seq_len)

            s11=array_ops.where(copy_cond, state[0][0], new_state[0][0])
            s12=array_ops.where(copy_cond, state[0][1], new_state[0][1])
            
            s21=array_ops.where(copy_cond, state[1][0], new_state[1][0])
            s22=array_ops.where(copy_cond, state[1][1], new_state[1][1])
            s3=array_ops.where(copy_cond, atc,attention_vector)

            new_state=[(s11,s12),(s21,s22)]
            
            new_atc=s3
            
        
        output_ta_t = output_ta_t.write(time_step, attention_vector)
        
        #out_i=out_i.write(time_step, i)  
        #out_j=out_j.write(time_step, j)  
        #out_f=out_f.write(time_step, f)  
        #out_o=out_o.write(time_step, o) 
        
        print(new_state)
        
        return inputs,attention_vector,new_state,time_step+1,time_max,output_ta_t

    def condition(inputs,atc,state,time_step,time_max,output_ta):
            
        return time_step < (time_max)
    inputs,new_atc,new_statet,time_step,time_max,output_tat=tf.while_loop(condition,body,[inputs,initcontext,state,time_step,time_max,output_ta])
    
    with tf.variable_scope('Std_at_stack',reuse=tf.AUTO_REUSE):
        
        final_outputs = output_tat.stack()
        
    final_outputs.set_shape([None,None,atc_num])
    final_outputs=tf.transpose(final_outputs,perm=[1,0,2])
    
    return final_outputs,seq_len,new_statet,new_atc

def Std_Attention_cell(inputs,prior_atc,state,num_hidden,atc_num,name,K,V,K_seqlen):
   '''
   单层的LSTM+attention的CELL，后面基本不用它了。
   '''
    c,h=state
    
    inputs_got_shape = inputs.get_shape().with_rank(2)  
    (const_batch_size,input_size) = inputs_got_shape.as_list()
    all_len=input_size+num_hidden+atc_num
    with tf.variable_scope("Std_AT_cell", reuse=tf.AUTO_REUSE):
        #创建一个(input_size+h_size)*(4*h_size)的矩阵 是要训练的权值矩阵
        weight=tf.get_variable(initializer=tf.random_uniform_initializer(-0.1,0.1),shape=[all_len,4*num_hidden],name=name)
        
        inf=array_ops.concat([inputs,h,prior_atc], 1)
        
        gate_weight=tf.matmul(inf,weight)
        #对这个向量进行四等分，得到四个向量，每个向量的长度都是H_size 分别是三个门和一个input变维到H_size大小的向量
        i, j, f, o = array_ops.split(value=gate_weight, num_or_size_splits=4, axis=1)
        
        # i = input_gate, j = new_input Trans to H_n, f = forget_gate, o = output_gate
        #计算cell
        with tf.variable_scope("new_c",reuse=tf.AUTO_REUSE):
        
            new_c = (c * sigmoid(f + 1.0) + sigmoid(i) * tanh(j))
        
        #计算H 
        with tf.variable_scope("new_h",reuse=tf.AUTO_REUSE):
        
            new_h = tanh(new_c) * sigmoid(o)
        
        new_state = (new_c, new_h)
    
        h_output=new_h
    
    context=Attention(tf.expand_dims(new_h,1),K,V,K_seqlen,1,name)
    
    flat_context=tf.layers.flatten(context)
    
    with tf.variable_scope("Std_At_project", reuse=tf.AUTO_REUSE):
    
        proj=tf.get_variable(initializer=tf.random_uniform_initializer(-0.1,0.1),shape=[2*num_hidden,num_hidden],name=name)
    
    attention_vector=tanh(tf.matmul(tf.concat([flat_context,h_output],1),proj))
    
    
    return flat_context,new_state,attention_vector

def Std_Attention_cel2layer(inputs,prior_atc,state,num_hidden,atc_num,name,K,V,K_seqlen,dropout):
    '''
    该函数是做一个两层LSTM+Attention的Cell。
    其中要改层数的话 按这个模板改就可以
    '''
    '''
    整体思路是将两个Cell按层数拼接在一起，
    同时把输入和前一时刻注意的向量拼接到一起，
    送给拼好的2层Cell得到输出H，
    输出H给Attention得到Context，后展开，
    H和展开后的Context拼接之后进行非线性变换得到AttentionVector，
    作为下一时刻需要填入的注意的向量。这里的写法兼容了金字塔结构。
    其中将多个Cell的C，H封装进list中了，方便多层cell修改。
    '''
    new_state=[]
    
    cell_input=tf.concat([inputs,prior_atc],1)
    
    h1_output,new_state1,_,_,_,_=cell_new(cell_input,state[0],num_hidden,name+'1',dropout)
    
    
    
    h2_output,new_state2,_,_,_,_=cell_new(h1_output,state[1],num_hidden,name+'2',dropout)
    
    new_state.append(new_state1)
    
    new_state.append(new_state2)
    
    print(len(new_state))
    
    context=Attention(tf.expand_dims(h2_output,1),K,V,K_seqlen,1,name)
    
    flat_context=tf.layers.flatten(context)
    
    with tf.variable_scope("Std_At_project", reuse=tf.AUTO_REUSE):
    
        proj=tf.get_variable(initializer=tf.random_uniform_initializer(-0.1,0.1),shape=[2*num_hidden,num_hidden],name=name)
    
    attention_vector=tanh(tf.matmul(tf.concat([flat_context,h2_output],1),proj))
    
    
    return flat_context,new_state,attention_vector


def dev_makegraph():
    '''
    进行测试集上计算时,计算图的构建。
    '''
    ##字典list转换成tensor
    mapping_strings_en = tf.constant(en_vocab)
    mapping_strings_vi = tf.constant(vi_vocab)
    ####将字典tensor转换成table（TF标准字典形式）
    table_sc = tf.contrib.lookup.index_table_from_tensor(
        mapping=mapping_strings_en, num_oov_buckets=0, default_value=0)


    table_tg = tf.contrib.lookup.index_table_from_tensor(
        mapping=mapping_strings_vi, num_oov_buckets=0, default_value=0)
    ###使用tf.dataset进行数据读取。分别为源语言和目标语言。
    train_dataset_sc = tf.data.TextLineDataset([test_set_source])
    train_dataset_tgin = tf.data.TextLineDataset([test_set_target])
    ####两个数据dataset按行对齐压缩成一个dataset。先压缩后预处理，不然打乱顺序时数据会乱。
    src_tgt_dataset = tf.data.Dataset.zip((train_dataset_sc,train_dataset_tgin))
    ##打乱顺序
    src_tgt_dataset=src_tgt_dataset.shuffle(buffer_size=buffer_size,reshuffle_each_iteration=True)
    ##将字符串根据字典转换成数字数组
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (tf.string_split([src]).values,
                                          tf.string_split([tgt]).values)).prefetch(buffer_size)
    ###长度异常检查
    src_tgt_dataset = src_tgt_dataset.filter(lambda src,tgt: tf.logical_and(tf.size(src)>0,tf.size(tgt)>0))
    ###长度截取
    src_tgt_dataset = src_tgt_dataset.map(lambda src,tgt: (src[0:max_length],tgt[0:max_length])).prefetch(buffer_size)
    ##目标语言进行前后添加开始和截止标识符，用于S2S模型。
    src_tgt_dataset = src_tgt_dataset.map(lambda src,tgt: (src,tf.concat(([SOS],tgt),0),tf.concat((tgt,[EOS]),0))).prefetch(buffer_size)
    
    
    '''
    f1 = lambda: tf.constant(1)
    f2 = lambda words: tf.size(words)

    train_dataset_sc = train_dataset_sc.map(lambda words: (words, tf.case([(tf.equal(tf.size(words),0),f1)],default=lambda : tf.size(words))))
    train_dataset_tgin = train_dataset_tgin.map(lambda words: (words, tf.case([(tf.equal(tf.size(words),0),f1)],default=lambda : tf.size(words))))
    train_dataset_tgout = train_dataset_tgout.map(lambda words: (words, tf.case([(tf.equal(tf.size(words),0),f1)],default=lambda : tf.size(words))))
    '''


    ##统计句子长度
    src_tgt_dataset = src_tgt_dataset.map(lambda src,tgtin,tgtout: ((src, tf.size(src)),
                                           (tgtin, tf.size(tgtin)),(tgtout, tf.size(tgtout)))).prefetch(buffer_size)



    src_tgt_dataset = src_tgt_dataset.map(lambda (src,src_size),(tgtin,tgtin_size),(tgtout,tgtout_size):
                                          ((table_sc.lookup(src), src_size),(table_tg.lookup(tgtin),tgtin_size),
                                          (table_tg.lookup(tgtout),tgtout_size))).prefetch(buffer_size)
    
    src_tgt_dataset = src_tgt_dataset.map(lambda (src,src_size),(tgtin,tgtin_size),(tgtout,tgtout_size):
                                          ((tf.one_hot(src,len(en_vocab)), src_size),(tf.one_hot(tgtin,len(vi_vocab)),tgtin_size),
                                          (tf.one_hot(tgtout,len(vi_vocab)),tgtout_size))).prefetch(buffer_size)

    
    #source_target_dataset=source_target_dataset.shuffle(buffer_size=buffer_size,reshuffle_each_iteration=True)
    print(src_tgt_dataset)

    #batched_dataset = source_target_dataset.batch(1)

    #设置dataset的取出方式，按batch取出，同时进行补零对齐。
    batched_dataset = src_tgt_dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=((tf.TensorShape([None,len(en_vocab)]),  
                            tf.TensorShape([])),     
                           (tf.TensorShape([None,len(vi_vocab)]),  
                            tf.TensorShape([])),
                          (tf.TensorShape([None,len(vi_vocab)]),  
                            tf.TensorShape([]))),   
            padding_values=((tf.constant(0,tf.float32), 
                             tf.constant(0,tf.int32)),          
                            (tf.constant(0,tf.float32),  
                             tf.constant(0,tf.int32)),
                            (tf.constant(0,tf.float32),  
                             tf.constant(0,tf.int32))))       



    #train_dataset = train_dataset.batch(1)
    #train_dataset=train_dataset.padded_batch(
    #    10,
    #    100,
    #    padding_values=None
    #)
    ###这个是dataset的必要配置，相当于为这个DATASET创建一个取数据装置。
    batched_iterator = batched_dataset.make_initializable_iterator()

    ((source, source_lengths), (targetin, targetin_lengths),(targetout, targetout_lengths)) = batched_iterator.get_next()

    next_element = ((source, source_lengths), (targetin, targetin_lengths),(targetout, targetout_lengths))

    #ids = table.lookup(next_element)
    '''
    with tf.variable_scope('emb',reuse=tf.AUTO_REUSE):
        
        embedding=tf.constant(embedding_matrix,dtype=tf.float32)
        inputs=tf.nn.embedding_lookup(embedding,source)
       '''
       #print(embedding.get_shape())
        #source=tf.cast(source,dtype=tf.float32)
        #inputs_shape=tf.shape(source)
        #flat_input=tf.reshape(source,shape=[None,inputs_shape[2]])
        #inputs=tf.matmul(flat_input,embedding)
        #inputs=tf.reshape(inputs,shape=[input_shape[0],inputs_shape[1],inputs_shape[2]])
        
        
    ##Dropout配置
    dropout = tf.placeholder(
                tf.float32, [], name='dropout')
    ##网络结构计算图的创建
    logits,seq_len,final_outputs,W,L1_seq_len,en_seq_len=train_model_graph(source,targetin,source_lengths,targetin_lengths,stride=stride,size=size,dropout=dropout)

  
    
    ###按长度将一Batch的数据的loss进行过滤，超过数据长度的loss舍弃。
    mask=tf.sequence_mask(seq_len,dtype=tf.float32)



    global_step = tf.Variable(0, trainable=False)
    with tf.name_scope('cross_entropy'):

        cross_entropy_mean = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(
                    labels=targetout, logits=logits)*mask)/tf.to_float(tf.reduce_sum(targetin_lengths))
    
    ###在可视化时候显示出来
    tf.summary.scalar("dev-loss",cross_entropy_mean)

    


        #beta1=0.9, beta2=0.999, 
                                   #       epsilon=1e-08, use_locking=False,
    tabinit=tf.tables_initializer()
    merged_summary = tf.summary.merge_all()
    dev_saver=tf.train.Saver()
    return merged_summary,batched_iterator,tabinit,dev_saver,cross_entropy_mean,dropout

def eval_makegraph():
    '''
    最后在测试集上进行算法评估计算BLEU时，计算图的构建。
    '''
    '''
    这个计算图和前面的流程一样，字典，dataset，网络结构图，但是分为两部分
    第一部分是字典,dataset，encoder端网络结构。
    第二部分是单个两层LSTM+Attention的Cell（并不是一个按时间展开的结构图，和第一部分区分）。
    在BeamSearch时会先run一次第一部分，之后将得到的数据给第二部分run 很多次，直到遇上终止符号。
    '''
    mapping_strings_en = tf.constant(en_vocab)
    mapping_strings_vi = tf.constant(vi_vocab)

    table_sc = tf.contrib.lookup.index_table_from_tensor(
        mapping=mapping_strings_en, num_oov_buckets=0, default_value=0)


    table_tg = tf.contrib.lookup.index_table_from_tensor(
        mapping=mapping_strings_vi, num_oov_buckets=0, default_value=0)

    train_dataset_sc = tf.data.TextLineDataset([test_set_source])
    train_dataset_tgin = tf.data.TextLineDataset([test_set_target])
    
    src_tgt_dataset = tf.data.Dataset.zip((train_dataset_sc,train_dataset_tgin))
    
    src_tgt_dataset=src_tgt_dataset.shuffle(buffer_size=buffer_size,reshuffle_each_iteration=True)

    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (tf.string_split([src]).values,
                                          tf.string_split([tgt]).values)).prefetch(buffer_size)
    
    src_tgt_dataset = src_tgt_dataset.filter(lambda src,tgt: tf.logical_and(tf.size(src)>0,tf.size(tgt)>0))

    src_tgt_dataset = src_tgt_dataset.map(lambda src,tgt: (src[0:100],tgt[0:100])).prefetch(buffer_size)
    
    src_tgt_dataset = src_tgt_dataset.map(lambda src,tgt: (src,tf.concat(([SOS],tgt),0),tf.concat((tgt,[EOS]),0))).prefetch(buffer_size)
    
    




    src_tgt_dataset = src_tgt_dataset.map(lambda src,tgtin,tgtout: ((src, tf.size(src)),
                                           (tgtin, tf.size(tgtin)),(tgtout, tf.size(tgtout)))).prefetch(buffer_size)



    src_tgt_dataset = src_tgt_dataset.map(lambda (src,src_size),(tgtin,tgtin_size),(tgtout,tgtout_size):
                                          ((table_sc.lookup(src), src_size),(table_tg.lookup(tgtin),tgtin_size),
                                          (table_tg.lookup(tgtout),tgtout_size))).prefetch(buffer_size)
    
    src_tgt_dataset = src_tgt_dataset.map(lambda (src,src_size),(tgtin,tgtin_size),(tgtout,tgtout_size):
                                          ((tf.one_hot(src,len(en_vocab)), src_size),(tgtin,tgtin_size),
                                          (tgtout,tgtout_size))).prefetch(buffer_size)

    
    #source_target_dataset=source_target_dataset.shuffle(buffer_size=buffer_size,reshuffle_each_iteration=True)
    print(src_tgt_dataset)

    eval_batched_dataset = src_tgt_dataset.batch(1)


    



    #train_dataset = train_dataset.batch(1)
    #train_dataset=train_dataset.padded_batch(
    #    10,
    #    100,
    #    padding_values=None
    #)

    eval_dataset_iterator = eval_batched_dataset.make_initializable_iterator()

    ((source, source_lengths), (targetin, targetin_lengths),(targetout, targetout_lengths)) = eval_dataset_iterator.get_next()

    next_element = ((source, source_lengths), (targetin, targetin_lengths),(targetout, targetout_lengths))
    
    #source_target_dataset = source_target_dataset.batch(1)
    
    #eval_dataset_iterator = source_target_dataset.make_initializable_iterator()
    
    dropout=1.0
    
    #((source,source_lengths),(target,target_lengths))=eval_dataset_iterator.get_next()
    
    _initial_state = zero_state(tf.shape(source)[0],num_hidden)
    
    '''
    with tf.variable_scope('emb',reuse=True):
        
        embedding=tf.constant(embedding_matrix,dtype=tf.float32)
        
        source=tf.nn.embedding_lookup(embedding,source)
    '''
    with tf.variable_scope("Encode_embedding",reuse=tf.AUTO_REUSE):
        
        enweight=tf.get_variable(shape=[inputs_num,enc_num],name='enc_em')
        
        source_shape=tf.shape(source)
        
        source=tf.cast(source,tf.float32)
        
        source=tf.matmul(tf.reshape(source,shape=[source_shape[0]*source_shape[1],source_shape[2]]),enweight)
        
        source_em=tf.reshape(source,shape=[source_shape[0],source_shape[1],enc_num])    
    
    
   
        
    state=_initial_state
    
    L1_result,L1_seq_len,EL1_state=LSTM_first(source_em,state,source_lengths,name='e_l1',dropout=dropout)
    
    
    L2_result,L2_seq_len,EL2_state=InterLayer_AL(Q=L1_result,K=source_em,V=source_em,
                               state=_initial_state,inputs=L1_result,sizec=size,
                                      stride=stride,seq_len=L1_seq_len,name='e_l1q',K_seqlen=source_lengths,dropout=dropout) 
    
    #en_final_outputs,en_seq_len,new_state=InterLayer_AL(Q=L1Q_result,K=L1_result,V=L1_result,
     #                           state=_initial_state,inputs=L1_result,sizec=size,
      #                                  stride=stride,seq_len=L1_seq_len,name='L2',K_seqlen=L1_seq_len,dropout=dropout)
    
    inputs=tf.placeholder(shape=[None,decode_inputnum],dtype=tf.float32)
    K=tf.placeholder(shape=[None,None,num_hidden],dtype=tf.float32)
    V=tf.placeholder(shape=[None,None,num_hidden],dtype=tf.float32)
    K_seqlen=tf.placeholder(shape=[None],dtype=tf.int32)
    PriorState=((tf.placeholder(shape=[None,num_hidden],dtype=tf.float32),tf.placeholder(shape=[None,num_hidden],dtype=tf.float32)),(tf.placeholder(shape=[None,num_hidden],dtype=tf.float32),tf.placeholder(shape=[None,num_hidden],dtype=tf.float32)))
    DL2_state=(tf.placeholder(shape=[None,num_hidden],dtype=tf.float32),tf.placeholder(shape=[None,num_hidden],dtype=tf.float32))
    prior_atc=tf.placeholder(shape=[None,num_hidden],dtype=tf.float32)
    
    with tf.variable_scope("Decode_embedding",reuse=tf.AUTO_REUSE):
        
        weight=tf.get_variable(shape=[labels_num,dem_num],name='dec_em')
        
        inputs_em=tf.matmul(inputs,weight)
    
    '''
    Qin,Q_state,i,j,f,o=cell_new(inputs_em,DL1_state,num_hidden,name='s2',dropout=dropout)
    Q=tf.expand_dims(Qin,1)
    h_output,DE_state,i,j,f,o=AttenAddCORE(Q,K,V,DL2_state,Qin,num_hidden,name='s22',sizec=1,K_seqlen=K_seqlen,dropout=dropout)
    '''
    #context,DE_state,_=Std_Attention_cell(inputs_em,prior_atc,DL2_state,num_hidden,num_hidden,'s2',K,V,K_seqlen)
    
    flat_context,new_state,attention_vector=Std_Attention_cel2layer(inputs_em,prior_atc,
                                                                        PriorState,num_hidden,num_hidden,'s2',K,V,K_seqlen,dropout)
    
    
    with tf.variable_scope("FC/FCC", reuse=tf.AUTO_REUSE):
            
            W=tf.get_variable(shape=[num_hidden,labels_num],name='kernel')
            
    logit=tf.matmul(attention_vector,W)
    
    logit=tf.nn.softmax(logits=logit)
    
    tabinit=tf.tables_initializer()
    
    eval_saver=tf.train.Saver()
    
    return inputs,K,V,K_seqlen,PriorState,logit,EL1_state,EL2_state,prior_atc,eval_dataset_iterator,eval_saver,tabinit,targetout,attention_vector,new_state,L2_result,L2_seq_len
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
def beam_search(sess,inputs,K,V,K_seqlen,PriorState,logit,EL1_state,EL2_state,prior_atc,
eval_dataset_iterator,eval_saver,tabinit,
targetout,attention_vector,new_state,L2_result,L2_seq_len):
    '''
    BeamSearch。。未实现，使用了贪心的方法。
    过程是run eval图的第一部分一次，之后run第二部分很多次，直到出现截止符。
    '''
    
    
    '''
    _initial_state = zero_state(tf.shape(source)[0],num_hidden)
        
    state=_initial_state
    
    L1_result,L1_seq_len,_=LSTM_first(source,state,source_lengths,name='e_l1')
    
    L1Q_result,L1Q_seq_len,_=LSTM_first(L1_result,state,L1_seq_len,name='e_l1q')
    
    en_final_outputs,en_seq_len,new_state=InterLayer_AL(Q=L1Q_result,K=L1_result,V=L1_result,
                                state=_initial_state,inputs=L1_result,sizec=size,
                                        stride=stride,seq_len=L1_seq_len,name='L2',K_seqlen=L1_seq_len)
    
    inputs=tf.placeholder(shape=[None,input_length])
    K=tf.placeholder(shape=[None,None,num_hidden])
    V=tf.placeholder(shape=[None,None,num_hidden])
    K_seqlen=tf.placeholder(shape=[None])
    DL1_state=tf.placeholder(shape=[None,num_hidden])
    DL2_state=tf.placeholder(shape=[None,num_hidden])
    
    Q,Q_state,i,j,f,o=cell_new(inputs,DL1_state,num_hidden,name='s2')
    
    h_output,DE_state,i,j,f,o=AttenAddCORE(Q,K,V,DL2_state,inputs,num_hidden,name='s22',sizec=1,K_seqlen=K_seqlen)
    
    with tf.variable_scope("FCC", reuse=tf.AUTO_REUSE):
            
            W=tf.get_variable(shape=[num_hidden,labels_num],name='kernel')
            
    logit=tf.matmul(h_output,W)
    
    logit=tf.nn.softmax(logits=logit)
    
    '''
        
    EN1_state,EN2_state,eval_target,en_final_outputs,en_seq_len=sess.run([EL1_state,EL2_state,targetout,L2_result,L2_seq_len])
        
    sos=np.zeros((1,labels_num))
        
    sos[:,SOS_ID]=1
    
    pr_state=(EN1_state,EN2_state)
    
    
    s_logit,next_state,next_attention=sess.run([logit,new_state,attention_vector],{inputs:sos,K:en_final_outputs,
                                                                        V:en_final_outputs,K_seqlen:en_seq_len,
                                                                        PriorState:(pr_state),
                                                                        prior_atc:np.zeros((1,num_hidden))})
    result=[]
    
    de_length=0
    while True:
            
        de_length=de_length+1
            
        num_input=np.argmax(s_logit,axis=1)
            
        result.extend(num_input.tolist())
            
        if((num_input[0]==EOS_ID)or(de_length>50)):
            
            return result,eval_target
                
            break
            
        next_input=np.zeros((1,labels_num))
            
        next_input[:,num_input]=1
    
        s_logit,next_state,next_attention=sess.run([logit,new_state,attention_vector],{inputs:next_input,K:en_final_outputs,
                                                                        V:en_final_outputs,K_seqlen:en_seq_len,
                                                                        PriorState:next_state,
                                                                        prior_atc:next_attention})
        #print(result)    
    
def train_makegraph():
    '''
    训练时计算图的构建
    '''
    '''
    图的前面和之前一样，区别是创建完网络图和Loss后，加入优化算法更新权值，使用的是SGD（Adam无法得到正确的解），学习率，和梯度裁剪。
    '''
    '''
    embedding_layer = Embedding(len(word_index) + 1,
                                    200,
                                    weights=[embedding_matrix],
                                    input_length=100,
                                    trainable=False)
    '''
    mapping_strings_en = tf.constant(en_vocab)
    mapping_strings_vi = tf.constant(vi_vocab)

    table_sc = tf.contrib.lookup.index_table_from_tensor(
        mapping=mapping_strings_en, num_oov_buckets=0, default_value=0)


    table_tg = tf.contrib.lookup.index_table_from_tensor(
        mapping=mapping_strings_vi, num_oov_buckets=0, default_value=0)

    train_dataset_sc = tf.data.TextLineDataset([train_set_en])
    train_dataset_tgin = tf.data.TextLineDataset([train_set_zh])
    
    src_tgt_dataset = tf.data.Dataset.zip((train_dataset_sc,train_dataset_tgin))
    
    src_tgt_dataset=src_tgt_dataset.shuffle(buffer_size=buffer_size,reshuffle_each_iteration=True)

    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (tf.string_split([src]).values,
                                          tf.string_split([tgt]).values)).prefetch(buffer_size)
    
    src_tgt_dataset = src_tgt_dataset.filter(lambda src,tgt: tf.logical_and(tf.size(src)>0,tf.size(tgt)>0))

    src_tgt_dataset = src_tgt_dataset.map(lambda src,tgt: (src[0:max_length],tgt[0:max_length])).prefetch(buffer_size)
    
    src_tgt_dataset = src_tgt_dataset.map(lambda src,tgt: (src,tf.concat(([SOS],tgt),0),tf.concat((tgt,[EOS]),0))).prefetch(buffer_size)
    
    
    '''
    f1 = lambda: tf.constant(1)
    f2 = lambda words: tf.size(words)

    train_dataset_sc = train_dataset_sc.map(lambda words: (words, tf.case([(tf.equal(tf.size(words),0),f1)],default=lambda : tf.size(words))))
    train_dataset_tgin = train_dataset_tgin.map(lambda words: (words, tf.case([(tf.equal(tf.size(words),0),f1)],default=lambda : tf.size(words))))
    train_dataset_tgout = train_dataset_tgout.map(lambda words: (words, tf.case([(tf.equal(tf.size(words),0),f1)],default=lambda : tf.size(words))))
    '''



    src_tgt_dataset = src_tgt_dataset.map(lambda src,tgtin,tgtout: ((src, tf.size(src)),
                                           (tgtin, tf.size(tgtin)),(tgtout, tf.size(tgtout)))).prefetch(buffer_size)



    src_tgt_dataset = src_tgt_dataset.map(lambda (src,src_size),(tgtin,tgtin_size),(tgtout,tgtout_size):
                                          ((table_sc.lookup(src), src_size),(table_tg.lookup(tgtin),tgtin_size),
                                          (table_tg.lookup(tgtout),tgtout_size))).prefetch(buffer_size)
    
    src_tgt_dataset = src_tgt_dataset.map(lambda (src,src_size),(tgtin,tgtin_size),(tgtout,tgtout_size):
                                          ((tf.one_hot(src,len(en_vocab)), src_size),(tf.one_hot(tgtin,len(vi_vocab)),tgtin_size),
                                          (tf.one_hot(tgtout,len(vi_vocab)),tgtout_size))).prefetch(buffer_size)

    
    #source_target_dataset=source_target_dataset.shuffle(buffer_size=buffer_size,reshuffle_each_iteration=True)
    print(src_tgt_dataset)

    #batched_dataset = source_target_dataset.batch(1)


    batched_dataset = src_tgt_dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=((tf.TensorShape([None,len(en_vocab)]),  
                            tf.TensorShape([])),     
                           (tf.TensorShape([None,len(vi_vocab)]),  
                            tf.TensorShape([])),
                          (tf.TensorShape([None,len(vi_vocab)]),  
                            tf.TensorShape([]))),   
            padding_values=((tf.constant(0,tf.float32), 
                             tf.constant(0,tf.int32)),          
                            (tf.constant(0,tf.float32),  
                             tf.constant(0,tf.int32)),
                            (tf.constant(0,tf.float32),  
                             tf.constant(0,tf.int32))))          



    #train_dataset = train_dataset.batch(1)
    #train_dataset=train_dataset.padded_batch(
    #    10,
    #    100,
    #    padding_values=None
    #)

    batched_iterator = batched_dataset.make_initializable_iterator()

    ((source, source_lengths), (targetin, targetin_lengths),(targetout, targetout_lengths)) = batched_iterator.get_next()

    next_element = ((source, source_lengths), (targetin, targetin_lengths),(targetout, targetout_lengths))

    #ids = table.lookup(next_element)
    '''
    with tf.variable_scope('emb',reuse=tf.AUTO_REUSE):
        
        embedding=tf.constant(embedding_matrix,dtype=tf.float32)
        inputs=tf.nn.embedding_lookup(embedding,source)
       '''
       #print(embedding.get_shape())
        #source=tf.cast(source,dtype=tf.float32)
        #inputs_shape=tf.shape(source)
        #flat_input=tf.reshape(source,shape=[None,inputs_shape[2]])
        #inputs=tf.matmul(flat_input,embedding)
        #inputs=tf.reshape(inputs,shape=[input_shape[0],inputs_shape[1],inputs_shape[2]])

    dropout = tf.placeholder(
                tf.float32, [], name='dropout')
    logits,seq_len,final_outputs,W,L1_seq_len,en_seq_len=train_model_graph(source,targetin,source_lengths,targetin_lengths,stride=stride,size=size,dropout=dropout)

    mask=tf.sequence_mask(seq_len,dtype=tf.float32)



    global_step = tf.Variable(0, trainable=False)
    with tf.name_scope('cross_entropy'):

        cross_entropy_mean = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(
                    labels=targetout, logits=logits)*mask)/batch_size
        #tf.to_float(tf.reduce_sum(targetin_lengths))
    params = tf.trainable_variables()
    ##计算梯度
    gradients = tf.gradients(cross_entropy_mean, params)
    ##梯度裁剪
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(
        gradients, clip_norm=5)
    tf.summary.scalar('grad_norm',gradient_norm)
    tf.summary.scalar("loss",cross_entropy_mean)

    

    with tf.variable_scope('train',reuse=tf.AUTO_REUSE):

        learning_rate_input = tf.placeholder(
                tf.float32, [], name='learning_rate_input')
	tf.summary.scalar("Learning_rate",learning_rate_input)
    ##SGD更新权值
        optimizer =tf.train.GradientDescentOptimizer(learning_rate=learning_rate_input,  name='sgd').apply_gradients(
            zip(clipped_gradients, params))
        #beta1=0.9, beta2=0.999, 
                                   #       epsilon=1e-08, use_locking=False,
    tabinit=tf.tables_initializer()
    merged_summary = tf.summary.merge_all()
    train_saver=tf.train.Saver()
    return learning_rate_input,merged_summary,optimizer,batched_iterator,tabinit,train_saver,dropout
def LSTM_first(inputs,state,seq_len,name,dropout):
    '''
    一个单层的LSTM按时间展开后计算图的构建。
    '''
    time_step=0
    
    time_max=tf.shape(inputs)[1]
    
    with tf.variable_scope("Create_Tensor_Array",reuse=tf.AUTO_REUSE):
            output_ta = tensor_array_ops.TensorArray(  
                          dtype=tf.float32, size=time_max)  
            
            out_i = tensor_array_ops.TensorArray(  
                          dtype=tf.float32, size=time_max)  
            
            out_j = tensor_array_ops.TensorArray(  
                          dtype=tf.float32, size=time_max)  
            out_f = tensor_array_ops.TensorArray(  
                          dtype=tf.float32, size=time_max)  
            out_o = tensor_array_ops.TensorArray(  
                          dtype=tf.float32, size=time_max)  
    def body(inputs,state,time_step,time_max,output_ta_t,out_i,out_j,out_f,out_o):
        tf.get_variable_scope().reuse_variables()
        #构建一个LSTM内核 输入tensor是 input的time_step 那一维度的矩阵，是一batch数据 ，和之前的状态，
        #输出一个H 和 （C，H）
        (cell_output, new_state,i,j,f,o) = cell_new(inputs[:, time_step, :], state,num_hidden,name,dropout=dropout)
        
        with tf.variable_scope("filter",reuse=tf.AUTO_REUSE):
            copy_cond = (time_step >= seq_len)

            s1=array_ops.where(copy_cond, state[0], new_state[0])
            s2=array_ops.where(copy_cond, state[1], new_state[1])

            new_state=(s1,s2)
            
            
        output_ta_t = output_ta_t.write(time_step, new_state[1])  
        out_i=out_i.write(time_step, i)  
        out_j=out_j.write(time_step, j)  
        out_f=out_f.write(time_step, f)  
        out_o=out_o.write(time_step, o)  
        
        return inputs,new_state,time_step+1,time_max,output_ta_t,out_i,out_j,out_f,out_o

    def condition(inputs,state,time_step,time_max,output_ta,out_i,out_j,out_f,out_o):
            
        return time_step < (time_max)
    inputs,state,time_step,time_max,output_tat,out_i,out_j,out_f,out_o=tf.while_loop(condition,body,[inputs,
                                                                                       state,
                                                                                       time_step,time_max,output_ta,
                                                                                  out_i,out_j,out_f,out_o])
    with tf.variable_scope('first_stack',reuse=tf.AUTO_REUSE):
        
        final_outputs = output_tat.stack()
        
    final_outputs.set_shape([None,None,num_hidden])
    final_outputs=tf.transpose(final_outputs,perm=[1,0,2])

    with tf.variable_scope("RNN", reuse=True):
        #创建一个(input_size+h_size)*(4*h_size)的矩阵 是要训练的权值矩阵
        weight=tf.get_variable(name=name)
    tf.summary.histogram('lstm/'+name,weight)
    
    return final_outputs,seq_len,state
def train_model_graph(source,target,source_lengths,target_lengths,stride,size,dropout):
    '''
    训练时用到的模型的计算图的构建（是网络模型不是训练时的整体计算图）。
    '''
    '''
    网络图构造是encoder：一层wordembedding，一层普通LSTM，一层Self-Attention的LSTM。
            decoder：一层wordembedding，一个两层LSTM+Attention，加一层全连接。
    '''
    
    with tf.variable_scope("Init_State",reuse=tf.AUTO_REUSE):
        _initial_state = zero_state(tf.shape(source)[0],num_hidden)
        
    state=_initial_state
    
    time_step=0
    outputs=[]
    
    with tf.variable_scope("Decode_embedding",reuse=tf.AUTO_REUSE):
        
        de_weight=tf.get_variable(initializer=tf.random_uniform_initializer(-0.1,0.1),shape=[labels_num,dem_num],name='dec_em')
        
        target_shape=tf.shape(target)
        
        target=tf.cast(target,tf.float32)
        
        target=tf.matmul(tf.reshape(target,shape=[target_shape[0]*target_shape[1],target_shape[2]]),de_weight)
        
        target_em=tf.reshape(target,shape=[target_shape[0],target_shape[1],dem_num])
        
    with tf.variable_scope("Encode_embedding",reuse=tf.AUTO_REUSE):
        
        enweight=tf.get_variable(initializer=tf.random_uniform_initializer(-0.1,0.1),shape=[inputs_num,enc_num],name='enc_em')
        
        source_shape=tf.shape(source)
        
        source=tf.cast(source,tf.float32)
        
        source=tf.matmul(tf.reshape(source,shape=[source_shape[0]*source_shape[1],source_shape[2]]),enweight)
        
        source_em=tf.reshape(source,shape=[source_shape[0],source_shape[1],enc_num])
        
    ###LSTM1
    L1_result,L1_seq_len,EL1_state=LSTM_first(source_em,state,source_lengths,name='e_l1',dropout=dropout)
    
    
    #L2_result,L2_seq_len,EL2_state=LSTM_first(L1_result,state,L1_seq_len,name='e_l1q',dropout=dropout)
    
    ##Self-Attention的LSTM
    L2_result,L2_seq_len,EL2_state=InterLayer_AL(Q=L1_result,K=source_em,V=source_em,
                               state=_initial_state,inputs=L1_result,sizec=size,
                                      stride=stride,seq_len=L1_seq_len,name='e_l1q',K_seqlen=source_lengths,dropout=dropout)   
    
    ###S2S train####
    ##Q dimension equal Input dimension####
    '''
    L2_result,L2_seq_len,DE_State=LSTM_first(target_em,new_state,target_lengths,name='s2',dropout=dropout)
    
    final_outputs,final_seq_len,_=InterLayer_AL(Q=L2_result,K=en_final_outputs,V=en_final_outputs,
                                state=EL1_state,inputs=L2_result,sizec=1,
                                        stride=1,seq_len=L2_seq_len,name='s22',K_seqlen=en_seq_len,dropout=dropout) 
    '''
    ##Decoder端
    final_outputs,final_seq_len,final_state,new_atc=Std_Att_Layer(inputs=target_em,name='s2',K=L2_result,V=L2_result,
                   K_seqlen=L2_seq_len,num_hidden=num_hidden,atc_num=num_hidden,seq_len=target_lengths,state=[EL1_state,EL2_state],dropout=dropout)
    #final_outputs,final_seq_len,_=LSTM_first(target_em,EL1_state,target_lengths,name='e_l1q',dropout=dropout)
    
    
    time_max=tf.shape(final_outputs)[1]
    out_final_tensor= final_outputs
    '''
    with tf.variable_scope("FC", reuse=tf.AUTO_REUSE):    
        
        W = tf.get_variable(shape=[num_hidden,labels_num],name='W')

        b = tf.get_variable(shape=[labels_num], name='b')
        
        logits = tf.matmul(out_final_tensor, W) + b
    '''
    ##全连接
    with tf.variable_scope("FC", reuse=tf.AUTO_REUSE):
        
        #projection_layer = tf.layers.Dense(labels_num, use_bias=False,name='FFC')
        #print(projection_layer.weights,'weight')
        logits=tf.layers.dense(use_bias=False,kernel_initializer=tf.random_uniform_initializer(-0.1,0.1),inputs=out_final_tensor,units=labels_num,name='FCC')
        with tf.variable_scope("FCC", reuse=tf.AUTO_REUSE):
            
            W=tf.get_variable(shape=[num_hidden,labels_num],name='kernel')

        tf.summary.histogram('FCout',W)
    return logits,final_seq_len,final_outputs,W,L1_seq_len,final_seq_len
from tensorflow.python.ops import tensor_array_ops 
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
sigmoid = math_ops.sigmoid
tanh=math_ops.tanh
#构建一个LSTM内核的计算图 输入tensor是input 和状态（C，H），返回是H 和 （C，H

#只有第一次调用时创建全值矩阵，二次调用时会使用权值构建一个新的图
#    cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, state_is_tuple=True)
# _, outs = tf.nn.dynamic_rnn(cell, inputs, seq_len, dtype=tf.float32)
def cell_new(inputs,state,num_hidden,name,dropout):
    
    '''
    一个LSTM单元的计算图构建，称为一个Cell。
    '''

    c,h=state
    inputs_got_shape = inputs.get_shape().with_rank(2)  
    (const_batch_size,input_size) = inputs_got_shape.as_list()
    all_len=input_size+num_hidden
    #print(all_len)
    #print(h.get_shape())
    with tf.variable_scope("RNN", reuse=tf.AUTO_REUSE):
        #创建一个(input_size+h_size)*(4*h_size)的矩阵 是要训练的权值矩阵
        weight=tf.get_variable(initializer=tf.random_uniform_initializer(-0.1,0.1),shape=[all_len,4*num_hidden],name=name)
        
        #先将Input和H 拼接成一个向量 之后和矩阵相乘得到一个向量
        inf=array_ops.concat([tf.nn.dropout(inputs,keep_prob=dropout),h], 1)
        print(inf.get_shape())
        gate_weight=tf.matmul(inf,weight)
        #对这个向量进行四等分，得到四个向量，每个向量的长度都是H_size 分别是三个门和一个input变维到H_size大小的向量
        i, j, f, o = array_ops.split(value=gate_weight, num_or_size_splits=4, axis=1)
        
        # i = input_gate, j = new_input Trans to H_n, f = forget_gate, o = output_gate
        #计算cell
        with tf.variable_scope("new_c",reuse=tf.AUTO_REUSE):
        
            new_c = (c * sigmoid(f + 1.0) + sigmoid(i) * tanh(j))
        #计算H 
        with tf.variable_scope("new_h",reuse=tf.AUTO_REUSE):
            new_h = tanh(new_c) * sigmoid(o)
        
        new_state = (new_c, new_h)
    
        h_output=new_h

    	#tf.summary.histogram("lstm/"+name,weight)
    
    return h_output,new_state,i,j,f,o

#构建一个图，输入tensor是input的batch_size大小，返回一个全是零的初始状态（C，H）=（zero，zero）
def zero_state(input_batch,num_hidden):
    '''
    用于创造一个LSTM初始状态的Tensor。
    '''
    i_c=array_ops.zeros([input_batch,num_hidden],tf.float32)
    i_h=array_ops.zeros([input_batch,num_hidden],tf.float32)
    
    init_state=(i_c,i_h)
    
    return init_state
#制作整体LSTM的图

def Attention(Q,K,V,K_seqlen,sizec,name):
    '''
    Attention函数，包括Score的计算，加权和的计算。以计算图的形式表现。
    '''
    
    print(Q.get_shape())
    with tf.variable_scope("Attention"+name, reuse=tf.AUTO_REUSE):
        
        ##对K乘上一个矩阵变成keys，（K，V一般是一样的）
        keys=tf.layers.dense(use_bias=False,kernel_initializer=tf.random_uniform_initializer(-0.1,0.1),
                             inputs=K,units=num_hidden,name=name)
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            
            W=tf.get_variable(shape=[num_hidden,num_hidden],name='kernel')
            ##一个标量，用于调整
            scale=tf.get_variable(shape=(),name='scale'+name,initializer=tf.ones_initializer)
    
    ##Q和keys做矩阵乘，得到score
    index=tf.matmul(Q,tf.transpose(keys,perm=[0,2,1]))
    ##对score进行一个放缩，系数可训练，这个系数如果很大会使得归一化后各个内容的得分区别很大。，如果小 会使得区分很小。初始是1。
    index_scale=scale*index
    
    time_max = tf.shape(K)[1]
    '''
    在对时间进行softmax时需要进行mask，填充数据，超过数据长度的位置会被填充-inf 这样在softmax得分就是0。
    '''
    buffer=tf.constant(value=1,shape=[1,sizec])
    
    seq_len=tf.matmul(tf.transpose(tf.expand_dims(K_seqlen,0),perm=[1,0]),buffer)
    
    score_mask=tf.sequence_mask(seq_len)
    
    mask_values=(-float('inf'))*tf.ones_like(index_scale)
    
    score_masked=tf.where(score_mask,index_scale,mask_values)
    '''
    out_scale = tensor_array_ops.TensorArray(  
                          dtype=tf.float32, size=time_max)  
    time_step=0
    seq_len=K_seqlen
    def body(time_step,out_scale,index_scale,seq_len,time_max):
    
        copy_cond = (time_step >= seq_len)

        s1=array_ops.where(copy_cond,tf.zeros(shape=[tf.shape(Q)[0],tf.shape(Q)[1]],dtype=tf.float32)+(-float('inf')),
                           index_scale[:,:,time_step])
        
        #index_scale[:,:,time_step]
        #tf.constant(value=-float('inf'),shape=[tf.shape(Q)[0],tf.shape(Q)[1]])
        out_scale=out_scale.write(time_step,s1)
            
        return time_step+1,out_scale,index_scale,seq_len,time_max
    def condition(time_step,out_scale,index_scale,seq_len,time_max):
        
        return time_step<time_max
        
    time_step,out_scale,index_scale,seq_len,time_max=tf.while_loop(condition,body,[time_step,out_scale,
                                                                          index_scale,seq_len,time_max])
    with tf.variable_scope('att'): 
        final=out_scale.stack()
    
    final.set_shape([None,None,sizec])
    
    final=tf.transpose(final,perm=[1,2,0])
    '''
    
    
    context=tf.matmul(tf.nn.softmax(score_masked),V)
    
    return context
def AttenAddCORE(Q,K,V,state,inputs,num_hidden,name,sizec,K_seqlen,dropout):
    '''
    在Self-Attention中使用的LSTM+Attention的一个Cell的计算图。
    '''
    
    context=Attention(Q,K,V,K_seqlen,sizec,name)
    context.set_shape([None,sizec,None])
    #print(context.get_shape())
    
    flat_context=tf.layers.flatten(context)
    
    print(flat_context.get_shape())
    
    inputs=array_ops.concat([inputs,flat_context], 1)
    
    print(inputs.get_shape(),'input')
    
    h_output,new_state,i,j,f,o=cell_new(inputs,state,num_hidden,name,dropout)
    
    return h_output,new_state,i,j,f,o

def InterLayer_AL(Q,K,V,state,inputs,sizec,stride,seq_len,name,K_seqlen,dropout):
    '''
    Self-Attention时将封装好Attention的LSTM的cell进行按时间展开的计算图。
    目前的Self-Attention支持金字塔结构，实验可以跑 可以训练，但是效果没测试。
    '''
    
    time_max=tf.shape(inputs)[1]
    #inputs=tf.concat([inputs,0],1)
    time_step=0
    layer_step=0
    H_outarray=tensor_array_ops.TensorArray(  
                          dtype=tf.float32, size=(tf.div(time_max,stride)+tf.cast(tf.not_equal(tf.mod(time_max,stride),0),
                                                                                   tf.int32)))  
    out_i = tensor_array_ops.TensorArray(  
                          dtype=tf.float32, size=time_max)  
            
    out_j = tensor_array_ops.TensorArray(  
                          dtype=tf.float32, size=time_max)  
    out_f = tensor_array_ops.TensorArray(  
                          dtype=tf.float32, size=time_max)  
    out_o = tensor_array_ops.TensorArray(  
                          dtype=tf.float32, size=time_max)  
    ##########或许不应该time_max 补零 应该是 seq_len补零，但现在没关系########
    mod=sizec
    zeros=tf.zeros([tf.shape(Q)[0],mod,tf.shape(Q)[2]])
    Q=tf.concat([Q,zeros],axis=1)
    
    
    
    def body(Q,K,V,state,inputs,seq_len,time_step,time_max,H_outarray,out_i,out_j,out_f,out_o,layer_step):
        tf.get_variable_scope().reuse_variables()
            #构建一个LSTM内核 输入tensor是 input的time_step 那一维度的矩阵，是一batch数据 ，和之前的状态，
            #输出一个H 和 （C，H）
        
        (cell_output, new_state,i,j,f,o) = AttenAddCORE(Q[:,time_step:time_step+sizec,:],
                                                        K,V,state,
                                                        inputs[:, time_step, :],
                                                        num_hidden,name,sizec,K_seqlen,dropout=dropout)
            #print(cell_output)
        #K=tf.concat([K,cell_output],2)
        #V=tf.concat([V,cell_output],2)
        
            #outputs.append(cell_output)
            #outputs.append(cell_output)
            #一个过滤器，当 当前时间大于等于输入给定的长度时，输出状态是和上一个时间一样的状态，如果小于，则正常更新状态
        with tf.variable_scope("filter",reuse=tf.AUTO_REUSE):
            copy_cond = (time_step >= seq_len)

            s1=array_ops.where(copy_cond, state[0], new_state[0])
            s2=array_ops.where(copy_cond, state[1], new_state[1])

            new_state=(s1,s2)
            #将过滤后的状态H写入tensorArray的当前时刻处    
        H_outarray = H_outarray.write(layer_step, new_state[1])  
        #out_i=out_i.write(time_step, i)  
        #out_j=out_j.write(time_step, j)  
        #out_f=out_f.write(time_step, f)  
        #out_o=out_o.write(time_step, o)  
        
        return Q,K,V,new_state,inputs,seq_len,time_step+stride,time_max,H_outarray,out_i,out_j,out_f,out_o,layer_step+1
    
    def condition(Q,K,V,new_state,inputs,seq_len,time_step,time_max,H_outarray,out_i,out_j,out_f,out_o,layer_step):
            
        return layer_step < (tf.div(time_max,stride)+tf.cast(tf.not_equal(tf.mod(time_max,stride),0),
                                                                                   tf.int32))  
    Q,K,V,new_state,inputs,seq_len,time_step,time_max,H_outarray_t,out_i,out_j,out_f,out_o,layer_step=tf.while_loop(condition,
                                                                                    body,[Q,K,V,state,inputs,
                                                                                          seq_len,
                                                                                          time_step,
                                                                                          time_max,H_outarray,
                                                                                    out_i,out_j,out_f,out_o,layer_step])
    print(H_outarray_t)
    with tf.variable_scope('RNN',reuse=True):
	weight=tf.get_variable(name=name)
    tf.summary.histogram('lstm/'+name,weight)
    with tf.variable_scope("Attention"+name, reuse=True):
        
        with tf.variable_scope(name, reuse=True):
            
            W=tf.get_variable(shape=[num_hidden,num_hidden],name='kernel')
            
    tf.summary.histogram('Attention/'+name,W)
        

    with tf.variable_scope('interlayer_stack',reuse=tf.AUTO_REUSE):
        
        final_outputs = H_outarray_t.stack()  
    #set and reshape 's really difference unknow
    #final_outputs.set_shape([const_time_steps,const_batch_size,num_hidden]) 
    final_outputs.set_shape([None,None,num_hidden])
    final_outputs=tf.transpose(final_outputs,perm=[1,0,2])
    seq_len=tf.cast(tf.not_equal(tf.mod(seq_len,stride),0),tf.int32)+tf.div(seq_len,stride)
    
    return final_outputs,seq_len,new_state
if __name__ == '__main__':

    ########word_index制作过程#######
    '''
    model = KeyedVectors.load("text8.model")

    vocab = model.wv.vocab.keys()

    vocab_len = len(vocab)
    words=list(vocab)
    print(words[51])
    inde=range(len(words))
    word_index=dict(zip(words,inde))
    print(len(word_index))
    np.save('word_index.npy',word_index)
    '''
    #word_index=np.load('word_index.npy').item()

    ######制作矩阵过程index-embeddings##########
    '''
    embeddings_index=model
    embedding_matrix = np.zeros((len(word_index) + 1, 200))
    for word, i in word_index.items():
        embedding_vector = embeddings_index[word]
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    np.save('embedding_matrix.npy',embedding_matrix)
    '''
    ###制作矩阵的过程index-embeddings###
    #embedding_matrix=np.load('embedding_matrix.npy')
    ##源语言字典 可以任意改。
    en_vocab=[]
    with open(vocab_source,'r') as file:
        while True:
            word=file.readline().strip()
            if not word:
                break
            else:
                en_vocab.append(word)
    ##目标语言字典 可以改。
    vi_vocab=[]
    with open(vocab_target,'r') as file:
        while True:
            word=file.readline().strip()
            if not word:
                break
            else:
                vi_vocab.append(word)
    ##目标语言字典长度和源语言字典长度。
    labels_num=len(vi_vocab)
    inputs_num=len(en_vocab)
    decode_inputnum=labels_num
    ####训练时计算图，评估BLEU时计算图，测试时计算图
    train_graph = tf.Graph()
    eval_graph = tf.Graph()
    dev_graph = tf.Graph()
    
    ###GPU显存的占用限制，按需分配单卡显存。
    gpu_options=tf.GPUOptions(allow_growth=True)       
  #,config=tf.ConfigProto(gpu_options=gpu_options)
#,config=tf.ConfigProto(gpu_options=gpu_options)
    ###将计算图封入一个会话中，分别对训练，评估，和测试封入三个不同的会话。
    train_sess = tf.Session(graph=train_graph,config=tf.ConfigProto(gpu_options=gpu_options))
    
    eval_sess = tf.Session(graph=eval_graph,config=tf.ConfigProto(gpu_options=gpu_options))
    dev_sess = tf.Session(graph=dev_graph,config=tf.ConfigProto(gpu_options=gpu_options))

    ###训练计算图的创建。
    with train_graph.as_default():
        learning_rate_input,merged_summary,optimizer,batched_iterator,tabinit,train_saver,train_dropout=train_makegraph()
        initializer = tf.global_variables_initializer()
        writer = tf.summary.FileWriter(graph_add, train_sess.graph)
    ###评估计算图的创建
    with eval_graph.as_default():
        inputs,K,V,K_seqlen,PriorState,logit,EL1_state,EL2_state,prior_atc,eval_dataset_iterator,eval_saver,eval_tabinit,targetout,attention_vector,new_state,L2_result,L2_seq_len=eval_makegraph()

    ###测试计算图的创建
    with dev_graph.as_default():
        
        dev_merged_summary,dev_batched_iterator,dev_tabinit,dev_saver,dev_loss,dev_dropout=dev_makegraph()
        dev_writer = tf.summary.FileWriter(graph_add)
    ##计算图初始化
    train_sess.run(batched_iterator.initializer)
    train_sess.run(initializer)
    train_sess.run(tabinit)

    eval_sess.run(eval_tabinit)
    eval_sess.run(eval_dataset_iterator.initializer)
    
    dev_sess.run(dev_tabinit)
    dev_sess.run(dev_batched_iterator.initializer)
    
    epoch=0
    ###检测当前是否重新训练，如果不重新训练则从地址读取权值。
    if not Trainnew_Flag:

        train_saver.restore(train_sess,Model_add+'./'+'model.ckpt')
    
    for i in range(10000000):
    #if i % 1000 ==0:
    #if learning_rate > (0.001):
    #learning_rate=learning_rate*0.5
    
    #开始训练每次训练一个Epoch后 计数+1
        try:
            summary,opt=train_sess.run([merged_summary,optimizer],{
            learning_rate_input:learning_rate,train_dropout:TrainDropout})
        except tf.errors.OutOfRangeError:
            epoch+=1
            print("epoch finished at %d."%epoch)
            train_sess.run(batched_iterator.initializer) 
            continue 
        ###训练次数是10 倍数就存储参数并且做一次测试
        if (i % 10) == 0:

            train_saver.save(train_sess,Model_add+'./'+'model.ckpt')

            writer.add_summary(summary,i)
            writer.flush()
            
            dev_saver.restore(dev_sess,Model_add+'./'+'model.ckpt')
            
            try:
                
                dev_summary,_=dev_sess.run([dev_merged_summary,dev_loss],{dev_dropout:DevDropout})
                
                dev_writer.add_summary(dev_summary,i)
                
                writer.flush
            
            except tf.errors.OutOfRangeError:
                
                dev_sess.run(dev_batched_iterator.initializer) 
                
                continue
            
            ###如果100倍数 则做一次测试集上的BeamSearch，并且将翻译结果写入文件。方便后续用标准BLEU函数对其进行BLEU评估。（自己写BLEU的发现结果不一致）
            if (i % 100) == 0:
                score=0
                eval_saver.restore(eval_sess,Model_add+'./'+'model.ckpt')
                swrdoc= open('result.s2','w')
                twrdoc= open('result.g2','w')
                eval_sess.run(eval_dataset_iterator.initializer)
                for j in range (1000):
                    try:
                        result,eval_target=beam_search(eval_sess,inputs,K,V,K_seqlen,PriorState,logit,EL1_state,EL2_state,prior_atc,
    eval_dataset_iterator,eval_saver,tabinit,
    targetout,attention_vector,new_state,L2_result,L2_seq_len)

                        standard=eval_target.tolist()[0]
                        re_string=[]
                        for word in result:
                            if word == 2 :
                                continue
            #print(vi_vocab[word])
                            re_string.append(vi_vocab[word])

        #print(re_string)
                        restring=' '.join(re_string)
                        swrdoc.write(restring+'\n')
                        re_string=[]
                        for word in standard:
                            try:
                                if word == 2 :
                                    continue
                                re_string.append(vi_vocab[word])
                            except IndexError:
                                re_string.append(vi_vocab[0])
                                continue
                        restring=' '.join(re_string)
                        twrdoc.write(restring+'\n')
                    except tf.errors.OutOfRangeError:
                
                        eval_sess.run(eval_dataset_iterator.initializer)
                
                        break
                     
	
                    #score=score+sentence_bleu(eval_target.tolist(),result)
                    

                    #print(eval_target.tolist())
                    #print(result)
                    #score=score+sentence_bleu(eval_target.tolist(),result)
                #score=score/400
                #print(score)
                swrdoc.close()  
                twrdoc.close() 
            
        '''
    embedding_layer = Embedding(len(word_index) + 1,
                                200,
                                weights=[embedding_matrix],
                                input_length=100,
                                trainable=False)

    mapping_strings_en = tf.constant(en_vocab)
    mapping_strings_vi = tf.constant(vi_vocab)
    '''
