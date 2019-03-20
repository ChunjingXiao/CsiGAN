import os
import time

import numpy as np
import tensorflow as tf
import sys
from sklearn import metrics

#from data import cifar10_input
from cifar_ganNN_NN import discriminator, generator
#from cifar_ganVGG_VGG import discriminator, generator
#from cifar_ganNN_VGG import discriminator, generator

#-------------------------------CycleGAN--1----------------begin-----------------
import argparse
from CycleGANmodel import cyclegan

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='horse2zebra', help='path of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=100, help='# of epoch to decay lr')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--load_size', dest='load_size', type=int, default=286, help='scale images to this size')
#parser.add_argument('--fine_size', dest='fine_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=200, help='then crop to this size')      #-------xiao-------------
parser.add_argument('--fine_size2', dest='fine_size2', type=int, default=30, help='then crop to this size') #-------xiao-------------

parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=1000, help='save a model every save_freq iterations')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=100, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=10.0, help='weight on L1 term in objective')
parser.add_argument('--use_resnet', dest='use_resnet', type=bool, default=True, help='generation network using reidule block')
parser.add_argument('--use_lsgan', dest='use_lsgan', type=bool, default=True, help='gan loss defined in lsgan')
parser.add_argument('--max_size', dest='max_size', type=int, default=50, help='max size of image pool, 0 means do not use image pool')

args = parser.parse_args()
'''
if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)
if not os.path.exists(args.sample_dir):
    os.makedirs(args.sample_dir)
if not os.path.exists(args.test_dir):
    os.makedirs(args.test_dir)
'''

#--------------------------------CycleGAN--1---------------end-------------------
categoryNum = 101




flags = tf.app.flags
flags.DEFINE_integer('gpu', 0, 'gpu [0]')
flags.DEFINE_integer('batch_size', 60, "batch size [25]")  # --------60:0.8417(1169)-------70:0.8597--------------
flags.DEFINE_string('data_dir', './data/cifar-10-python/','data directory')
#flags.DEFINE_string('logdir', './log/cifar', 'log directory')
flags.DEFINE_integer('seed', 10, 'seed numpy')
flags.DEFINE_integer('labeled', 400, 'labeled data per class [100]')
flags.DEFINE_float('learning_rate', 0.0003, 'learning_rate[0.0003]')
flags.DEFINE_float('unl_weight', 1.0, 'unlabeled weight [1.]')
flags.DEFINE_float('lbl_weight', 1.0, 'unlabeled weight [1.]')
flags.DEFINE_float('ma_decay', 0.9999, 'exponential moving average for inference [0.9999]')
flags.DEFINE_integer('decay_start', 1200, 'start learning rate decay [1200]')
flags.DEFINE_integer('epoch', 1600, 'epochs [1400]')
flags.DEFINE_boolean('validation', False, 'validation [False]')  #-------------not used for xiao-------------
flags.DEFINE_boolean('clamp', False, 'validation [False]')
flags.DEFINE_boolean('abs', False, 'validation [False]')

flags.DEFINE_float('lmin', 1.0, 'unlabeled weight [1.]')
flags.DEFINE_float('lmax', 1.0, 'unlabeled weight [1.]')

flags.DEFINE_integer('nabla', 1, 'choose regularization [1]')  #-------------xiao-------------
flags.DEFINE_float('gamma', 0.001, 'weight regularization')
flags.DEFINE_float('alpha', 40., 'displacement along data manifold')  #-------------default epsilon is 20. ---------
flags.DEFINE_float('eta', 1., 'perturbation latent code')
flags.DEFINE_integer('freq_print', 10000, 'frequency image print tensorboard [10000]')
flags.DEFINE_integer('step_print', 50, 'frequency scalar print tensorboard [50]')
#flags.DEFINE_integer('freq_test', 1, 'frequency test [500]')
flags.DEFINE_integer('freq_test', 1, 'frequency test [500]')
flags.DEFINE_integer('freq_save', 10, 'frequency saver epoch[50]')
FLAGS = flags.FLAGS



def get_getter(ema):
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var
    return ema_getter


def display_progression_epoch(j, id_max):
    batch_progression = int((j / id_max) * 100)
    sys.stdout.write(str(batch_progression) + ' % epoch' + chr(13))
    _ = sys.stdout.flush


def linear_decay(decay_start, decay_end, epoch):
    return min(-1 / (decay_end - decay_start) * epoch + 1 + decay_start / (decay_end - decay_start),1)


def scaler(x, feature_range=(-1, 1)):
    # scale to (0, 1)
    getMax = x.max()
    x = ((x - x.min())/(getMax - x.min()))
    
    # scale to feature_range
    min, max = feature_range
    x = x * (max - min) + min
    return x
def main(_):
    '''
    print("\nParameters:")
    for attr,value in tf.app.flags.FLAGS.flag_values_dict().items():
        print("{}={}".format(attr,value))
    print("")
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    #if not os.path.exists(FLAGS.logdir):
    #    os.makedirs(FLAGS.logdir)
    # Random seed
    rng = np.random.RandomState(FLAGS.seed)  # seed labels
    rng_data = np.random.RandomState(rng.randint(0, 2**10))  # seed shuffling

    # load Data    
    data_dir = 'data/'
    trainx=np.load(data_dir+'csi_tensorTrain.npy')   #-----------------xiao------load data--------------------
    trainy=np.load(data_dir+'labelTrain.npy')
    testx=np.load(data_dir+'csi_tensorTest.npy')
    testy=np.load(data_dir+'labelTest.npy')
    trainx_unl=np.load(data_dir+'csi_tensorUnlabel.npy') 
    
    print('trainx.shape     ::', trainx.shape)
    print('trainy.shape     ::', trainy.shape)
    print('testx.shape      ::', testx.shape)
    print('testy.shape      ::', testy.shape)   
    print('trainx_unl.shape ::', trainx_unl.shape)
    
    
    trainx = scaler(trainx)                         #-----------------xiao--------normalize data--------------
    testx = scaler(testx)   
    trainx_unl = scaler(trainx_unl)  #-------------20181101-------------------


    inds = rng_data.permutation(trainx.shape[0])   #-----------------xiao--------shuffling data-------------
    trainx = trainx[inds]
    trainy = trainy[inds]
    inds = rng_data.permutation(testx.shape[0])
    testx = testx[inds]
    testy = testy[inds]
    inds = rng_data.permutation(trainx_unl.shape[0])
    trainx_unl = trainx_unl[inds]




    trainxTemp = trainx
    trainyTemp = trainy
    trainx = []
    trainy = []
    #print('trainx_unl.shape[0]:',trainx_unl.shape[0],'-------------;trainxTemp.shape[0]:',trainxTemp.shape[0])
    #print('int(np.ceil(trainx_unl.shape[0] / float(trainxTemp.shape[0]))):', int(np.ceil(trainx_unl.shape[0] / float(trainxTemp.shape[0]))))
    for t in range(int(np.ceil(trainx_unl.shape[0] / float(trainxTemp.shape[0])))):  # same size lbl and unlb
        inds = rng.permutation(trainxTemp.shape[0])              #------every epoch, the train will shuttle------------------------------------------------------------
        trainx.append(trainxTemp[inds])                          # -----the number of trainx, trainy and trainx_unl should be more than nr_batches_train * batch_size--
        trainy.append(trainyTemp[inds])                          # -----when trainx_unl is larger than trainx, there will be a problem
    trainx = np.concatenate(trainx, axis=0)
    trainy = np.concatenate(trainy, axis=0)
    #------------------xiao------------------added for when trainx_unl is lower than trainx-------------begin------------
    trainx_unlTemp = trainx_unl
    trainx_unl = []
    for t in range(int(np.ceil(trainx.shape[0] / float(trainx_unlTemp.shape[0])))):  # same size lbl and unlb
        inds = rng.permutation(trainx_unlTemp.shape[0]) 
        trainx_unl.append(trainx_unlTemp[inds]) 
    trainx_unl = np.concatenate(trainx_unl, axis=0)
    trainx_unl2 = trainx_unl.copy()
    trainx_unl2 = trainx_unl2[rng.permutation(trainx_unl2.shape[0])]
    #print('trainx_unl.shape:',trainx_unl.shape,'-------------;trainx_unl2.shape:',trainx_unl2.shape)
    #------------------xiao------------------added for when trainx_unl is lower than trainx-------------end--------------
    

    
    nr_batches_train = int(trainx.shape[0] / FLAGS.batch_size)
    nr_batches_test = int(testx.shape[0] / FLAGS.batch_size)
        
    print("Data:")
    print('train examples %d, batch %d, test examples %d, batch %d' % (trainx.shape[0], nr_batches_train, testx.shape[0], nr_batches_test))
    print('histogram train', np.histogram(trainy, bins=10)[0])
    print('histogram test ', np.histogram(testy, bins=10)[0])
    print("histogram labeled", np.histogram(trainy, bins=10)[0])
    print("")
    
    
    
    
    
    
    #-------------------------------CycleGAN--2----------------begin-----------------
    
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    
    
    data_dir = 'data/'
    dataA=np.load(data_dir+'cycleCsi_tensorTrain.npy') #-----------------xiao------load data--------------------
    dataB=np.load(data_dir+'cycleCsi_tensorUnlabel.npy')  
    
    
    cycleTrainData=np.load(data_dir+'cycleCsi_tensorTrain_all.npy')
    cycleTrainLabel = np.load(data_dir+'cycleLabelTrain_all.npy')
    
    print('cycleCsi_tensorTrain.shape ::', dataA.shape)
    print('cycleCsi_tensorUnlabel.shape ::', dataB.shape)
    print('cycleCsi_tensorTrain_all.shape ::', cycleTrainData.shape)
    print('cycleTrainLabel.shape ::', cycleTrainLabel.shape)
    
    sessCycle = tf.Session(config=tfconfig)
    model = cyclegan(sessCycle, args)
    #model.train(args,epoch,dataA,dataB) 
    #cycleGenData = model.test(sample_files)
    #print(cycleGenData.shape)
    
    
    '''
    trainCycleDataTemp = trainCycleData
    trainCycleLabelTemp = trainCycleLabel
    trainCycleData = []
    trainCycleLabel = []            
    for t in range(int(np.ceil(trainx.shape[0] / float(trainCycleDataTemp.shape[0])))):  # same size lbl and unlb
        inds = rng.permutation(trainCycleDataTemp.shape[0])              #------every epoch, the train will shuttle------------------------------------------------------------
        trainCycleData.append(trainCycleDataTemp[inds])                  # -----the number of trainx, trainy and trainx_unl should be more than nr_batches_train * batch_size--
        trainCycleLabel.append(trainCycleLabelTemp[inds])                # -----when trainx_unl is larger than trainx, there will be a problem
    trainCycleData = np.concatenate(trainCycleData, axis=0)
    trainCycleLabel = np.concatenate(trainCycleLabel, axis=0)
    '''
    #-------------------------------CycleGAN--2----------------end-------------------
    
    
    
    
    '''construct graph'''
    #unl = tf.placeholder(tf.float32, [FLAGS.batch_size, 32, 32, 3], name='unlabeled_data_input_pl')
    unl = tf.placeholder(tf.float32, [FLAGS.batch_size, 200, 30, 3], name='unlabeled_data_input_pl') #-------xiao----
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    #inp = tf.placeholder(tf.float32, [FLAGS.batch_size, 32, 32, 3], name='labeled_data_input_pl')
    inp = tf.placeholder(tf.float32, [FLAGS.batch_size, 200, 30, 3], name='labeled_data_input_pl')   #-------xiao----
    lbl = tf.placeholder(tf.int32, [FLAGS.batch_size], name='lbl_input_pl')
    # scalar pl
    lr_pl = tf.placeholder(tf.float32, [], name='learning_rate_pl')
    acc_train_pl = tf.placeholder(tf.float32, [], 'acc_train_pl')
    acc_test_pl = tf.placeholder(tf.float32, [], 'acc_test_pl')
    acc_test_pl_ema = tf.placeholder(tf.float32, [], 'acc_test_pl')

    random_z = tf.random_uniform([FLAGS.batch_size, 100], name='random_z')    
    generator(random_z, is_training_pl, init=True)  # init of weightnorm weights
    gen_inp = generator(random_z, is_training_pl, init=False, reuse=True)    
    gen_inpMani = model.generator(inp, model.options, True, name="generatorA2B")   # for my manifold
    pert_n = tf.nn.l2_normalize(tf.random_normal(shape=[FLAGS.batch_size, 200, 30, 3]), dim=[1,2,3])
    random_z_pert = inp + pert_n # random_z_pert = inp + FLAGS.eta * pert_n
    gen_inp_pert = model.generator(random_z_pert, model.options, True, name="generatorA2B")    
    gen_adv = (1-FLAGS.alpha)*gen_inpMani + FLAGS.alpha * gen_inp_pert #gen_adv=gen_inp+FLAGS.alpha*tf.nn.l2_normalize(gen_inp_pert-gen_inp,dim=[1,2,3])
    discriminator(unl, is_training_pl, init=True,category=categoryNum)
    
    #----------this is original----------------begin-----------------
    '''
    logits_lab, layer_label = discriminator(inp, is_training_pl, init=False, reuse=True,category=categoryNum)                # labeled_data_input_pl
    logits_gen, layer_fake = discriminator(gen_inp, is_training_pl, init=False, reuse=True,category=categoryNum)   # generator -- random_z
    logits_unl, layer_real = discriminator(unl, is_training_pl, init=False, reuse=True,category=categoryNum)       # unlabeled_data_input_pl
    logits_gen_adv, _ = discriminator(gen_adv, is_training_pl, init=False, reuse=True,category=categoryNum)        # generator -- random_z + FLAGS.eta * pert_n
    '''
    #----------this is original----------------end-------------------
    
    # ------------------------------------------------will change this---------------begin------------------
    logits_lab, layer_label = discriminator(inp, is_training_pl, init=False, reuse=True,category=categoryNum)                # labeled_data_input_pl
    logits_gen, layer_fake = discriminator(gen_inp, is_training_pl, init=False, reuse=True,category=categoryNum)   # generator -- random_z
    logits_unl, layer_real = discriminator(unl, is_training_pl, init=False, reuse=True,category=categoryNum)       # unlabeled_data_input_pl
    logits_gen_adv, _ = discriminator(gen_adv, is_training_pl, init=False, reuse=True,category=categoryNum)        # generator(cycleG + pert_n) # for my manifold
    logits_genMani, _ = discriminator(gen_inpMani, is_training_pl, init=False, reuse=True,category=categoryNum)   # generator(cycleG)     # for my manifold
    noiseDataLabel  = inp + 0.01*gen_inp
    noiseDataUnlabel= unl + 0.01*gen_inp
    logits_lab_noise, _ = discriminator(noiseDataLabel, is_training_pl, init=False, reuse=True,category=categoryNum) 
    logits_unl_noise, _ = discriminator(noiseDataUnlabel, is_training_pl, init=False, reuse=True,category=categoryNum) 
    # ------------------------------------------------will change this---------------end--------------------
    
    #--------------------------------CycleGAN--2---------------begin-----------------
    lblCycle = tf.placeholder(tf.int32, [FLAGS.batch_size], name='lblCycle_input_pl')
    gen_inpCycle = tf.placeholder(tf.float32, [FLAGS.batch_size, 200, 30, 3], name='labeled_dataCycle_input_pl') 
    logits_labCycle, _ = discriminator(gen_inpCycle, is_training_pl, init=False, reuse=True,category=categoryNum)
    
    #--------------------------------CycleGAN--2---------------end-------------------
    
    genLabel = tf.concat([tf.zeros(shape=(FLAGS.batch_size, int(categoryNum/2) )), tf.ones(shape=(FLAGS.batch_size, int(categoryNum/2)+1 ))], axis=1)
    unlLabel = tf.concat([tf.ones(shape=(FLAGS.batch_size, int(categoryNum/2) )), tf.zeros(shape=(FLAGS.batch_size, int(categoryNum/2)+1 ))], axis=1)    

    
    with tf.name_scope('loss_functions'):
        # discriminator
        unlPred = tf.nn.softmax(logits_unl) # l_unl = tf.reduce_logsumexp(logits_unl, axis=1)        # unlabeled_data_input_pl      
        loss_lab = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lbl, logits=logits_lab)) # labeled_data_input_pl
    
        #--------------------------------CycleGAN--3---------------begin-----------------
        genPred = tf.nn.softmax(logits_gen)  #l_gen2 = tf.reduce_logsumexp(logits_gen, axis=1)        # generator(random_z), batch_size*1
        #loss_unl = - 0.5 * tf.reduce_mean(l_unl)+ 0.5 * tf.reduce_mean(tf.nn.softplus(l_unl))+ 0.5 * tf.reduce_mean(tf.nn.softplus(l_gen))
        
        l_gen = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lblCycle, logits=logits_labCycle))
        #loss_unl = - 0.5 * tf.reduce_mean(l_unl)+ 0.5 * tf.reduce_mean(tf.nn.softplus(l_unl)) + 0.1 * l_gen + 0.1 * tf.reduce_mean(tf.nn.softplus(l_gen2))
        loss_unl = - 0.6 * tf.reduce_mean(tf.multiply(unlPred, unlLabel))+  0.2 * l_gen - 0.1 * tf.reduce_mean(tf.multiply(genPred, genLabel))
        #--------------------------------CycleGAN--3---------------end-------------------

        # generator

        #----------this is original----------------begin-----------------
        '''
        m1 = tf.reduce_mean(layer_real, axis=0)
        m2 = tf.reduce_mean(layer_fake, axis=0)
        manifold = tf.sqrt(tf.reduce_sum(tf.square(logits_gen - logits_gen_adv), axis=1))    # generator-random_z ~~ random_z+FLAGS.eta*pert_n
        loss_gen = tf.reduce_mean(tf.abs(m1 - m2))
        '''
        #----------this is original----------------end-------------------
        
        # ------------------------------------------------will change this---------------begin------------------
        #layer_all= layer_real*0.5 + layer_label*0.5 
        m1 = tf.reduce_mean(layer_real, axis=0)
        #m1 = tf.reduce_mean(layer_label, axis=0)
        m2 = tf.reduce_mean(layer_fake, axis=0)
        manifold = tf.sqrt(tf.reduce_sum(tf.square(logits_genMani - logits_gen_adv), axis=1))
        #manifold = tf.sqrt(tf.reduce_sum(0.5*tf.square(logits_lab - logits_lab_noise) + 0.5*tf.square(logits_unl - logits_unl_noise) , axis=1))
        loss_gen = tf.reduce_mean(tf.abs(m1 - m2))
        # ------------------------------------------------will change this---------------end--------------------


        if FLAGS.clamp:
            print('clamped mode')
            if FLAGS.abs:
                print('abs_clamp')
                manifold_clamped = tf.abs(manifold - tf.clip_by_value(manifold, FLAGS.lmin, FLAGS.lmax))
            else:
                manifold_clamped = tf.square(manifold - tf.clip_by_value(manifold, FLAGS.lmin, FLAGS.lmax))
            j_loss = tf.reduce_mean(manifold_clamped)
        else:
            j_loss = tf.reduce_mean(manifold)

        if FLAGS.nabla == 1:
            loss_dis = FLAGS.unl_weight * loss_unl + FLAGS.lbl_weight * loss_lab + FLAGS.gamma * j_loss
            #loss_gen = tf.reduce_mean(tf.abs(m1 - m2))
            print('manifold reg enabled')
        elif FLAGS.nabla == 2:
            pz = tf.random_normal([FLAGS.batch_size, 200, 30, 3])    #----------------------------------------------
            pert_n = FLAGS.alpha * tf.nn.l2_normalize(pz, dim=[1,2,3])
            logits_unl_pert, layer_real = discriminator(unl+pert_n, is_training_pl, init=False, reuse=True,category=categoryNum)
            ambient = tf.reduce_sum(tf.sqrt(tf.square(logits_unl - logits_unl_pert) + 1e-8), axis=1)
            ambient_loss = tf.reduce_mean(ambient)
            print('ambient enabled')
            loss_dis = FLAGS.unl_weight * loss_unl + FLAGS.lbl_weight * loss_lab + FLAGS.gamma * ambient_loss
            #loss_gen = tf.reduce_mean(tf.abs(m1 - m2))
        else:
            loss_dis = FLAGS.unl_weight * loss_unl + FLAGS.lbl_weight * loss_lab
            #loss_gen = tf.reduce_mean(tf.abs(m1 - m2))
            print('vanilla reg')

        correct_pred = tf.equal(tf.cast(tf.argmax(logits_lab, 1), tf.int32), tf.cast(lbl, tf.int32))
        accuracy_classifier = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        y_predict = tf.cast(tf.argmax(logits_lab, 1), tf.int32)

    # log condition number
    # mz =

    with tf.name_scope('optimizers'):
        # control op dependencies for batch norm and trainable variables
        tvars = tf.trainable_variables()
        dvars = [var for var in tvars if 'discriminator_model' in var.name]
        gvars = [var for var in tvars if 'generator_model' in var.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops_gen = [x for x in update_ops if ('generator_model' in x.name)]
        update_ops_dis = [x for x in update_ops if ('discriminator_model' in x.name)]
        optimizer_dis = tf.train.AdamOptimizer(learning_rate=lr_pl, beta1=0.5, name='dis_optimizer')
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=lr_pl, beta1=0.5, name='gen_optimizer')

        with tf.control_dependencies(update_ops_gen):
            train_gen_op = optimizer_gen.minimize(loss_gen, var_list=gvars)

        dis_op = optimizer_dis.minimize(loss_dis, var_list=dvars)
        ema = tf.train.ExponentialMovingAverage(decay=FLAGS.ma_decay)
        maintain_averages_op = ema.apply(dvars)

        with tf.control_dependencies([dis_op]):
            train_dis_op = tf.group(maintain_averages_op)

        logits_ema, _ = discriminator(inp, is_training_pl, getter=get_getter(ema), reuse=True,category=categoryNum)
        correct_pred_ema = tf.equal(tf.cast(tf.argmax(logits_ema, 1), tf.int32), tf.cast(lbl, tf.int32))
        accuracy_ema = tf.reduce_mean(tf.cast(correct_pred_ema, tf.float32))
        y_predict_ema = tf.cast(tf.argmax(logits_ema, 1), tf.int32)
    # training global varialble
    global_epoch = tf.Variable(0, trainable=False, name='global_epoch')
    global_step = tf.Variable(0, trainable=False, name='global_step')
    inc_global_step = tf.assign(global_step, global_step+1)
    inc_global_epoch = tf.assign(global_epoch, global_epoch+1)



    
    max_acc = tf.Variable([0.000001,0.0])        #----------[maxAcc, epoch]-------------------------xiao----save max accuracy--------------------------------
    accTemp = tf.placeholder(tf.float32,shape=(2)) 
    maxAccAssign = tf.assign(max_acc,accTemp)
    max_acc_ema = tf.Variable([0.000001,0.000001,0.0]) #----------[maxAcc_ema, f1_ema, epoch]-------xiao----save max ema-----------------------------------
    accTemp_ema = tf.placeholder(tf.float32,shape=(3)) 
    maxAccAssign_ema = tf.assign(max_acc_ema,accTemp_ema)
    # op initializer for session manager
    init_gen = [var.initializer for var in gvars][:-3]
    with tf.control_dependencies(init_gen):
        op = tf.global_variables_initializer()
    init_feed_dict = {inp: trainx_unl[:FLAGS.batch_size], unl: trainx_unl[:FLAGS.batch_size], is_training_pl: True}

    #sv = tf.train.Supervisor(logdir=FLAGS.logdir, global_step=global_epoch, summary_op=None, save_model_secs=0,init_op=op,init_feed_dict=init_feed_dict,max_to_keep=2000)
    #sv = tf.train.Supervisor(logdir=FLAGS.logdir, global_step=global_epoch, summary_op=None, save_model_secs=0,init_op=op,init_feed_dict=init_feed_dict)
    sv = tf.train.Supervisor(global_step=global_epoch, summary_op=None, save_model_secs=0,init_op=op,init_feed_dict=init_feed_dict)
    # sv.saver(max_to_keep = 2000)
    '''//////training //////'''
    
    
    
    



    
    print('start training')
    with sv.managed_session() as sess:   # tf.train.Supervisor----------------mainly used for saving model-------------xiao------------------
        tf.set_random_seed(rng.randint(2 ** 10))
        print('\ninitialization done')
        print('Starting training from epoch :%d, step:%d \n'%(sess.run(global_epoch),sess.run(global_step)))

        #writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
        #sv.saver(max_to_keep=2000)


        
        while not sv.should_stop():
            epoch = sess.run(global_epoch)
            train_batch = sess.run(global_step)
            
            #sess = tf_debug.LocalCLIDebugWrapperSession(sess,ui_type="readline") #----------------------------debug---------------------------------
            #-------------------------------CycleGAN--4----------------begin-----------------
            if(epoch%50 == 0): #----------------redyce CycleGAN-------------------------------------------------------------------------------
                model.train(args,epoch,dataA,dataB) 
            #model.train(args,epoch,dataA,dataB) 
            cycleGenData = model.test(cycleTrainData)   
            
            
            
            trainCycleDataTemp = cycleGenData
            trainCycleLabelTemp = cycleTrainLabel
            cycleGenData = []
            cycleTrainLabel = []            
            for t in range(int(np.ceil(trainx.shape[0] / float(trainCycleDataTemp.shape[0])))):  # same size lbl and unlb
                inds = rng.permutation(trainCycleDataTemp.shape[0])              #------every epoch, the train will shuttle------------------------------------------------------------
                cycleGenData.append(trainCycleDataTemp[inds])                  # -----the number of trainx, trainy and trainx_unl should be more than nr_batches_train * batch_size--
                cycleTrainLabel.append(trainCycleLabelTemp[inds])                # -----when trainx_unl is larger than trainx, there will be a problem
            cycleGenData = np.concatenate(cycleGenData, axis=0)
            cycleTrainLabel = np.concatenate(cycleTrainLabel, axis=0)
                        
            #-------------------------------CycleGAN--4----------------begin-----------------
            

            if (epoch >= FLAGS.epoch):
                print("Training done")
                sv.stop()
                break

            begin = time.time()
            train_loss_lab=train_loss_unl=train_loss_gen=train_acc=test_acc=test_acc_ema=train_j_loss = 0
            lr = FLAGS.learning_rate #lr = FLAGS.learning_rate * linear_decay(FLAGS.decay_start,FLAGS.epoch,epoch)
            precsionAll = recallAll = f1All = acc2All= f1All_ema= 0


            # construct randomly permuted batches
            inds = rng.permutation(trainx.shape[0]) 
            trainx = trainx[inds]              
            trainy = trainy[inds]  
            trainx_unl = trainx_unl[rng.permutation(trainx_unl.shape[0])]  
            trainx_unl2 = trainx_unl2[rng.permutation(trainx_unl2.shape[0])]            

            # training
            for t in range(nr_batches_train):

                display_progression_epoch(t, nr_batches_train)
                ran_from = t * FLAGS.batch_size
                ran_to = (t + 1) * FLAGS.batch_size
                

                # train discriminator
                #--------------------------------CycleGAN--5---------------begin-----------------
                #feed_dict = {unl: trainx_unl[ran_from:ran_to],is_training_pl: True,inp: trainx[ran_from:ran_to],lbl: trainy[ran_from:ran_to],lr_pl: lr}
                
                feed_dict = {unl: trainx_unl[ran_from:ran_to],is_training_pl: True,inp: trainx[ran_from:ran_to],lbl: trainy[ran_from:ran_to],lr_pl: lr,
                             gen_inpCycle:cycleGenData[ran_from:ran_to],lblCycle:cycleTrainLabel[ran_from:ran_to]}
                #print('ran_from:',ran_from,'; ran_to:', ran_to)
                
                #lblCycle = tf.placeholder(tf.int32, [FLAGS.batch_size], name='lblCycle_input_pl')
                #logits_labCycle, _ = discriminator(gen_inpCycle, is_training_pl, init=False, reuse=True,category=categoryNum)
                
                #--------------------------------CycleGAN--5---------------end-------------------
                    
                #_, acc, lu, lb, jl, sm = sess.run([train_dis_op, accuracy_classifier, loss_lab, loss_unl, j_loss, sum_op_dis],feed_dict=feed_dict)
                #_, acc, lb, lu, jl = sess.run([train_dis_op, accuracy_classifier, loss_lab, loss_unl, j_loss],feed_dict=feed_dict)
                _, acc, lb, lu, jl = sess.run([train_dis_op, accuracy_classifier, loss_lab, l_gen, j_loss],feed_dict=feed_dict) #-------loss_unl-->l_gen for Cycle---- 
                train_loss_unl += lu
                train_loss_lab += lb
                train_acc += acc
                train_j_loss += jl
                #if (train_batch % FLAGS.step_print) == 0:
                #    writer.add_summary(sm, train_batch)

                # train generator
                _, lg = sess.run([train_gen_op, loss_gen], feed_dict={unl: trainx_unl2[ran_from:ran_to],is_training_pl: True,lr_pl: lr})
                #_, lg = sess.run([train_gen_op, loss_gen], feed_dict={unl: trainx_unl2[ran_from:ran_to],is_training_pl: True,lr_pl: lr,inp: trainx[ran_from:ran_to]})
                train_loss_gen += lg
                #if (train_batch % FLAGS.step_print) == 0:
                #    writer.add_summary(sm, train_batch)

                if (train_batch % FLAGS.freq_print == 0) & (train_batch != 0):
                    ran_from = np.random.randint(0, trainx_unl.shape[0] - FLAGS.batch_size)
                    ran_to = ran_from + FLAGS.batch_size
                    #sm = sess.run(sum_op_im,feed_dict={is_training_pl: True, unl: trainx_unl[ran_from:ran_to]})
                    #writer.add_summary(sm, train_batch)

                train_batch += 1
                sess.run(inc_global_step)

            train_loss_lab /= nr_batches_train
            train_loss_unl /= nr_batches_train
            train_loss_gen /= nr_batches_train
            train_acc /= nr_batches_train
            train_j_loss /= nr_batches_train

            # Testing moving averaged model and raw model
            if (epoch % FLAGS.freq_test == 0) | (epoch == FLAGS.epoch-1):
                for t in range(nr_batches_test):
                    ran_from = t * FLAGS.batch_size
                    ran_to = (t + 1) * FLAGS.batch_size
                    feed_dict = {inp: testx[ran_from:ran_to],
                                 lbl: testy[ran_from:ran_to],
                                 is_training_pl: False}
                    #acc, acc_ema = sess.run([accuracy_classifier, accuracy_ema], feed_dict=feed_dict)
                    acc, acc_ema,y_pred,y_pred_ema = sess.run([accuracy_classifier, accuracy_ema,y_predict,y_predict_ema], feed_dict=feed_dict)
                    y_true = testy[ran_from:ran_to]
                    f1 = metrics.f1_score(y_true, y_pred,average="weighted")      # weighted  macro  micro
                    f1_ema = metrics.f1_score(y_true, y_pred_ema,average="weighted") #acc2= metrics.accuracy_score(y_true, y_pred) 
                    precsion = metrics.precision_score(y_true, y_pred,average="micro")
                    recall = metrics.recall_score(y_true, y_pred,average="micro")           
                
                    f1All += f1
                    f1All_ema += f1_ema # acc2All += acc2                  
                    precsionAll += precsion
                    recallAll += recall                
                    
                    test_acc += acc
                    test_acc_ema += acc_ema
                test_acc /= nr_batches_test
                test_acc_ema /= nr_batches_test

                f1All /= nr_batches_test
                f1All_ema /= nr_batches_test # acc2All /= nr_batches_test  
                precsionAll /= nr_batches_test
                recallAll /= nr_batches_test
                
                
                #sum = sess.run(sum_op_epoch, feed_dict={acc_train_pl: train_acc,acc_test_pl: test_acc, acc_test_pl_ema: test_acc_ema,lr_pl: lr})
                #writer.add_summary(sum, epoch)
               
                #print("Epoch%d %ds lossGen=%.4f lossLab=%.4f lossUnl=%.4f trainAcc=%.4f testAcc=%.4f testAccEma=%0.4f"
                #    % (epoch, time.time() - begin, train_loss_gen, train_loss_lab, train_loss_unl, train_acc,test_acc, test_acc_ema))

                
                accMax = sess.run(max_acc)
                if(accMax[0] < test_acc):
                    accMax = [test_acc,epoch]
                    sess.run(maxAccAssign,feed_dict={accTemp:[test_acc,epoch]})
                accMax_ema = sess.run(max_acc_ema)
                if(accMax_ema[0] < test_acc_ema):
                    accMax_ema = [test_acc_ema,f1All_ema,epoch]
                    sess.run(maxAccAssign_ema,feed_dict={accTemp_ema:[test_acc_ema,f1All_ema,epoch]})
                    
                print("Ep%d %ds lGen=%.4f lLab=%.4f lUnl=%.4f train=%.4f test=%.2f max=%0.2f Ep%d testE=%0.2f f1E=%0.2f maxA=%0.2f maxF=%0.2f Ep%d"
                      % (epoch, time.time() - begin, train_loss_gen, train_loss_lab, train_loss_unl, train_acc,test_acc*100,accMax[0]*100, accMax[1], \
                         test_acc_ema*100,f1All_ema*100, accMax_ema[0]*100,accMax_ema[1]*100, accMax_ema[2]))
                
                '''print("Ep%d %ds lGen=%.4f lLab=%.4f lUnl=%.4f train=%.4f test=%.4f max=%0.4f Ep%d testE=%0.4f f1E=%0.4f maxE=%0.4f Ep%d"
                                  % (epoch, time.time() - begin, train_loss_gen, train_loss_lab, train_loss_unl, train_acc,test_acc,accMax[0], accMax[1], \
                                     test_acc_ema,f1All_ema, accMax_ema[0], accMax_ema[2]))  '''               
            else:
                print("Epoch%d %ds" % (epoch, time.time() - begin))                   
                
                

                   

            sess.run(inc_global_epoch)

            # save snapshots of model
            '''
            if ((epoch % FLAGS.freq_save == 0)) | (epoch == FLAGS.epoch-1):
                string = 'model-' + str(epoch)
                save_path = os.path.join(FLAGS.logdir, string)
                sv.saver.save(sess, save_path)
                print("Model saved in file: %s" % (save_path))
            '''


if __name__ == '__main__':
    tf.app.run()
