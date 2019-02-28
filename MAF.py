
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np


# In[2]:


# ### RUN TF ON GPU??


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


import numpy as np
import _pickle as pickle


# In[5]:


DTYPE=tf.float32
NP_DTYPE=np.float32


# In[6]:
import gzip, numpy

# Load the dataset

# import datasets was giving error so changed it to this form
f = open('mnist.pkl', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

#import datasets
#data = datasets.MNIST(logit=True, dequantize=True)


# In[7]:


# root = '/home/risheek/maf/datasets/data/'


# In[8]:


# import datasets
# data = datasets.CIFAR10(logit=True, flip=True, dequantize=True)


# In[9]:

# Here trn contains the training set of images - as an list of images - as numpy tensors - of dimension 784
# Total of 10000 images each represented as 784 element array

# tst contains the validation dataset.

trn= [train_set[0]]
# val=[data.val.x]
tst= [valid_set[0]]

train_label= [train_set[1]]
# val_label=[data.val.y] #for both this and next lable is updated to label.
test_label= [valid_set[1]]

data_values=np.concatenate([trn,tst],axis=1)
print (data_values.shape)
data_label=np.concatenate([train_label,test_label],axis=1)
print (data_label.shape)

data_label_new = []
for data_element in data_label[0]:
    tensor = [0,0,0,0,0,0,0,0,0,0]
    tensor[data_element] = 1
    data_label_new.append(tensor)

train_label = [data_label_new[0:50000]]

data_label_new = np.array([data_label_new])
print (data_label_new.shape)


print (len(data_values[0]))
print (len(data_label[0]))

# No idea why the data and labels are concatenated here

data=np.concatenate([data_label_new[0],data_values[0]],axis=1)

print ("Data shape " + str(data.shape))
#print(data.shape)
# In[10]:

print("data_lenth",len(trn[0][0]))
print("label_lenth",len(train_label[0]))


# In[11]:


# for memory issues
# data=data[0:25000,:]
trn=0;tst=0;test_label=0;data_values=0;data_label=0;
# train_label=0;#commented because used below


# In[12]:


n_samples_g=40000
# n_samples_g=1500
train_maf=data[0:n_samples_g,10:]
# nominal_clean_cdf=data[n_samples_g:2*n_samples_g,:]
# nominal_mix_cdf=data[2*n_samples_g:3*n_samples_g,:]
# nominal_test_cdf=data[3*n_samples_g:4*n_samples_g,:]
rest_data=data[n_samples_g:,:]
print(train_maf.shape)
print(rest_data[:,:10].shape)
# trn=train_maf


# In[13]:


# print(rest_data[1:10,10:15])


# In[14]:


# ## MNIST
temp_c1=np.zeros([25000,784])
temp_rest=np.zeros([50000,784])
# ## CIFAR10
# temp_c1=np.zeros([6000,3072])
# temp_rest=np.zeros([50000,3072])
# train_d=trn[0]
train_d=train_maf


# In[15]:


j1=0;j2=0;j3=0;j4=0;
j=0
for i in range(len(train_d)):
# print(train_label[0][1])
#     a1=(train_label[0][i]==[0,0,0,1,0,0,0,0,0,0])
      ##training class chosen
#     if a1.all():

    if (train_label[0][i]==[0,0,0,1,0,0,0,0,0,0]):
        temp_c1[j1,:]=train_d[i]
        j1=j1+1
    elif (train_label[0][i]==[0,0,1,0,0,0,0,0,0,0]):
        temp_c1[j1,:]=train_d[i]
        j1=j1+1
    elif (train_label[0][i]==[0,1,0,0,0,0,0,0,0,0]):
        temp_c1[j1,:]=train_d[i]
        j1=j1+1
    elif (train_label[0][i]==[1,0,0,0,0,0,0,0,0,0]):
        temp_c1[j1,:]=train_d[i]
        j1=j1+1
    else:
        temp_rest[j,:]=train_d[i]
        j=j+1


# In[16]:


# ##Multiple classes.
# ##remove zero rows r[~np.all(r == 0, axis=1)]
train_d1=temp_c1[~np.all(temp_c1==0,axis=1)].astype(NP_DTYPE)
train_rest=temp_rest[~np.all(temp_rest==0,axis=1)].astype(NP_DTYPE)


# In[17]:


trn=0;tst=0;train_label=0;test_label=0;data_values=0;data_label=0;


# In[18]:


print(train_rest.shape,len(train_rest),train_d1.shape)
print(train_d1.shape,int(train_d1.shape[0]/2))


# In[19]:


k=int((train_d1.shape[0]*2)/3)
train_d1_2=train_d1[k:,]
train_d1=train_d1[0:k,]
train_rest=train_rest[:5000,]


# In[20]:


# print(train_d1.shape,train_d1_2.shape)


# In[21]:


##If multiple classes.
dataset1 = tf.data.Dataset.from_tensor_slices(train_d1.astype(NP_DTYPE))
dataset1 = dataset1.repeat()
dataset1 = dataset1.shuffle(buffer_size=train_d1.shape[0])
dataset1 = dataset1.prefetch(3 * 200)
dataset1 = dataset1.batch(200)
data_iterator1 = dataset1.make_one_shot_iterator()
x_samples_train1 = data_iterator1.get_next()


# In[22]:


# print(train_d[0:10])


# In[23]:


tfd = tf.contrib.distributions
tfb = tfd.bijectors


# In[24]:


# sess = tf.InteractiveSession(config=config)
# sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))

sess = tf.InteractiveSession()


# In[25]:


dims=[len(train_d[0])]


# In[26]:


# tfb.BatchNormalization()


# In[27]:


class BatchNorm(tfb.Bijector):
    def __init__(self, eps=1e-5, decay=0.95, validate_args=False, name="batch_norm"):
        super(BatchNorm, self).__init__(
            event_ndims=1, validate_args=validate_args, name=name)
        self._vars_created = False
        self.eps = eps
        self.decay = decay

    def _create_vars(self, x):
        n = x.get_shape().as_list()[1]
        with tf.variable_scope(self.name):
            self.beta = tf.get_variable('beta', [1, n], dtype=DTYPE)
            self.gamma = tf.get_variable('gamma', [1, n], dtype=DTYPE)
            self.train_m = tf.get_variable(
                'mean', [1, n], dtype=DTYPE, trainable=False)
            self.train_v = tf.get_variable(
                'var', [1, n], dtype=DTYPE, initializer=tf.ones_initializer, trainable=False)
        self._vars_created = True

    def _forward(self, u):
        if not self._vars_created:
            self._create_vars(u)
        return (u - self.beta) * tf.exp(-self.gamma) * tf.sqrt(self.train_v + self.eps) + self.train_m

    def _inverse(self, x):
        # Eq 22. Called during training of a normalizing flow.
        if not self._vars_created:
            self._create_vars(x)
        # statistics of current minibatch
        m, v = tf.nn.moments(x, axes=[0], keep_dims=True)
        # update train statistics via exponential moving average
        update_train_m = tf.assign_sub(
            self.train_m, self.decay * (self.train_m - m))
        update_train_v = tf.assign_sub(
            self.train_v, self.decay * (self.train_v - v))
        # normalize using current minibatch statistics, followed by BN scale and shift
        with tf.control_dependencies([update_train_m, update_train_v]):
            return (x - m) * 1. / tf.sqrt(v + self.eps) * tf.exp(self.gamma) + self.beta

    def _inverse_log_det_jacobian(self, x):
        # at training time, the log_det_jacobian is computed from statistics of the
        # current minibatch.
        if not self._vars_created:
            self._create_vars(x)
        _, v = tf.nn.moments(x, axes=[0], keep_dims=True)
        abs_log_det_J_inv = tf.reduce_sum(
            self.gamma - .5 * tf.log(v + self.eps))
        return abs_log_det_J_inv


# In[28]:


num_bijectors1=2 # number of MADES*2 if BN is used else *2 not required

bijectors1=[]
for i in range(num_bijectors1):
    bijectors1.append(tfb.MaskedAutoregressiveFlow(
        shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
            hidden_layers=[4048,3024,2000])))
#     if (i % 2)==0: #BN
    #bijectors1.append(BatchNorm(name='batch_norm1%d' % i))
    bijectors1.append(tfb.Permute(np.random.permutation(784)))
    #bijectors1.append(tfb.Permute(np.random.permutation(3072)))

flow_bijector1=tfb.Chain(list(reversed(bijectors1[:-1])))## MAF/MADE with batchnorm

# flow_bijector1=tfb.Chain(list(bijectors1)) #MADE without batchnorm


# In[29]:


maf1=tfd.TransformedDistribution(
    tfd.MultivariateNormalDiag(loc=tf.zeros(dims)),
    bijector=flow_bijector1)#MAF<-MADES


# In[30]:


loss=-tf.reduce_mean(maf1.log_prob(x_samples_train1))

loss1 = loss+sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

#train_op1 = tf.train.AdamOptimizer(1e-4).minimize(loss1)
train_op1 = tf.train.GradientDescentOptimizer(learning_rate=1e-4, use_locking=True).minimize(loss1)

###################-----------------------
# beta=1e-8
# reg = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2) + tf.nn.l2_loss(weights_3) + \
#           tf.nn.l2_loss(weights_4)
# loss2 = tf.reduce_mean(loss1 + reg * beta)
# train_op1 = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(loss2)
# extend_with_weight_decay(tf.train.AdamOptimizer, weight_decay=weight_decay)


# In[31]:


sess.run(tf.global_variables_initializer())


# In[32]:


# sess.run([train_op, loss])


# In[ ]:


global_step = []
# np_losses = []
np_losses1 = [];
val_losses1=[];
num_iterations=20
# ## keep iterations 126 for all classes?;for now 126 for class1 and 250 for class 0,2,3
for i in range(num_iterations):
    print("i=",i)
    _, np_loss1 = sess.run([train_op1, loss1])
    print ("ran session1 ")

#     if i % 1000 == 0:
    global_step.append(i)
# #     np_losses.append(np_loss)
    np_losses1.append(np_loss1)
    if (i%25)==0:
        temp_val_loss1=-tf.reduce_mean(maf1.log_prob(train_d1_2)).eval()
        if temp_val_loss1>=-50000:
            val_loss1=temp_val_loss1

    val_losses1.append(val_loss1)
# #     if i % int(1e4) == 0:
# #         print(i, np_loss)
# # print(np_losses)


# In[ ]:


# # plt.plot(np_losses)
# # plt.xlabel("epoch")
# # plt.ylabel("NLL")
# # # print(np_losses,np_losses1)
plt.plot(np_losses1,color='b',label='training')
plt.xlabel("epoch")
plt.ylabel("NLL")
plt.show()
# plt.plot(np_losses2,color='g')
# plt.xlabel("epoch")
# plt.ylabel("NLL")
# plt.plot(np_losses3,color='m')
# plt.xlabel("epoch")
# plt.ylabel("NLL")

plt.plot(val_losses1,color='r',label='validation')
plt.xlabel("epoch")
plt.ylabel("NLL")
plt.legend()
plt.show()
# plt.plot(val_losses2,color='y')
# plt.xlabel("epoch")
# plt.ylabel("NLL")
# plt.plot(val_losses3,color='c')
# plt.xlabel("epoch")
# plt.ylabel("NLL")

# plt.plot(np_losses3)
# plt.xlabel("epoch")
# plt.ylabel("NLL")
# plt.plot(np_losses4)
# plt.xlabel("epoch")
# plt.ylabel("NLL")


# In[ ]:


# print(val_losses1)


# In[ ]:


trn=0;tst=0;train_label=0;test_label=0;
data_values=0;data_label=0;data=0;

temp_c1=0;temp_c2=0;temp_c3=0;temp_c4=0;
temp_rest=0

# train_d=trn[0]
train_d=0
train_d1=0;train_d2=0;train_d3=0;train_d4=0;
# train_rest=0;
train_maf=0;

x_samples_train1=0;x_samples_train2=0;x_samples_train3=0;x_samples_train4=0;
dataset=0;dataset1=0;dataset2=0;datase3=0;dataset4=0;

data_iterator1=0;data_iterator2=0;data_iterator3=0;data_iterator4=0;


# In[ ]:


maf1_score=maf1.log_prob(rest_data[:,10:].astype(NP_DTYPE)).eval()


# In[ ]:


# # print(np.random.choice(train_rest.shape[0],,replace=False))
# train_rest_red=train_rest[np.random.choice(train_rest.shape[0],train_d1_2.shape[0],replace=False),:]
# # maf1_score_rest=maf1.log_prob(train_rest.astype(NP_DTYPE)).eval()
# maf1_score_rest=maf1.log_prob(train_rest_red.astype(NP_DTYPE)).eval()
# maf1_score_rest=np.array([maf1_score_rest]).T


# In[ ]:


maf1_score=np.array([maf1_score]).T
maf_valid_score=maf1.log_prob(train_d1_2).eval()
maf_valid_score=np.array([maf_valid_score]).T


# In[ ]:


np.savetxt('maf_all_known_classes',maf1_score)


# In[ ]:


# np.savetxt('maf_valid_l2_class_1',maf_valid_score)
# np.savetxt('maf_valid_l2_class_2',maf_valid_score)
# np.savetxt('maf_valid_l2_class_3',maf_valid_score)
# np.savetxt('maf_valid_l2_class_4',maf_valid_score)


# In[ ]:


rest_data_maf=np.concatenate([maf1_score,rest_data[:,:10]],axis=1)
print(np.mean(maf1_score),np.mean(maf1_score_rest),np.mean(maf_valid_score))
plt.hist(maf1_score_rest,label='novel')
plt.axvline(np.mean(maf1_score_rest), color='b', linestyle='dashed', linewidth=1)
plt.xlim([-3000,-1200])
plt.legend()
# plt.title(str(np.mean(maf1_score_rest))
# plt.show()
plt.hist(maf_valid_score,label='nominal')
plt.legend()
plt.xlabel("log likelihood")
plt.ylabel("count")
plt.axvline(np.mean(maf_valid_score), color='r', linestyle='dashed', linewidth=1)
plt.savefig('/Users/rogen/Desktop/image1')


# In[ ]:


# print(np.mean(maf1_score_rest))
# # print(maf1_score_rest)
# maf1_score_rest_sort=maf1_score_rest
# maf1_score_rest_sort=np.sort(maf1_score_rest_sort,axis=None)
# maf1_score_sort=maf1_score
# maf1_score_sort=np.sort(maf1_score_sort,axis=None)
# print(maf1_score.shape,'and \n',maf1_score_rest_sort[:200])
# # print(maf1_score_rest_sort[0:50])


# In[ ]:


# import numpy as np
# rest_data_prev=np.loadtxt('maf_l2_class_1')
# rest_data_prev=np.loadtxt('maf_l2_class_1_2')
# rest_data_prev=np.loadtxt('maf_l2_class_1_2_3')
# rest_data_maf=np.loadtxt('maf_l2_class_20ktrn_1_2_3_4')
# print(rest_data_prev.shape)


# In[ ]:


# rest_data_maf=np.concatenate([maf1_score,rest_data_prev],axis=1)
# print(rest_data_maf.shape)


# In[ ]:


# print(len(rest_data_maf[0,:]))
print(rest_data_maf[0:10,0:4])
# log_prob_diff=np.amax(rest_data_maf[:,0:4],axis=1)-np.amin(rest_data_maf[:,0:3],axis=1)
# print(log_prob_diff[0:100])
# plt.hist(log_prob_diff)


# In[ ]:


# np.savetxt('maf_l2_class_1',rest_data_maf)
# np.savetxt('maf_l2_class_1_2',rest_data_maf)
# np.savetxt('maf_l2_class_1_2_3',rest_data_maf)
# np.savetxt('maf_4_l2_class_4k3k2k_40ktrn_1200_epoch_1_2_3_4',rest_data_maf)
