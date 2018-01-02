#Based on 'Continuous Deep Q-learning with Model Based Acceleration' by Gu et al, 2016. Available from: https://arxiv.org/pdf/1603.00748.pdf

#TODO
#investigate and fix nan action bug
#confirm saver working
#additional observation/action space support (continuous)
#experiment with different advantage functions/covariance matrices for advantage function
#more clever exploration policy
#adaptive batch size?
#memories weighted by information/loss?
#improved network initialisation?
from baselines.deepqnaf import naf
from baselines import logger
import tensorflow as tf
import numpy as np
import random
from tensorflow.python.ops.distributions.util import fill_triangular

class Memory:
  def __init__(self, capacity, batch_size, v):
    self.m = []
    self.ready = 0
    self.full = 0
    self.capacity = capacity
    self.batch_size = batch_size
    self.v = v

  def store(self,d):
    [s,a,r,s_next,terminal] = d
    self.m.append([s,a,r,s_next,terminal])
    if self.full:
      self.m.pop(0)
    if not self.ready and len(self.m) >= self.batch_size:
      self.ready = 1
      if self.v > 0:
        print("[Memory ready]")

  def sample(self):
    return random.sample(self.m, self.batch_size)

class Layer:
  def __init__(self, input_layer, out_n, activation=None, batch_normalize=False):
    x = input_layer
    batch_size, in_n = np.shape(x)
    in_n = int(in_n)
    if batch_normalize:
      variance_epsilon = 0.000001
      decay = 0.999
      self.gamma = tf.Variable(tf.constant(1,shape=[in_n],dtype=tf.float32), trainable=True)
      self.beta = tf.Variable(tf.constant(0,shape=[in_n],dtype=tf.float32), trainable=True)
      self.moving_mean = tf.Variable(tf.constant(0,shape=[in_n],dtype=tf.float32), trainable=False)
      self.moving_var = tf.Variable(tf.constant(1,shape=[in_n],dtype=tf.float32), trainable=False)
      mean,var = tf.nn.moments(x, axes=[0])
      update_mean = self.moving_mean.assign(decay*self.moving_mean + (1-decay)*mean)
      update_var = self.moving_mean.assign(decay*self.moving_var + (1-decay)*var)
      with tf.control_dependencies([update_mean, update_var]):
        x = tf.nn.batch_normalization(x, self.moving_mean, self.moving_var, self.beta, self.gamma, variance_epsilon)

    self.w = tf.Variable(tf.random_uniform([in_n,out_n],-0.1,0.1), trainable=True)
    self.b = tf.Variable(tf.random_uniform([out_n],-0.1,0.1), trainable=True)
    self.z = tf.matmul(x, self.w) + self.b

    if activation is not None:
      self.h = activation(self.z)
    else:
      self.h = self.z

    self.variables = [self.w, self.b]
    if batch_normalize:
      self.variables += [self.gamma, self.beta, self.moving_mean, self.moving_var]

  def construct_update(self, from_layer, tau):
    update = []
    for x,y in zip(self.variables, from_layer.variables):
      update += [x.assign(x*tau + (1-tau)*y)]
    return update

class Agent:
  def __init__(self, v, observation_space, action_space, learning_rate, batch_normalize, gamma, tau, epsilon, hidden_size, hidden_n, hidden_activation, batch_size, memory_capacity, load_path, covariance):
    self.v = v

    self.memory = Memory(memory_capacity,batch_size,v)

    self.observation_space = observation_space
    self.action_space = action_space

    self.state_n = observation_space.shape[0]
    self.action_n = action_space.shape[0]

    self.learning_rate = learning_rate
    self.gamma = gamma
    self.tau = tau
    self.epsilon = epsilon
    self.resets = 0

    H_layer_n = hidden_n
    H_n = hidden_size
    M_n = int((self.action_n)*(self.action_n+1)/2)
    V_n = 1
    mu_n = self.action_n

    tf.reset_default_graph()

    #neural network architecture
    self.x = tf.placeholder(shape=[None,self.state_n], dtype=tf.float32, name="state")
    self.u = tf.placeholder(shape=[None,self.action_n], dtype=tf.float32, name="action")
    self.target = tf.placeholder(shape=[None,1], dtype=tf.float32, name="target")

    self.H = Layer(self.x, H_n, activation=hidden_activation, batch_normalize=batch_normalize)
    self.t_H = Layer(self.x, H_n, activation=hidden_activation, batch_normalize=batch_normalize) #target
    self.updates = self.t_H.construct_update(self.H, self.tau)
    for i in range(1,H_layer_n):
      self.H = Layer(self.H.h, H_n, activation=hidden_activation, batch_normalize=batch_normalize)
      self.t_H = Layer(self.t_H.h, H_n, activation=hidden_activation, batch_normalize=batch_normalize) #target
      self.updates += self.t_H.construct_update(self.H, self.tau)

    self.V = Layer(self.H.h, V_n, batch_normalize=batch_normalize)
    self.t_V = Layer(self.t_H.h, V_n, batch_normalize=batch_normalize) #target
    self.updates += self.t_V.construct_update(self.V, self.tau)
    self.mu = Layer(self.H.h, mu_n, activation=tf.nn.tanh, batch_normalize=batch_normalize)

    if covariance == "identity": #identity covariance
      self.P = tf.eye(self.action_n,batch_shape=[tf.shape(self.x)[0]]) #identity covariance

    elif covariance == "diagonal": #diagonal covariance with nn inputs
      self.O = Layer(self.H.h, mu_n, batch_normalize=batch_normalize) #nn input to diagonal covariance
      self.P = tf.matrix_set_diag(tf.eye(self.action_n,batch_shape=[tf.shape(self.x)[0]]), self.O.h) #diagonal covariance

    else:  #original NAF covariance
      self.M = Layer(self.H.h, M_n, activation=tf.nn.tanh, batch_normalize=batch_normalize)
      self.N = fill_triangular(self.M.h)
      self.L = tf.matrix_set_diag(self.N, tf.exp(tf.matrix_diag_part(self.N)))
      self.P = tf.matmul(self.L, tf.matrix_transpose(self.L)) #original NAF covariance

    #self.P_inverse = tf.matrix_inverse(self.P) #precision matrix for exploration policy

    self.D = tf.reshape(self.u - self.mu.h, [-1,1,self.action_n])

    self.A =  (-1.0/2.0)*tf.reshape(tf.matmul(tf.matmul(self.D, self.P), tf.transpose(self.D, perm=[0,2,1])), [-1,1]) #advantage function

    self.Q = self.A + self.V.h
    self.loss = tf.reduce_sum(tf.square(self.target - self.Q))
    self.optimiser = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    self.sess = tf.Session()

    init = tf.global_variables_initializer()
    self.sess.run(init)

    self.saver = tf.train.Saver()
    if load_path is not None:
      self.saver.restore(self.sess, load_path)


  def reset(self):  #reset in between episodes (for example update epsilon), i is episode number
    if self.memory.ready:
      self.resets = self.resets + 1
      i = self.resets
     # self.epsilon = 1.0/(1+i)
      self.epsilon = 1.0/(1.0+0.1*i+(1.0/(i+1))*np.log(i)) #derived through black magic for inverted double pendulum
      if self.v > 1:
        print("[Update epsilon: " + str(self.epsilon) + "]")
    #self.epsilon = 1.0/(np.log(i+1)/np.log(3) + 0.001) #derived through black magic for inverted double pendulum 2

  def save(self, path):
    self.saver.save(self.sess, path)

  def get_action(self,s):
    mu = self.sess.run(self.mu.h, feed_dict={self.x:np.reshape(s,[1,-1])})[0]

    #random action with probability epsilon
    if np.random.rand() < self.epsilon:
      action = np.random.rand(self.action_n)*2-1 #random action
    else:
      action = mu

    return action

 #  covariance = np.eye(self.action_n)
 #   return self.noise(mu, covariance)

    #mu,p_inv = self.sess.run([self.mu.h,self.P_inverse],feed_dict={self.x:np.reshape(s,[1,-1])})[0]
    #return self.noise(mu, p_inv)

  def observe(self,state,action,reward,state_next,terminal):
    self.memory.store((state,action,reward,state_next,terminal))

  def learn(self):
    if self.memory.ready:
      batch_target = []
      batch_state = []
      batch_action = []
      batch_reward = []
      batch_state_next = []
      batch_terminal = []
      for [t_s,t_a,t_r,t_s_next,t_terminal] in self.memory.sample():
        batch_state_next += [t_s_next]
        batch_state += [t_s]
        batch_action += [t_a]
        batch_reward += [t_r]
        batch_terminal += [t_terminal]
      batch_target = self.get_target(batch_action, batch_reward, batch_state_next, batch_terminal)
      #l,a,self.p_inv = self.backprop(batch_state, batch_action, batch_target)
      l,a = self.backprop(batch_state, batch_action, batch_target)
      self.update_target()
      if (self.v > 2):
        print(np.mean(a))

  def noise(self, mean, covariance):
    return np.random.multivariate_normal(mean,self.epsilon*covariance)

  def scale(self,actions, low, high): #assume domain [-1,1]
    actions = np.clip(actions, -1, 1)
    scaled_actions = []
    for a in actions:
      scaled_actions += [(a+1)*(high-low)/2+low]
    return np.reshape(scaled_actions,[-1]) #range [low,high]

  def get_target(self,a,r,s_next,terminal):
    targets = np.reshape(r,[-1,1]) + np.reshape(self.gamma*self.sess.run(self.t_V.h,feed_dict={self.x:s_next,self.u:a}),[-1,1])
    for i in range(len(terminal)):
     if terminal[i]:
       targets[i] = r[i]
    return targets

  def backprop(self,batch_state,batch_action,batch_target):
    l,a,_ = self.sess.run([self.loss,self.A,self.optimiser],feed_dict={self.x:batch_state, self.target:batch_target, self.u:batch_action})
    return l,a

  def update_target(self):
    for update in self.updates:
      self.sess.run(update)
