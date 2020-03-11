import torch
import torch.nn as nn
import numpy as np
import logging
import time
import math
from matplotlib.pyplot import cm

def get_logger(filename):
  """
  Return a logger instance to a file
  """
  logger = logging.getLogger('logger')
  logger.setLevel(logging.DEBUG)
  logging.basicConfig(format='%(message)s', level=logging.DEBUG)
  handler = logging.FileHandler(filename)
  handler.setLevel(logging.DEBUG)
  handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
  logging.getLogger().addHandler(handler)
  return logger


class pol_net(nn.Module):
  """
  Abstract Class for implementing a Policy Gradient Based Algorithm
  """
  def __init__(self, net, logger=None):
    super(pol_net, self).__init__()

    if logger is None:
      self.logger = get_logger('.log.txt')
    else:
      self.logger = logger

    self.net = net
    self.gamma = 0.9
    self.lr = 0.001

    ### HOW LONG ARE WE TRAINING FOR !!!!! CHANGE HERE DON'T HARDCODE ###
    self.num_trainings = 10
    self.num_epochs = 15
    self.num_batches = 10000

    self.initialize_lstm()
    self.linear = nn.Linear(self.lstm_hidden_size, 1, bias = False)
    with torch.no_grad():
      self.linear.weight.data.fill_(-0.1)

    self.log_std = nn.Parameter(torch.empty(1).fill_(-2))
    self.saved_log_probs = []

    self.optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)

    self.color = cm.cool(np.linspace(0,1, self.num_batches * self.num_trainings // self.net.plot_freq + 2 ))

  def initialize_hidden(self):
    self.hiddens_list = []
    for idx in range(self.num_trainings):
      hiddens = []
      for p in self.net.parameters():
        n_params = len(p.view(-1))
        h0 = torch.zeros(1, n_params, self.lstm_hidden_size)
        c0 = torch.zeros(1, n_params, self.lstm_hidden_size)
        hiddens.append((h0,c0))
      self.hiddens_list.append(hiddens) 

  def initialize_lstm(self):
      # note, confusing thing: 
      # The output of an LSTM cell or layer of cells is called the hidden state. 
      # This is confusing, because each LSTM cell retains an internal state that is not output, 
      #  called the cell state, or c. 
      # further, cell state and hidden state need to be the same dimensions
      # 2 options to increase dimensionality of h, c
      #   have them be higher dim but just use the last output of h
      #   I think, maybe equivalently, have a stacked LSTM

      self.lstm_input_size = 1  # [g] or [g, g^2]??
      self.lstm_hidden_size = 32

      # note: "batch" is all the parameters since this is elementwise for each param
      # Input and output thus (batch, seq, feature), but sequence length is 1 for each
      #  call since we are doing this iteratively
      # print("hi!!!!!")
      # state_dict = self.net.state_dict()
      # for key in state_dict:
      #   print("param",self.net.state_dict()[key].grad)
      #   # print("zeros", self.net.state_dict()[key]*0)
      #   state_dict[key].grad = self.net.state_dict()[key].grad * 0 + 1
      #   print("post",self.net.state_dict()[key].grad)
      #   break
      
      self.initialize_hidden()   
      self.lstm = nn.LSTM(
          input_size = self.lstm_input_size,
          hidden_size = self.lstm_hidden_size,
          num_layers = 1,
      )


      ####### HERE IS WHERE TO TOGGLE FOR LSTM ON/OFF #######
  def forward(self, vanilla_grad, hidden):
      x, hidden = self.lstm(vanilla_grad, hidden)
      chocolate_grad = self.linear(x)

      # chocolate_grad = self.linear(vanilla_grad)

      return chocolate_grad, hidden

  def select_actions(self, vanilla_grads, training_it):
      chocolate_grads = []
      tot_log_prob = 0.
      for idx, vanilla_grad in enumerate(vanilla_grads):
        hidden = self.hiddens_list[training_it][idx]
        final_shape = vanilla_grad.shape

        vanilla_grad = vanilla_grad.reshape(1, -1, 1)
        means, hidden = self.forward(vanilla_grad, hidden)
        means = torch.squeeze(means)
        self.hiddens_list[training_it][idx] = hidden

        dist = torch.distributions.normal.Normal(means, torch.exp(self.log_std))
        chocolate_grad = dist.sample()

        tot_log_prob += dist.log_prob(chocolate_grad).sum()

        chocolate_grad = chocolate_grad.reshape(final_shape)
        chocolate_grads.append(chocolate_grad)

      self.saved_log_probs.append(tot_log_prob)

      return chocolate_grads


  def sample_trajectories(self):
      episode_rewards = []
      paths = []

      for training_it in range(self.num_trainings):
          self.net.unlearn(self.color)
          
          # investigate this tabbing !!!
          rewards = []
          ep_reward = 0

          for epoch in range(self.num_epochs):

            for vanilla_grads, pre_loss, images, labels in self.net.train_batch():

              chocolate_grads = self.select_actions(vanilla_grads, training_it)
              self.net.take_grad_step(chocolate_grads)

              # with torch.no_grad():
              #   print('vanilla grad norm', torch.norm(vanilla_grads[0], 2))
              #   print('choc grad norm', torch.norm(chocolate_grads[0], 2))
              
              # # for debugging!!!
              # self.net.take_grad_step(vanilla_grads, alpha = .01)

              with torch.no_grad():
                logits = self.net.forward(images)
                post_loss = self.net.criterion(logits, labels)

              ### REWARD OPTION #1 ### (bad lol)
              # if isinstance(post_loss, np.float64):
              #   reward = - post_loss
              # else:
              #   reward = - post_loss.data

              ### REWARD OPTION #2 ###
              # reward = pre_loss - post_loss

              ### REWARD OPTION #3 ###
              if epoch == self.num_epochs - 1:
                reward = -post_loss
              else:
                reward = 0  
              
              rewards.append(reward)
              ep_reward += reward

          # this tabbing for ellipse time.. maybe???
          episode_rewards.append(ep_reward)
          # print(ep_reward)
          path = {"reward" : np.array(rewards)}

          paths.append(path)

            # print('training iter', training_it, 'epoch', epoch)
            # print('training acc', self.net.train_accuracy(), 'test acc', self.net.test_accuracy())  

      return paths, episode_rewards            


  def get_returns(self, paths):
      all_returns = []
      for path in paths:
        rewards = path["reward"]

        returns = []
        T = len(rewards)
        for t in range(T-1, -1, -1):
          if t == T-1:
            returns.append(rewards[t])
          else:
            returns.append(self.gamma * returns[-1] + rewards[t])

        returns.reverse()

        all_returns.append(returns)

      returns = np.concatenate(all_returns)
      return returns

  def update_pol(self, returns):
    normed_returns = (returns-np.mean(returns))/(np.std(returns)+1e-10)
    policy_loss = -1 * (self.saved_log_probs * normed_returns).sum()
    #print("policy loss", policy_loss.data)
    
    self.optimizer.zero_grad()
    policy_loss.backward()
    for name, p in self.named_parameters():
      if "linear" in name:
        self.lin_weights.append(p.data.numpy()[0][0])
    #     print("lin value:", p.data)
    #     print("lin grad:", p.grad)
    #   if name == "log_std":
    #     print('log_std value:', p.data)
    #     print('log_std grad:', p.grad)
      # else:
      #   print(name, p.data)  

    self.optimizer.step()

    self.saved_log_probs = []

  def train(self):
      print("we have begun to train....")
      last_eval = 0
      last_record = 0
      scores_eval = []

      self.avg_reward = 0.

      scores_eval = [] # list of scores computed at iteration time

      self.avg_rewards = []
      self.sigma_rewards = []
      self.lin_weights = []

      for t in range(self.num_batches):
        if t % 1000 == 0:
          print("batch", t)
        start = time.time()
        #print("batch ", t)
        # collect a minibatch of samples
        paths, episode_rewards = self.sample_trajectories()

        scores_eval = scores_eval + episode_rewards
        rewards = np.concatenate([path["reward"] for path in paths])

        returns = self.get_returns(paths)
        self.update_pol(returns)

        # compute reward statistics for this batch and log
        avg_reward = np.mean(episode_rewards)
        self.avg_rewards.append(avg_reward)

        sigma_reward = np.sqrt(np.var(episode_rewards) / len(episode_rewards))
        self.sigma_rewards.append(sigma_reward)

        #msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
        #self.logger.info(msg)

        self.initialize_hidden()

        # if  self.config.record and (last_record > self.config.record_freq):
        #   self.logger.info("Recording...")
        #   last_record =0
        #   self.record()
        end = time.time()
        #print('batch ' + str(t) +  ' took : ' + str(-start + end))
      self.logger.info("- Training done.")

      #export_plot(scores_eval, "Score", self.config.env_name, self.config.plot_output)

