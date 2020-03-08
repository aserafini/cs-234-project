import torch
import torch.nn as nn
import numpy as np
import logging

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

    self.initialize_lstm()
    self.linear = nn.Linear(self.lstm_hidden_size, 1, bias = False)

    self.log_std = nn.Parameter(torch.empty(1).fill_(-2.0))
    self.saved_log_probs = []

    self.optimizer = torch.optim.Adam(self.parameters())

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
      self.lstm_hidden_size = 8
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
      self.hiddens = []
      for p in self.net.parameters():
        n_params = len(p.view(-1))
        h0 = torch.zeros(1, n_params, self.lstm_hidden_size)
        c0 = torch.zeros(1, n_params, self.lstm_hidden_size)
        self.hiddens.append((h0,c0))

      self.lstm = nn.LSTM(
          input_size = self.lstm_input_size,
          hidden_size = self.lstm_hidden_size,
          num_layers = 1,
      )


  def forward(self, vanilla_grad, hidden):
      # print("vanilla shpae", vanilla_grad.shape)
      # print("hidden shape", hidden[0].shape)
      x, hidden = self.lstm(vanilla_grad, hidden)
      chocolate_grad = self.linear(x)

      return chocolate_grad, hidden

  def select_actions(self, vanilla_grads):
      chocolate_grads = []
      tot_log_prob = 0.
      for idx, vanilla_grad in enumerate(vanilla_grads):
        hidden = self.hiddens[idx]
        final_shape = vanilla_grad.shape

        vanilla_grad = vanilla_grad.reshape(1, -1, 1)
        means, hidden = self.forward(vanilla_grad, hidden)
        means = torch.squeeze(means)
        self.hiddens[idx] = hidden

        # make list for compute graph???
        dist = torch.distributions.normal.Normal(means, torch.exp(self.log_std))

        chocolate_grad = dist.rsample()
        tot_log_prob += dist.log_prob(chocolate_grad).sum()

        chocolate_grad = chocolate_grad.reshape(final_shape)
        chocolate_grads.append(chocolate_grad)

      self.saved_log_probs.append(tot_log_prob)
      return chocolate_grads #.item()


  def sample_trajectories(self):
      self.batch_size = 10
      self.n_epochs = 1

      episode_rewards = []
      paths = []


      for batch_idx in range(self.batch_size):
          self.net.unlearn()

          rewards = []
          ep_reward = 0


          for epoch in range(self.n_epochs):
            for vanilla_grads in self.net.train_batch():

              chocolate_grads = self.select_actions(vanilla_grads)

              # new_loss = self.net.take_grad_step(chocolate_grads)
              self.net.take_grad_step(chocolate_grads)
              new_loss = self.net.total_loss()

              # FIX THIS: reward = curr_loss - new_loss

              rewards.append(reward)
              ep_reward += reward 

          episode_rewards.append(ep_reward)

          path = {"reward" : np.array(rewards)}

          paths.append(path)

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
    print("policy loss", policy_loss)

    self.optimizer.zero_grad()
    policy_loss.backward()
    self.optimizer.step()

    self.saved_log_probs = []

  def train(self):
      """
      Performs training

      You do not have to change or use anything here, but take a look
      to see how all the code you've written fits together!
      """
      print("we have begun to train....")
      last_eval = 0
      last_record = 0
      scores_eval = []

      self.avg_reward = 0.

      scores_eval = [] # list of scores computed at iteration time

      self.num_batches = 5
      for t in range(self.num_batches):
        print("batch ", t)
        # collect a minibatch of samples
        paths, total_rewards = self.sample_trajectories() 

        scores_eval = scores_eval + total_rewards
        # observations = np.concatenate([path["observation"] for path in paths])
        # actions = np.concatenate([path["action"] for path in paths])
        rewards = np.concatenate([path["reward"] for path in paths])

        returns = self.get_returns(paths)
        
        self.update_pol(returns)

        # compute reward statistics for this batch and log
        avg_reward = np.mean(total_rewards)
        sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
        msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
        self.logger.info(msg)

        # if  self.config.record and (last_record > self.config.record_freq):
        #   self.logger.info("Recording...")
        #   last_record =0
        #   self.record()

      self.logger.info("- Training done.")
      #export_plot(scores_eval, "Score", self.config.env_name, self.config.plot_output)

