import torch.nn as nn
import numpy as np

class PG(nn.Module):
  """
  Abstract Class for implementing a Policy Gradient Based Algorithm
  """
  def __init__(self, net, logger=None):
    super(PG, self).__init__()

    self.logger = logger
    if logger is None:
      self.logger = get_logger('./')

    self.net = net
    self.gamma = 0.9

    self.window = 3
    self.linear = nn.Linear(2 * self.window, 1)
    self.log_std = nn.Parameter(torch.empty(1).fill_(-10.0))

    self.saved_log_probs = []

    # self.observation_dim = self.env.observation_space.shape[0]

def forward(self, x):
    x = self.linear(x)
    return x

def select_action(self, state):
    mean = self.forward(state)

    dist = torch.distributions.normal.Normal(mean, torch.exp(self.log_std))
    action = dist.rsample()

    self.saved_log_probs.append(dist.log_prob(action))

    return action.item()

    # m = Categorical(probs)
    # action = m.sample()
    # policy.saved_log_probs.append(m.log_prob(action))
    # return action.item()

def sample_trajectories(self):
    self.batch_size = 1
    self.n_epochs = 4

    episode_rewards = []
    paths = []

    for batch_idx in range(self.batch_size):
        self.net.unlearn()

        base = 1.0 / len(self.net.classes)
        state = [base for i in range(2 * self.window)] # [tr_1, te_1, tr_2, te_2, tr_3, te_3]
        states, actions, rewards = [], [], []
        ep_reward = 0

        for epoch in range(self.n_epochs):
            states.append(state)
            log_lr = self.select_action(state)

            self.net.train_epoch(epoch, torch.exp(log_lr))

            train_a = self.net.train_accuracy()
            test_a = self.net.test_accuracy()

            if self.window > 1:
                state = state[2:] + [train_a, test_a]
            else:
                state = [train_a, test_a]

            actions.append(log_lr)
            rewards.append(test_a)
            ep_reward += test_a

        episode_rewards.append(ep_reward)

        path = {"observation" : np.array(states),
                "reward" : np.array(rewards),
                "action" : np.array(actions)}
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

def update_pol(self):
  normed_returns = (returns-np.mean(returns))/(np.std(returns)+1e-10)
  policy_loss = -1 * (self.saved_log_probs * normed_returns).sum()

  optimizer.zero_grad()
  policy_loss.backward()
  optimizer.step()
  self.saved_log_probs = []

def train(self):
    """
    Performs training

    You do not have to change or use anything here, but take a look
    to see how all the code you've written fits together!
    """
    last_eval = 0
    last_record = 0
    scores_eval = []

    self.avg_reward = 0.

    scores_eval = [] # list of scores computed at iteration time

    self.num_batches = 5
    for t in range(self.num_batches):

      # collect a minibatch of samples
      paths, total_rewards = self.sample_trajectories() 

      scores_eval = scores_eval + total_rewards
      observations = np.concatenate([path["observation"] for path in paths])
      actions = np.concatenate([path["action"] for path in paths])
      rewards = np.concatenate([path["reward"] for path in paths])
      # compute Q-val estimates (discounted future returns) for each time step
      returns = self.get_returns(paths)

      # advantage will depend on the baseline implementation
      # advantages = self.calculate_advantage(returns, observations)
      # IF NO BASELINE:
      
      ## TODO: SUM OVER BATCH SIZES?
      
      self.update_pol(returns)

      # run training operations
      # if self.config.use_baseline:
      #   self.baseline_network.update_baseline(returns, observations)
      
      # self.sess.run(self.train_op, feed_dict={
      #               self.observation_placeholder : observations,
      #               self.action_placeholder : actions,
      #               self.advantage_placeholder : advantages})

      # tf stuff
      # if (t % self.config.summary_freq == 0):
      #   self.update_averages(total_rewards, scores_eval)
      #   self.record_summary(t)

      # compute reward statistics for this batch and log
      avg_reward = np.mean(total_rewards)
      sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
      msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
      self.logger.info(msg)

      if  self.config.record and (last_record > self.config.record_freq):
        self.logger.info("Recording...")
        last_record =0
        self.record()

    self.logger.info("- Training done.")
    export_plot(scores_eval, "Score", self.config.env_name, self.config.plot_output)

