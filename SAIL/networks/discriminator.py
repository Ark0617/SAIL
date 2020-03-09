import torch
import torch.nn as nn
from SAIL.misc.utils import obs_batch_normalize, log_sum_exp, RunningMeanStd


class Discriminator(nn.Module):
    def __init__(self, ob_dim, ac_dim, hidden_dim, device):
        super(Discriminator, self).__init__()
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        input_dim = ob_dim + ac_dim
        self.tower = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(),
                                   nn.Linear(input_dim, hidden_dim), nn.Tanh(),
                                   nn.Linear(hidden_dim, 1, bias=False))
        self.logZ = nn.Parameter(torch.ones(1))
        self.input_rms = RunningMeanStd(shape=input_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        self.device = device
        self.to(device)
        self.train()

    def forward(self):
        raise NotImplementedError()

    def update(self, ac_eval_fn, exp_buffer, rollouts, num_grad_steps):
        self.train()
        exp_data_gen = exp_buffer.data_gen_finite(len(exp_buffer)//num_grad_steps)
        pol_data_gen = rollouts.feed_forward_generator(fetch_normalized=False, advantages=None,
                                                       mini_batch_size=rollouts.num_steps//num_grad_steps)
        loss_val = 0
        n = 0
        for _ in range(num_grad_steps):
            pol_batch = next(pol_data_gen)
            exp_batch = next(exp_data_gen)
            exp_state, exp_action = exp_batch
            pol_state, pol_action, pol_log_probs = pol_batch[0], pol_batch[2], pol_batch[7]
            exp_state_normalized = obs_batch_normalize(exp_state, update_rms=False, rms_obj=rollouts.ob_rms)
            with torch.no_grad():
                exp_log_probs = ac_eval_fn(exp_state_normalized, rnn_hxs=None, masks=None, action=exp_action, pretanh_action=exp_action)[1]
            pol_log_probs = pol_log_probs.detach() / self.ac_dim
            exp_log_probs = exp_log_probs.detach() / self.ac_dim
            pol_sa = torch.cat([pol_state, pol_action], dim=1)
            exp_sa = torch.cat([exp_state, exp_action], dim=1)
            normalized_sa = obs_batch_normalize(torch.cat([pol_sa, exp_sa], dim=0), update_rms=True, rms_obj=self.input_rms)
            pol_sa, exp_sa = torch.split(normalized_sa, [pol_state.size(0), exp_state.size(0)], dim=0)
            pol_logp = self.tower(pol_sa)
            exp_logp = self.tower(exp_sa)

            pol_logq = pol_log_probs + self.logZ.expand_as(pol_log_probs)
            exp_logq = exp_log_probs + self.logZ.expand_as(exp_log_probs)
            pol_log_pq = torch.cat([pol_logp, pol_logq], dim=1)
            pol_log_pq = log_sum_exp(pol_log_pq, dim=1, keepdim=True)
            exp_log_pq = torch.cat([exp_logp, exp_logq], dim=1)
            exp_log_pq = log_sum_exp(exp_log_pq, dim=1, keepdim=True)
            pol_loss = -(pol_logq - pol_log_pq).mean(0)
            exp_loss = -(exp_logp - exp_log_pq).mean(0)
            reward_bias = (-torch.cat([pol_logp, exp_logp], dim=0)).clamp_(min=0).mean(0)
            loss = exp_loss + pol_loss + 2*reward_bias

            loss_val += loss.item()
            n += 1
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.)
            self.optimizer.step()
        return loss_val / n

    def predict_batch_rewards(self, rollouts):
        obs = rollouts.raw_obs[:-1].view(-1, self.ob_dim)
        acs = rollouts.actions.view(-1, self.ac_dim)
        sa = torch.cat([obs, acs], dim=1)
        sa = obs_batch_normalize(sa, update_rms=False, rms_obj=self.input_rms)
        with torch.no_grad():
            self.eval()
            rewards = self.tower(sa).view(rollouts.num_steps, -1, 1)
            rollouts.rewards.copy(rewards)