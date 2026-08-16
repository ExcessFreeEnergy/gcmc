"""
High-throughput PPO / PuffeRL training for cDFT fluid manipulation.
"""

import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gcmc.envs.cdft_puffer import BatchedCdftVecEnv, CDFT_OBS_SIZE, CDFT_NUM_ACTIONS


class CdftContinuousPolicy(nn.Module):
    """
    Gaussian Actor-Critic policy for continuous fluid manipulation.
    """
    def __init__(self, obs_dim=CDFT_OBS_SIZE, act_dim=CDFT_NUM_ACTIONS, hidden_dim=128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        return self.actor(obs)

    def get_action_and_value(self, obs, action=None):
        mean = self.actor(obs)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, entropy, value


def train_cdft_puffer(
    num_envs=128,
    num_steps=128,
    total_timesteps=100000,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_coef=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    print(f"[PuffeRL] Initializing {num_envs} vectorized cDFT environments on device: {device}...")
    vec_env = BatchedCdftVecEnv(num_envs=num_envs)
    policy = CdftContinuousPolicy().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate, eps=1e-5)

    obs_tensor = torch.zeros((num_steps, num_envs, CDFT_OBS_SIZE), device=device)
    act_tensor = torch.zeros((num_steps, num_envs, CDFT_NUM_ACTIONS), device=device)
    logprob_tensor = torch.zeros((num_steps, num_envs), device=device)
    rew_tensor = torch.zeros((num_steps, num_envs), device=device)
    done_tensor = torch.zeros((num_steps, num_envs), device=device)
    val_tensor = torch.zeros((num_steps, num_envs), device=device)

    next_obs, _ = vec_env.reset()
    next_obs = torch.tensor(next_obs, device=device)
    next_done = torch.zeros(num_envs, device=device)

    num_iterations = total_timesteps // (num_envs * num_steps)
    global_step = 0
    t_start = time.perf_counter()

    print(f"[PuffeRL] Starting training: {num_iterations} iterations, target timesteps = {total_timesteps:,}")

    for iteration in range(1, num_iterations + 1):
        # 1. Rollout phase
        for step in range(num_steps):
            global_step += num_envs
            obs_tensor[step] = next_obs
            done_tensor[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = policy.get_action_and_value(next_obs)
                val_tensor[step] = value

            act_tensor[step] = action
            logprob_tensor[step] = logprob

            # Step vectorized environment
            act_numpy = action.cpu().numpy()
            act_numpy = np.clip(act_numpy, -1.0, 1.0)
            obs_np, rews_np, terms_np, _ = vec_env.step(act_numpy)

            rew_tensor[step] = torch.tensor(rews_np, device=device)
            next_obs = torch.tensor(obs_np, device=device)
            next_done = torch.tensor(terms_np, device=device, dtype=torch.float32)

        # 2. GAE Advantage estimation
        with torch.no_grad():
            _, _, _, next_val = policy.get_action_and_value(next_obs)
            advantages = torch.zeros_like(rew_tensor, device=device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_val
                else:
                    nextnonterminal = 1.0 - done_tensor[t + 1]
                    nextvalues = val_tensor[t + 1]
                delta = rew_tensor[t] + gamma * nextvalues * nextnonterminal - val_tensor[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + val_tensor

        # 3. PPO Optimization
        b_obs = obs_tensor.reshape(-1, CDFT_OBS_SIZE)
        b_act = act_tensor.reshape(-1, CDFT_NUM_ACTIONS)
        b_logprobs = logprob_tensor.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        _, new_logprob, entropy, new_val = policy.get_action_and_value(b_obs, b_act)
        logratio = new_logprob - b_logprobs
        ratio = logratio.exp()

        pg_loss1 = -b_advantages * ratio
        pg_loss2 = -b_advantages * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        v_loss = 0.5 * ((new_val - b_returns) ** 2).mean()
        loss = pg_loss - ent_coef * entropy.mean() + vf_coef * v_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()

        if iteration % 5 == 0 or iteration == num_iterations:
            elapsed = time.perf_counter() - t_start
            sps = global_step / elapsed
            avg_rew = rew_tensor.mean().item()
            print(f"Iter {iteration:4d}/{num_iterations} | Step: {global_step:7,d} | SPS: {sps:,.0f} | Avg Rew: {avg_rew:7.3f} | Loss: {loss.item():7.4f}")

    print(f"\n[PuffeRL] Training complete in {time.perf_counter() - t_start:.2f}s! Final SPS: {global_step / (time.perf_counter() - t_start):,.0f}")
    return policy


CdftPolicy = CdftContinuousPolicy


def save_policy_checkpoint(policy, path="cdft_policy.pt"):
    torch.save(policy.state_dict(), path)
    print(f"[PuffeRL] Saved policy checkpoint to '{path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PuffeRL vectorized training for cDFT fluid manipulation.")
    parser.add_argument("--num_envs", type=int, default=128, help="Number of parallel environments.")
    parser.add_argument("--total_timesteps", type=int, default=100000, help="Total environment timesteps.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--save_path", type=str, default="cdft_policy.pt", help="Path to save trained policy checkpoint.")
    parser.add_argument("-i", "--interactive", action="store_true", default=False, help="Launch interactive Raylib UI.")
    args = parser.parse_args()

    if args.interactive:
        from gcmc.ui import launch_interactive_cdft_rl
        launch_interactive_cdft_rl(policy_path=args.save_path if os.path.exists(args.save_path) else None)
    else:
        policy = train_cdft_puffer(num_envs=args.num_envs, total_timesteps=args.total_timesteps, learning_rate=args.lr)
        save_policy_checkpoint(policy, args.save_path)

