"""
This file provides helper functions to visualize your agent.

-----

2022-2023 fall quarter, CS269 Seminar 5: Reinforcement Learning.
Department of Computer Science at University of California, Los Angeles.
Course Instructor: Professor Bolei ZHOU.
Assignment Author: Zhenghao PENG.
"""
import tempfile

import IPython
import PIL
import gym
import numpy as np
from IPython.display import clear_output

from core.ppo_trainer import PPOTrainer, PPOConfig
from core.td3_trainer import TD3Trainer


def evaluate(
        policy, num_episodes=1, seed=0, env_name='FrozenLake8x8-v1',
        render=None, existing_env=None, max_episode_length=1000,
        verbose=False
):
    """
    This function evaluate the given policy and return the mean episode
    reward.

    This function does not support vectorized (stacked) environments.

    :param policy: a function whose input is the observation
    :param num_episodes: number of episodes you wish to run
    :param seed: the random seed
    :param env_name: the name of the environment
    :param render: a boolean flag indicating whether to render policy
    :return: the averaged episode reward of the given policy.
    """
    if existing_env is None:
        env = gym.make(env_name)
        env.seed(seed)
    else:
        env = existing_env

    try:

        rewards = []
        frames = []
        if render: num_episodes = 1
        for i in range(num_episodes):
            obs = env.reset()
            act = policy(obs)
            ep_reward = 0
            for step_count in range(max_episode_length):
                obs, reward, done, info = env.step(act)
                act = policy(obs)
                ep_reward += reward

                if verbose and step_count % 50 == 0:
                    print("Evaluating {}/{} episodes. We are in {}/{} steps. Current episode reward: {:.3f}".format(
                        i + 1, num_episodes, step_count + 1, max_episode_length, ep_reward
                    ))

                if render:
                    frames.append(env.render(render))
                if done:
                    print("Evaluating {}/{} episodes. Episode is done in {} steps. Episode reward: {:.3f}".format(
                        i + 1, num_episodes, step_count + 1, ep_reward
                    ))
                    break
            rewards.append(ep_reward)

    finally:
        env.close()

    return np.mean(rewards), {"frames": frames}


def evaluate_in_batch(policy, envs, num_episodes=1):
    """
    This function evaluate the given policy and return the mean episode
    reward.

    This function does not support single environment, must be vectorized environment.

    :param policy: a function whose input is the observation
    :param envs: a vectorized environment
    :param num_episodes: number of episodes you wish to run
    :return: the averaged episode reward of the given policy.
    """
    num_envs = envs.num_envs
    total_episodes = 0
    batch_steps = 0
    rewards = []
    successes = []

    try:

        obs = envs.reset()

        episode_rewards = np.ones(num_envs)

        while total_episodes < num_episodes:

            batch_steps += 1

            actions = policy(obs)
            obs, reward, done, info = envs.step(actions)

            episode_rewards_old_shape = episode_rewards.shape

            episode_rewards += reward.reshape(episode_rewards.shape)
            for idx, d in enumerate(done):
                if d:  # the episode is done
                    # Record the reward of the terminated episode to
                    rewards.append(episode_rewards[idx].copy())
                    successes.append(info[idx].get("arrive_dest", 0))
                    total_episodes += 1
                    if total_episodes % 10 == 0:
                        print("Finished {}/{} episodes. Average episode reward: {:.3f}".format(
                            total_episodes, num_episodes, np.mean(rewards)
                        ))
                    if total_episodes < num_episodes:
                        break

            masks = 1. - done.astype(np.float32)

            episode_rewards *= masks.reshape(-1, )

            assert episode_rewards.shape == episode_rewards_old_shape

    finally:
        envs.close()

    assert len(rewards) == num_episodes

    return np.mean(rewards), {
        "successes": successes,
        "rewards": rewards,
        "std": np.std(rewards),
        "mean": np.mean(rewards)
    }

def animate(img_array, duration=0.05):
    """A function that can generate GIF file and show in Notebook."""
    path = tempfile.mkstemp(suffix=".gif")[1]
    images = [PIL.Image.fromarray(frame) for frame in img_array]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    with open(path, "rb") as f:
        IPython.display.display(
            IPython.display.Image(data=f.read(), format='png'))



class PPOPolicy:
    """
    This class wrap an agent into a callable function that return action given
    a raw observation or a batch of raw observations from environment.
    """

    def __init__(self, env_id, num_envs=1, log_dir=None, suffix=None):
        if "MetaDrive" in env_id:
            from core.utils import register_metadrive
            register_metadrive()
        env = gym.make(env_id)
        self.agent = PPOTrainer(env, PPOConfig())
        if log_dir is not None:  # log_dir is None only in testing
            success = self.agent.load_w(log_dir, suffix)
            if not success:
                raise ValueError("Failed to load agent!")
        self.num_envs = num_envs

    def reset(self):
        pass

    def __call__(self, obs):
        action = self.agent.compute_action(obs)[1]
        action = action.detach().cpu().numpy()
        if self.num_envs == 1:
            return action[0]
        return action


class TD3Policy:
    """
    This class wrap an agent into a callable function that return action given
    an raw observation or a batch of raw observations from environment.

    """

    def __init__(self, env_id, num_envs=1, log_dir=None, suffix=None):
        self.agent = TD3Trainer()
        if log_dir is not None:  # log_dir is None only in testing
            success = self.agent.load_w(log_dir, suffix)
            if not success:
                raise ValueError("Failed to load agent!")
        self.num_envs = num_envs

    def reset(self):
        pass

    def __call__(self, obs):
        action = self.agent.compute_action(obs)[1]
        action = action.detach().cpu().numpy()
        if self.num_envs == 1:
            return action[0]
        return action
