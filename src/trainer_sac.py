from utils import evaluate_policy, Action_adapter, Action_adapter_reverse, Reward_adapter
from datetime import datetime
from algorithms.sac_continuous import SacCountinuous
import gymnasium as gym
import os
import shutil
import torch
import logging

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, name, dvc, env, eval_env, seed, max_train_episodes, save_interval,
                 eval_interval, write, load_model, actor_model_file, critic_model_file,
                 gamma, net_width, a_lr, c_lr, batch_size, alpha, adaptive_alpha):
        logger.info(f'Init Trainer: {name}')
        self.name = name
        self.env = env
        self.eval_env = eval_env
        self.max_train_episodes = max_train_episodes
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.write = write
        self.load_model = load_model
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.max_action = float(env.action_space.high[0])
        self.max_e_steps = env._max_episode_steps
        logger.info(f'Env:env,  state_dim:{self.state_dim},  action_dim:{self.action_dim},'
                    f'max_a:{self.max_action},  min_a:{self.env.action_space.low[0]},  max_e_steps:{self.max_e_steps}')

        self.env_seed = seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Random Seed: {seed}")

        if write:
            from torch.utils.tensorboard import SummaryWriter
            timenow = str(datetime.now())[0:-10]
            timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
            writepath = f'runs/{name}-{timenow}'
            if os.path.exists(writepath):
                shutil.rmtree(writepath)
            self.writer = SummaryWriter(log_dir=writepath)

        if not os.path.exists('model'):
            os.mkdir('model')

        self.agent = SacCountinuous(
            state_dim=self.state_dim, action_dim=self.action_dim, net_width=net_width,
            gamma=gamma, a_lr=a_lr, c_lr=c_lr, alpha=alpha, batch_size=batch_size,
            dvc=dvc, adaptive_alpha=adaptive_alpha)

        if (load_model):
            actor_model_file_path = f'./model/{actor_model_file}.pth'
            critic_model_file_path = f'./model/{critic_model_file}.pth'
            logger.info(
                f'Load model from {actor_model_file_path} and {critic_model_file_path}')
            if not os.path.exists(actor_model_file_path) or not os.path.exists(critic_model_file_path):
                logger.error('No such model file')
                raise FileNotFoundError
            self.agent.load(actor_model_file_path, critic_model_file_path)

    def start(self):
        logger.info(f'Start Trainer: {self.name}')
        total_steps = 0
        total_episodes = 0
        last_episode_total_steps = 0
        eval_steps_count = 0
        while total_episodes < self.max_train_episodes:
            s, info = self.env.reset(seed=self.env_seed)
            self.env_seed += 1
            done = False

            '''Interact & trian'''
            while not done:
                if total_steps < (5*self.max_e_steps):
                    act = self.env.action_space.sample()
                    a = Action_adapter_reverse(act, self.max_action)
                else:
                    a = self.agent.select_action(s, deterministic=False)
                    act = Action_adapter(a, self.max_action)

                s_next, r, dw, tr, info = self.env.step(act)
                r = Reward_adapter(r, 1)
                done = (dw or tr)
                self.agent.replay_buffer.add(s, a, r, s_next, dw)
                s = s_next
                total_steps += 1

            total_episodes += 1
            this_episode_total_steps = total_steps - last_episode_total_steps
            eval_steps_count += this_episode_total_steps
            '''train'''
            if (total_steps >= 2*self.max_e_steps or self.load_model):
                logger.info(
                    f"Train at episode {total_episodes}, total steps: {total_steps}, train times: {this_episode_total_steps}")
                for j in range(this_episode_total_steps):
                    self.agent.train()

            ep_r = 0
            '''record & log'''
            if eval_steps_count >= self.eval_interval:
                logger.info(
                    f'Evaluate at episode {total_episodes}, steps: {total_steps}')
                ep_r = evaluate_policy(self.eval_env, self.agent, turns=3)
                if self.write:
                    self.writer.add_scalar(
                        'ep_r', ep_r, global_step=total_steps)
                eval_steps_count = 0

            '''save model'''
            if total_episodes % self.save_interval == 0:
                logger.info(
                    f"Save model at episode {total_episodes}, steps: {total_steps}")
                self.agent.save(self.name, int(total_steps/1000))

            logger.info(
                f'EnvName:{self.name}, Episode: {total_episodes}, Steps: {int(total_steps/1000)}k, Episode Reward:{ep_r}')

            last_episode_total_steps = total_steps

        logger.info(f'End Trainer: {self.name}')

    def __del__(self):
        if self.write:
            self.writer.close()
        self.env.close()
        self.eval_env.close()
        logger.info(f'Del Trainer: {self.name}')
