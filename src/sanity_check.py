from utils import str2bool, evaluate_policy, Action_adapter, Action_adapter_reverse, Reward_adapter
from datetime import datetime
import gymnasium as gym
import os
import shutil
import argparse
import torch
from trainer import Trainer
import logging

logger = logging.getLogger(__name__)

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda',
                    help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0,
                    help='PV1, Lch_Cv2, Humanv4, HCv4, BWv3, BWHv3')
parser.add_argument('--write', type=str2bool, default=False,
                    help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool,
                    default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool,
                    default=False, help='Load pretrained model or Not')
parser.add_argument('--actor_model', type=str, default='actor',
                    help='which actor model to load')
parser.add_argument('--critic_model', type=str, default='critic',
                    help='which critic model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--max_train_episodes', type=int,
                    default=int(1e3), help='Max training episodes')
parser.add_argument('--save_interval', type=int,
                    default=int(50), help='Model saving interval, in episodes.')
parser.add_argument('--eval_interval', type=int, default=int(50),
                    help='Model evaluating interval, in episodes.')

# some hyperparameters for SAC
parser.add_argument('--gamma', type=float, default=0.99,
                    help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=256,
                    help='Hidden net width, s_dim-400-300-a_dim')
parser.add_argument('--a_lr', type=float, default=3e-4,
                    help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=3e-4,
                    help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=256,
                    help='batch_size of training')
parser.add_argument('--alpha', type=float, default=0.12,
                    help='Entropy coefficient')
parser.add_argument('--adaptive_alpha', type=str2bool,
                    default=True, help='Use adaptive_alpha or Not')
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc)  # from str to torch.device
logger.info(opt)


def main():
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    EnvName = ['Pendulum-v1', 'LunarLanderContinuous-v2', 'Humanoid-v4',
               'HalfCheetah-v4', 'BipedalWalker-v3', 'BipedalWalkerHardcore-v3']
    BrifEnvName = ['PV1', 'LLdV2', 'Humanv4', 'HCv4', 'BWv3', 'BWHv3']

    logger.info(f"Start Training on {BrifEnvName[opt.EnvIdex]}")

    # Build Env
    env = gym.make(EnvName[opt.EnvIdex],
                   render_mode="human" if opt.render else None)
    eval_env = gym.make(EnvName[opt.EnvIdex])

    trainer = Trainer(EnvName[opt.EnvIdex], opt.dvc, env, eval_env, opt.seed, opt.max_train_episodes, opt.save_interval,
                      opt.eval_interval, opt.write, opt.Loadmodel, opt.actor_model, opt.critic_model,
                      opt.gamma, opt.net_width, opt.a_lr, opt.c_lr, opt.batch_size, opt.alpha, opt.adaptive_alpha)

    trainer.start()


if __name__ == '__main__':
    main()
