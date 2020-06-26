"""Reinforment learning DQN and Dueling DQN
CS489 project
Author: Lu Jiarui 
Date: 2020/06/11
"""

from __future__ import print_function
import torch
import argparse

from agent import Agent

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description='DQN-Atari')

parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate (default: 1e-3')
parser.add_argument('--batch-size', type=int, default=32,
                    help='training batch size from buffer')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--dueling', type=bool, default=False,
                    help='whether to choose dueling DQN')
parser.add_argument('--do-test', type=bool, default=True,
                    help='load local checkpoint and do predict(go on an infinite episode)')
parser.add_argument('--init-checkpoint', type=str, default='dqn_boxing_model.ckpt',
                    help='load model ckpt and predict or train')               
parser.add_argument('--env-name', type=str, default='BoxingNoFrameskip-v4',
                    help='The rl algorithm for atari game to learn')


if __name__ == "__main__":
    args = parser.parse_args()
    agent = Agent(args.env_name, args.batch_size, args.gamma, args.lr, target_update=1000, initial_memory=1000, memory_size=10000, dueling=args.dueling)
    # train model
    if args.do_test:
        print('[INFO] Starting testing subroutine. (Default: Test)')
        # load local checkpoint
        agent.policy_net = torch.load(args.init_checkpoint, map_location=device)
        agent.test()
    else:
        print('[INFO] Starting training subroutine.')
        agent.train(1000)
        torch.save(agent.policy_net, "dqn_{}_model.ckpt".format(args.env_name))
