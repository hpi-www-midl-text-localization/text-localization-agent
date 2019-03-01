import os
import numpy as np
import chainer
import chainerrl
from text_localization_environment import TextLocEnv


def load_agent(env, directory="agent", gpu=0, epsilon=0.3):
    obs_size = 2139
    n_actions = env.action_space.n
    q_func = chainerrl.q_functions.FCStateQFunctionWithDiscreteAction(
        obs_size, n_actions,
        n_hidden_layers=2, n_hidden_channels=1024)

    if gpu != -1:
        q_func = q_func.to_gpu(gpu)

    # Use Adam to optimize q_func. eps=1e-2 is for stability.
    optimizer = chainer.optimizers.Adam(eps=1e-2)
    optimizer.setup(q_func)

    # Set the discount factor that discounts future rewards.
    gamma = 0.95

    # Use epsilon-greedy for exploration
    explorer = chainerrl.explorers.ConstantEpsilonGreedy(
        epsilon=epsilon, random_action_func=env.action_space.sample)

    # DQN uses Experience Replay.
    # Specify a replay buffer and its capacity.
    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)

    # Now create an agent that will interact with the environment.
    agent = chainerrl.agents.DQN(
        q_func, optimizer, replay_buffer, gamma, explorer,
        gpu=gpu,
        replay_start_size=500, update_interval=1,
        target_update_interval=100)

    agent.load(directory)

    return agent


def create_environment(imagefile='image_locations.txt', boxfile='bounding_boxes.npy', gpu=0):
    relative_paths = np.loadtxt(imagefile, dtype=str)
    images_base_path = os.path.dirname(imagefile)
    absolute_paths = [images_base_path + i.strip('.') for i in relative_paths]

    bboxes = np.load(boxfile)

    return TextLocEnv(absolute_paths, bboxes, gpu)


def episode(env, agent):
    obs = env.reset()
    done = False
    R = 0
    t = 0
    while not done and t < 50:
        env.render()
        action = agent.act(obs)
        obs, r, done, _ = env.step(action)
        R += r
        t += 1
        if done:
            print("Done!")
    print('test episode:', 'R:', R)
    agent.stop_episode()
