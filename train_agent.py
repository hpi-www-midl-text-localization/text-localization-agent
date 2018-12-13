import click
import os
from PIL import Image
import numpy as np
from text_localization_environment import TextLocEnv
import chainer
import chainerrl
import logging
import sys
from tb_chainer import SummaryWriter
import time

@click.command()
@click.option("--steps", "-s", default=2000, help="Amount of steps to train the agent.")
@click.option("--gpu/--no-gpu", default=False)
@click.option("--imagefile", "-i", default='image_locations.txt', help="Path to the file containing the image locations.", type=click.Path(exists=True))
@click.option("--boxfile", "-b", default='bounding_boxes.npy', help="Path to the bounding boxes.", type=click.Path(exists=True))
@click.option("--tensorboard/--no-tensorboard", default=True)
def main(steps, gpu, imagefile, boxfile, tensorboard):
    print(steps)
    print(gpu)
    print(imagefile)
    print(boxfile)
    gpu_number = 0

    if not gpu:
        gpu_number = -1

    locations = np.loadtxt(imagefile, dtype=str)
    images_base_path = os.path.dirname(imagefile)
    images = [Image.open(images_base_path + i.strip('.')) for i in locations]
    bboxes = np.load(boxfile)

    env = TextLocEnv(images, bboxes, gpu)

    obs_size = 4186
    n_actions = env.action_space.n
    q_func = chainerrl.q_functions.FCStateQFunctionWithDiscreteAction(
        obs_size, n_actions,
        n_hidden_layers=2, n_hidden_channels=1024)
    if gpu:
        q_func = q_func.to_gpu(gpu_number)

    # Use Adam to optimize q_func. eps=1e-2 is for stability.
    optimizer = chainer.optimizers.Adam(eps=1e-2)
    optimizer.setup(q_func)

    # Set the discount factor that discounts future rewards.
    gamma = 0.95

    # Use epsilon-greedy for exploration
    explorer = chainerrl.explorers.ConstantEpsilonGreedy(
        epsilon=0.3, random_action_func=env.action_space.sample)

    # DQN uses Experience Replay.
    # Specify a replay buffer and its capacity.
    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)

    # Now create an agent that will interact with the environment.
    agent = chainerrl.agents.DQN(
        q_func, optimizer, replay_buffer, gamma, explorer,
        gpu=gpu_number,
        replay_start_size=500, update_interval=1,
        target_update_interval=100)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

    step_hooks = []
    if tensorboard:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        agentClassName = agent.__class__.__name__[:10]
        writer = SummaryWriter("tensorboard/tensorBoard_exp_" + timestr + "_" + agentClassName)
        step_hooks = [TensorBoardLoggingStepHook(writer)]

    chainerrl.experiments.train_agent_with_evaluation(
        agent, env,
        steps=steps,  # Train the agent for 5000 steps
        eval_n_runs=10,  # 10 episodes are sampled for each evaluation
        max_episode_len=50,  # Maximum length of each episodes
        eval_interval=500,  # Evaluate the agent after every 100 steps
        outdir='result', # Save everything to 'result' directory
        step_hooks=step_hooks)

    agent.save('agent')

class TensorBoardLoggingStepHook(chainerrl.experiments.StepHook):
    def __init__(self, summary_writer):
        self.summary_writer = summary_writer
        return

    def __call__(self, env, agent, step):
        return

if __name__ == '__main__':
    main()
