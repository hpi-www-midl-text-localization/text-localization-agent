from text_localization_environment import TextLocEnv

import os
import sys
import click
import chainer
import chainerrl
import numpy as np
from tqdm import tqdm

def create_environment(imagefile='image_locations.txt', boxfile='bounding_boxes.npy', gpu=0):
    locations = np.loadtxt(imagefile, dtype=str).tolist()
    bboxes = np.load(boxfile)

    env = TextLocEnv(locations, bboxes, gpu)

    return env


def load_agent(env, directory="agent", gpu=0, epsilon=0.3):
    obs_size = 4187
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

@click.command()
@click.option("--gpu", default=-1, help="ID of the GPU to be used. -1 if the CPU should be used instead.")
@click.option("--imagefile", "-i", default='image_locations.txt', help="Path to the file containing the image locations. Has to contain exactly 1 image!", type=click.Path(exists=True))
@click.option("--boxfile", "-b", default='bounding_boxes.npy', help="Path to the bounding boxes. Has to be of exactly 1 image!", type=click.Path(exists=True))
@click.option("--agentdirectory", "-a", default='./agent', help="Path to the directory containing the agent that should be executed.", type=click.Path(exists=True))
def generate_image_sequence(gpu, imagefile, boxfile, agentdirectory):
    """
    Usage:
    * Generate a dataset with exactly one image
      $ cd dataset-generator/ && python main.py -c1
    * Generate step images for the agent that should be run
      $ python load_agent.py (params: see @click.options)
    * Generate a video out of the images using ffmpeg:
      $ ffmpeg -framerate 2 -i human/%03d.png \
        -framerate 2 -i box/%03d.png \
        -filter_complex "[0:v]scale=224:-1,pad=iw+6:ih:color=white[v0];[v0][1:v]hstack=inputs=2" \
        -pix_fmt yuv420p \
        output.mp4
    """
    max_steps_per_image = 50

    relative_paths = np.loadtxt(imagefile, dtype=str)
    images_base_path = os.path.dirname(imagefile)
    absolute_paths = [images_base_path + relative_paths.item().strip('.')]

    gt_bboxes = np.load(boxfile)

    env = TextLocEnv(absolute_paths, gt_bboxes, gpu)
    agent = load_agent(env, agentdirectory, gpu, epsilon = 0.0)

    images_human = []
    images_box = []

    obs = env.reset()

    images_human.append(env.render('human', True))
    images_box.append(env.render('box', True))

    steps_in_current_image = 1
    with tqdm(total=max_steps_per_image) as pbar:
        while steps_in_current_image <= max_steps_per_image:
            action = agent.act(obs)
            obs, r, done, _ = env.step(action)
            images_human.append(env.render('human', True))
            images_box.append(env.render('box', True))
            steps_in_current_image += 1
            pbar.update(1)
            if done:
                print('Agent pulled the trigger at step ' + str(steps_in_current_image))
                break

    print('Trying to save the resulting images …')

    if not os.path.exists('human'):
        os.makedirs('human')
    if not os.path.exists('box'):
        os.makedirs('box')

    for index, image in enumerate(images_human):
        image.save('human/' + str(index).zfill(3) + '.png', 'PNG')

    for index, image in enumerate(images_box):
        image.save('box/' + str(index).zfill(3) + '.png', 'PNG')

    print('Sucessfully saved the resulting images.')

    return



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

if __name__ == '__main__':
    generate_image_sequence(sys.argv[1:])
