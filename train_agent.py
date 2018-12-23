import click
import os
from PIL import Image, ImageDraw, ImageFont
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
@click.option("--gpu", default=-1, help="ID of the GPU to be used. -1 if the CPU should be used instead.")
@click.option("--imagefile", "-i", default='image_locations.txt', help="Path to the file containing the image locations.", type=click.Path(exists=True))
@click.option("--relative/--absolute", default=True, help="Whether the imagefile uses relative or absolute paths.")
@click.option("--boxfile", "-b", default='bounding_boxes.npy', help="Path to the bounding boxes.", type=click.Path(exists=True))
@click.option("--tensorboard/--no-tensorboard", default=False)
def main(steps, gpu, imagefile, relative, boxfile, tensorboard):
    print(steps)
    print(gpu)
    print(imagefile)
    print(boxfile)
    print(relative)

    if relative:
        relative_paths = np.loadtxt(imagefile, dtype=str)
        images_base_path = os.path.dirname(imagefile)
        absolute_paths = [images_base_path + i.strip('.') for i in relative_paths]
    else:
        absolute_paths = np.loadtxt(imagefile, dtype=str).tolist()
    bboxes = np.load(boxfile)

    env = TextLocEnv(absolute_paths, bboxes, gpu)

    obs_size = 4186
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
        epsilon=0.3, random_action_func=env.action_space.sample)

    # DQN uses Experience Replay.
    # Specify a replay buffer and its capacity.
    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)

    # Now create an agent that will interact with the environment.
    agent = chainerrl.agents.DQN(
        q_func, optimizer, replay_buffer, gamma, explorer,
        gpu=gpu,
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
        step_count = agent.t

        self.summary_writer.add_scalar('average_q', agent.average_q, step_count)

        self.summary_writer.add_scalar('average_loss', agent.average_loss, step_count)

        last_action_name = env.action_set[agent.last_action].__name__
        debug_image = self.get_debug_image_from_environment_image(image = env.episode_image.copy(),
                                                                  bbox = env.bbox,
                                                                  last_action_name = last_action_name)

        # Prepare the image for further use in rb_chainer's add_image() method
        debug_image = np.array(debug_image, np.float32) / 255
        debug_image = debug_image.transpose((2, 0, 1))

        self.summary_writer.add_image('episode_image_debug', debug_image, step_count)
        return

    def get_debug_image_from_environment_image(self, image, bbox, last_action_name):
        image = image.copy()

        debug_image = Image.new(mode='RGB',
                                size=(image.width, image.height + 16),
                                color='black')
        debug_image.paste(image)

        draw = ImageDraw.Draw(debug_image)
        draw.rectangle(bbox.tolist(), outline='white')
        draw.rectangle((0, image.height, image.width, image.height + 16), fill='black')
        font = ImageFont.truetype('fonts/open-sans/OpenSans-SemiBold.ttf', 12)
        draw.text(xy=(2, image.height),
                  text='Last action: %s' % last_action_name,
                  fill='white',
                  font=font)

        return debug_image

if __name__ == '__main__':
    main()
