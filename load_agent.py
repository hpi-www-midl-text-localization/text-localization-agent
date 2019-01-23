from text_localization_environment import TextLocEnv

import os
import numpy as np
import click
from tqdm import tqdm
import chainer
import chainerrl
from chainercv.evaluations.eval_detection_voc import eval_detection_voc


def create_environment(imagefile='image_locations.txt', boxfile='bounding_boxes.npy', gpu=0):
    locations = np.loadtxt(imagefile, dtype=str).tolist()
    bboxes = np.load(boxfile)

    env = TextLocEnv(locations, bboxes, gpu)

    return env


def load_agent(env, directory="agent", gpu=0):
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

    agent.load(directory)

    return agent


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

@click.command()
@click.option("--gpu", default=-1, help="ID of the GPU to be used. -1 if the CPU should be used instead.")
@click.option("--imagefile", "-i", default='image_locations.txt', help="Path to the file containing the image locations.", type=click.Path(exists=True))
@click.option("--boxfile", "-b", default='bounding_boxes.npy', help="Path to the bounding boxes.", type=click.Path(exists=True))
@click.option("--agentdirectory", "-a", default='./agent', help="Path to the directory containing the agent that should be evaluated.", type=click.Path(exists=True))
@click.option("--save/--dont-save", default=False, help="Boolean indicating whether the intermediate results of the evaluation should be saved.")
def evaluate(gpu, imagefile, boxfile, agentdirectory, save):
    max_steps_per_image = 200
    max_steps_per_run = 40
    max_trigger_events_per_image = 1

    relative_paths = np.loadtxt(imagefile, dtype=str)
    images_base_path = os.path.dirname(imagefile)
    absolute_paths = [images_base_path + i.strip('.') for i in relative_paths]

    images = np.loadtxt(imagefile, dtype=str).tolist()

    gt_bboxes = np.load(boxfile)

    env = TextLocEnv(absolute_paths, gt_bboxes, gpu)
    agent = load_agent(env, agentdirectory, gpu)

    pred_bboxes = []

    pbar = tqdm(images, unit="images")
    for image_index, image in enumerate(pbar):
        predicted_bboxes_for_image = []
        _ = env.reset(image_index)
        steps_in_current_image = 1
        trigger_events_in_current_image = 0
        while steps_in_current_image <= max_steps_per_image \
                and trigger_events_in_current_image < max_trigger_events_per_image:
            obs = env.reset(image_index, True)
            steps_in_current_run = 1
            while steps_in_current_run <= max_steps_per_run:
                pbar.set_postfix(image=image[-11:], step=steps_in_current_image)
                action = agent.act(obs)
                obs, r, done, _ = env.step(action)
                if done:
                    predicted_bboxes_for_image.append(env.bbox)
                    trigger_events_in_current_image += 1
                    break
                steps_in_current_run += 1
                steps_in_current_image += 1
        pred_bboxes.append(predicted_bboxes_for_image)

    gt_bboxes = list(map(convert_gt_bboxes_to_chainercv_format, gt_bboxes))
    gt_labels = []
    for bbox in gt_bboxes:
        labels = np.ones((len(bbox), 1), dtype=np.bool_)
        gt_labels.append(labels)

    pred_bboxes = list(map(convert_env_bboxes_to_chainercv_format, pred_bboxes))
    pred_labels = []
    pred_scores = []
    for bbox in pred_bboxes:
        labels = np.ones((len(bbox), 1), dtype=np.bool_)
        scores = np.ones((len(bbox), 1), dtype=np.float32)
        pred_labels.append(labels)
        pred_scores.append(scores)

    np.save('gt_bboxes.npy', gt_bboxes)
    np.save('gt_labels.npy', gt_labels)
    np.save('pred_bboxes.npy', pred_bboxes)
    np.save('pred_labels.npy', pred_labels)
    np.save('pred_scores.npy', pred_scores)

    eval = eval_detection_voc(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels)

    print(eval)
    return

def convert_gt_bboxes_to_chainercv_format(bboxes):
    """
    Converts a list of bounding boxes to ChainerCV's internal format
    :param bboxes: a list of bboxes in the format ((x_min, y_min), (x_max, y_max))
    :return: a numpy.array of bboxes in the format [[y_min, x_min, y_max, x_max]]
    """
    bboxes_chainercv = list()
    for bbox in bboxes:
        x_min = bbox[0][0]
        y_min = bbox[0][1]
        x_max = bbox[1][0]
        y_max = bbox[1][1]
        bboxes_chainercv.append([y_min, x_min, y_max, x_max])
    return np.array(bboxes_chainercv, dtype=np.float32)

def convert_env_bboxes_to_chainercv_format(bboxes):
    """
    Converts a list of bounding boxes to ChainerCV's internal format
    :param bboxes: a list of bboxes in the format [[x_min, y_min, x_max, y_max]]
    :return: a numpy.array of bboxes in the format [[y_min, x_min, y_max, x_max]]
    """
    bboxes_chainercv = list()
    for bbox in bboxes:
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]
        bboxes_chainercv.append([y_min, x_min, y_max, x_max])
    return np.array(bboxes_chainercv, dtype=np.float32)

if __name__ == '__main__':
    evaluate()
