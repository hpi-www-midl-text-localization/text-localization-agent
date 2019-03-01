import click
import os
import numpy as np
from text_localization_environment import TextLocEnv
from chainerrl.experiments.train_agent import train_agent_with_evaluation
import chainer
import chainerrl
import logging
import sys
from tb_chainer import SummaryWriter
import time
import re


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
    explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
        start_epsilon=1.0,
        end_epsilon=0.1,
        decay_steps=300000,
        random_action_func=env.action_space.sample)

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

    eval_run_count = 10

    timestr = time.strftime("%Y%m%d-%H%M%S")
    agentClassName = agent.__class__.__name__[:10]

    step_hooks = []
    logger = None
    if tensorboard:
        writer = SummaryWriter("tensorboard/tensorBoard_exp_" + timestr + "_" + agentClassName)
        step_hooks = [TensorBoardLoggingStepHook(writer)]
        handler = TensorBoardEvaluationLoggingHandler(writer, agent, eval_run_count)
        logger = logging.getLogger()
        logger.addHandler(handler)

    # Overwrite the normal evaluation method
    chainerrl.experiments.evaluator.run_evaluation_episodes = run_localization_evaluation_episodes

    train_agent_with_evaluation(
        agent, env,
        steps=steps,  # Train the agent for 5000 steps
        eval_n_runs=eval_run_count,  # 10 episodes are sampled for each evaluation
        max_episode_len=50,  # Maximum length of each episodes
        eval_interval=500,  # Evaluate the agent after every 100 steps
        outdir='result',  # Save everything to 'result' directory
        step_hooks=step_hooks,
        logger=logger)

    agent.save('agent_' + timestr + "_" + agentClassName)


class TensorBoardLoggingStepHook(chainerrl.experiments.StepHook):
    def __init__(self, summary_writer):
        self.summary_writer = summary_writer
        return

    def __call__(self, env, agent, step):
        step_count = agent.t

        self.summary_writer.add_scalar('average_q', agent.average_q, step_count)

        self.summary_writer.add_scalar('average_loss', agent.average_loss, step_count)

        return


class TensorBoardEvaluationLoggingHandler(logging.Handler):
    def __init__(self, summary_writer, agent, eval_run_count, level=logging.NOTSET):
        logging.Handler.__init__(self, level)
        self.summary_writer = summary_writer
        self.agent = agent
        self.eval_run_count = eval_run_count
        self.episode_rewards = np.empty(eval_run_count)
        self.episode_lengths = np.empty(eval_run_count)
        self.episode_ious = np.empty(eval_run_count)
        self.episode_max_ious = np.empty(eval_run_count)
        return

    def emit(self, record):
        match_new_best = re.search(r'The best score is updated ([^ ]*) -> ([^ ]*)', record.getMessage())
        if match_new_best:
            new_best_score = match_new_best.group(2)
            step_count = self.agent.t
            self.summary_writer.add_scalar('evaluation_new_best_score', new_best_score, step_count)

        match_reward = re.search(r'evaluation episode ([^ ]*) length:([^ ]*) R:([^ ]*) IoU:([^ ]*) Max_IoU:([^ ]*)', record.getMessage())
        if match_reward:
            episode_number = int(match_reward.group(1))
            episode_length = int(match_reward.group(2))
            episode_reward = float(match_reward.group(3))
            episode_iou = float(match_reward.group(4))
            episode_max_iou = float(match_reward.group(5))

            self.episode_lengths[episode_number] = episode_length
            self.episode_rewards[episode_number] = episode_reward
            self.episode_ious[episode_number] = episode_iou
            self.episode_max_ious[episode_number] = episode_max_iou

            if episode_number == self.eval_run_count - 1:
                step_count = self.agent.t
                self.summary_writer.add_scalar('evaluation_length_mean', np.mean(self.episode_lengths), step_count)
                self.summary_writer.add_scalar('evaluation_reward_mean', np.mean(self.episode_rewards), step_count)
                self.summary_writer.add_scalar('evaluation_reward_median', np.median(self.episode_rewards), step_count)
                self.summary_writer.add_scalar('evaluation_reward_variance', np.var(self.episode_rewards), step_count)
                self.summary_writer.add_scalar('evaluation_iou_mean', np.mean(self.episode_ious), step_count)
                self.summary_writer.add_scalar('evaluation_iou_median', np.median(self.episode_ious), step_count)
                self.summary_writer.add_scalar('evaluation_max_iou_mean', np.mean(self.episode_max_ious), step_count)
        return


def run_localization_evaluation_episodes(env, agent, n_runs, max_episode_len=None,
                            logger=None):
    """Run multiple evaluation episodes and return returns.
    Args:
        env (Environment): Environment used for evaluation
        agent (Agent): Agent to evaluate.
        n_runs (int): Number of evaluation runs.
        max_episode_len (int or None): If specified, episodes longer than this
            value will be truncated.
        logger (Logger or None): If specified, the given Logger object will be
            used for logging results. If not specified, the default logger of
            this module will be used.
    Returns:
        List of returns of evaluation runs.
    """
    logger = logger or logging.getLogger(__name__)
    scores = []
    for i in range(n_runs):
        obs = env.reset()
        done = False
        test_r = 0
        t = 0
        while not (done or t == max_episode_len):
            a = agent.act(obs)
            obs, r, done, info = env.step(a)
            test_r += r
            t += 1
        agent.stop_episode()
        # As mixing float and numpy float causes errors in statistics
        # functions, here every score is cast to float.
        iou = float(env.iou) if done else float(0)
        max_iou = float(env.max_iou)
        scores.append(float(test_r))
        logger.info('evaluation episode %s length:%s R:%s IoU:%s Max_IoU:%s', i, t, test_r, iou, max_iou)
    return scores


if __name__ == '__main__':
    main()
