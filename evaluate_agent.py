import sys
import numpy as np
import click
from tqdm import tqdm
from load_agent import load_agent, create_environment
from chainercv.evaluations.eval_detection_voc import eval_detection_voc


@click.command()
@click.option("--gpu", default=-1, help="ID of the GPU to be used. -1 if the CPU should be used instead.")
@click.option("--imagefile", "-i", default='image_locations.txt', help="Path to the file containing the image locations.", type=click.Path(exists=True))
@click.option("--boxfile", "-b", default='bounding_boxes.npy', help="Path to the bounding boxes.", type=click.Path(exists=True))
@click.option("--agentdirectory", "-a", default='./agent', help="Path to the directory containing the agent that should be evaluated.", type=click.Path(exists=True))
@click.option("--save/--dont-save", default=False, help="Boolean indicating whether the intermediate results of the evaluation should be saved.")
def evaluate(gpu, imagefile, boxfile, agentdirectory, save):
    max_steps_per_image = 200
    max_steps_per_run = 50
    max_trigger_events_per_image = 1

    images = np.loadtxt(imagefile, dtype=str).tolist()
    gt_bboxes = np.load(boxfile)

    env = create_environment(imagefile, boxfile, gpu)
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
        labels = []
        for _ in range(len(bbox)):
            labels.append(True)
        gt_labels.append(labels)

    pred_bboxes = list(map(convert_env_bboxes_to_chainercv_format, pred_bboxes))
    pred_labels = []
    pred_scores = []
    for bbox in pred_bboxes:
        labels = []
        scores = []
        for _ in range(len(bbox)):
            labels.append(True)
            scores.append(1.0)
        pred_labels.append(labels)
        scores = np.array(scores, dtype=np.float32)
        pred_scores.append(scores)

    if save:
        np.save('gt_bboxes.npy', gt_bboxes)
        np.save('gt_labels.npy', gt_labels)
        np.save('pred_bboxes.npy', pred_bboxes)
        np.save('pred_labels.npy', pred_labels)
        np.save('pred_scores.npy', pred_scores)

    eval = eval_detection_voc(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels)

    print('Mean average precision (map): %f' % eval['map'])
    print('You may use the evaluate_from_files.py script to calculate the map, precision and recall for differnent IoU thresholds.')
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
    evaluate(sys.argv[1:])
