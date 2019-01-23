import click
import numpy as np
from chainercv.evaluations.eval_detection_voc import eval_detection_voc

@click.command()
@click.option("--pred_bboxes", default='pred_bboxes.npy', type=click.Path(exists=True))
@click.option("--pred_labels", default='pred_labels.npy', type=click.Path(exists=True))
@click.option("--pred_scores", default='pred_scores.npy', type=click.Path(exists=True))
@click.option("--gt_bboxes", default='gt_bboxes.npy', type=click.Path(exists=True))
@click.option("--gt_labels", default='gt_labels.npy', type=click.Path(exists=True))
def main(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels):
    pred_bboxes = np.load(pred_bboxes)
    pred_labels = np.load(pred_labels)
    pred_scores = np.load(pred_scores)
    gt_bboxes = np.load(gt_bboxes)
    gt_labels = np.load(gt_labels)
    eval = eval_detection_voc(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels)
    print(eval)
    return

if __name__ == '__main__':
    main()
