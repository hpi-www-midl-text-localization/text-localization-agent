import sys
import click
import numpy as np
from chainercv.evaluations.eval_detection_voc import eval_detection_voc, calc_detection_voc_prec_rec


@click.command()
@click.option("--pred_bboxes", default='pred_bboxes.npy', type=click.Path(exists=True))
@click.option("--pred_labels", default='pred_labels.npy', type=click.Path(exists=True))
@click.option("--pred_scores", default='pred_scores.npy', type=click.Path(exists=True))
@click.option("--gt_bboxes", default='gt_bboxes.npy', type=click.Path(exists=True))
@click.option("--gt_labels", default='gt_labels.npy', type=click.Path(exists=True))
@click.option("--iou_threshold", default=0.5)
def evaluate_from_files(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, iou_threshold):
    pred_bboxes = np.load(pred_bboxes)
    pred_labels = np.load(pred_labels)
    pred_scores = np.load(pred_scores)
    gt_bboxes = np.load(gt_bboxes)
    gt_labels = np.load(gt_labels)

    map = eval_detection_voc(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, iou_thresh=iou_threshold)['map']
    prec, rec = calc_detection_voc_prec_rec(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
                                            iou_thresh=iou_threshold)

    print('IoU threshold used: %f' % iou_threshold)
    print('Mean average precision (map): %f' % map)
    print('Mean precision: %f' % np.mean(prec[1]))
    print('Mean recall: %f' % np.mean(rec[1]))
    return


if __name__ == '__main__':
    evaluate_from_files(sys.argv[1:])
