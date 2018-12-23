import coco_text
import numpy as np
import click


@click.command()
@click.option("--image_path", "-i", default="./images/", help="Directoy where the COCO-2014 images are located.", type=click.Path(exists=True))
@click.option("--annotations", "-a", default="cocotext.v2.json", help="The json file that contains the annotations.", type=click.Path(exists=True))
def prepare_coco(image_path, annotations):
    ct = coco_text.COCO_Text(annotations)

    # image_path = 'C:/Users/Jona/.chainer/dataset/pfnet/chainercv/coco/images/train2014'

    imgIds = ct.getImgIds(imgIds=ct.train, catIds=[('legibility', 'legible'), ('class', 'machine printed')])

    imgs = ct.loadImgs(imgIds)

    img_ids = [imgs[i]['id'] for i in range(len(imgs))]
    img_paths = [image_path + "/" + imgs[i]['file_name'] for i in range(len(imgs))]

    np.savetxt("image_locations.txt", img_paths, fmt="%s")

    bounding_boxes = []

    for img_id in img_ids:
        current_boxes = []
        ann_ids = ct.getAnnIds(imgIds=img_id)
        annos = ct.loadAnns(ann_ids)

        for ann in annos:
            box_w_h = ann['bbox']
            box_tl_br = [(round(box_w_h[0]), round(box_w_h[1])),
                         (round(box_w_h[0] + box_w_h[2]), round(box_w_h[1] + box_w_h[3]))]
            current_boxes.append(box_tl_br)

        bounding_boxes.append(current_boxes)

    np.save("bounding_boxes.npy", bounding_boxes)


if __name__ == '__main__':
    prepare_coco()
