import coco_text
import numpy as np
import click


@click.command()
@click.option("--image_path", "-i", default="./images", help="Directory where the COCO-2014 images are located.", type=click.Path(exists=True))
@click.option("--annotations", "-a", default="cocotext.v2.json", help="The json file that contains the annotations.", type=click.Path(exists=True))
def prepare_coco(image_path, annotations):
    ct = coco_text.COCO_Text(annotations)

    imgIds = ct.getImgIds(imgIds=ct.train, catIds=[('legibility', 'legible')])

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
            [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
            box_tl_br = [(round(bbox_x), round(bbox_y)),
                         (round(bbox_x + bbox_w), round(bbox_y + bbox_h))]
            current_boxes.append(box_tl_br)

        bounding_boxes.append(current_boxes)

    np.save("bounding_boxes.npy", bounding_boxes)


if __name__ == '__main__':
    prepare_coco()
