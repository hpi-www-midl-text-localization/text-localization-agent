import click


@click.command()
@click.option("--steps", "-s", default=2000, help="Amount of steps to train the agent.")
@click.option("--gpu/--no-gpu", default=False)
@click.option("--imagefile", "-i", default='image_locations.txt', help="Path to the file containing the image locations.", type=click.Path(exists=True))
@click.option("--boxfile", "-b", default='bounding_boxes.npy', help="Path to the bounding boxes.", type=click.Path(exists=True))
def main(steps, gpu, imagefile, boxfile):
    print(steps)
    print(gpu)
    print(imagefile)
    print(boxfile)


if __name__ == '__main__':
    main()
