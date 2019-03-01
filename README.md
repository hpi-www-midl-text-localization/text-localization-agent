# text-localization-agent

The code to train the agent.

## Attention!

This project currently contains a memory leak, which means that during long training runs it might use up all your memory and make the server slow down or crash!

## Prerequisites

You need Python 3 (preferably 3.6) installed, as well as the requirements from `requirements.txt`:

```bash
$ pip install -r requirements.txt 
```

Furthermore, you need to install the [text-localization-environment](https://github.com/hpi-www-midl-text-localization/text-localization-environment) by following its **Installation** instructions.

## TensorBoard

If you would like the program to generate log-files appropriate for visualization in TensorBoard, you need to:

* Install **tensorflow**
  ```bash
  $ pip install tensorflow
  ```
  (If you use Python 3.7 and the installation fails, use: `pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.12.0-py3-none-any.whl
` instead. [See here, why.](https://github.com/tensorflow/tensorflow/issues/20444#issuecomment-442767411))
* Run the *text-localization-agent* program with the `--tensorboard` flag
   ```bash
   $ python train-agent.py --tensorboard --imagefile … --boxfile …
   ``` 
* Start TensorBoard pointing to the `tensorboard/` directory inside the *text-localization-agent* project
   ```bash
   $ tensorboard --logdir=<path to text-localization-agent>/tensorboard/
   …
   TensorBoard 1.12.0 at <link to TensorBoard UI> (Press CTRL+C to quit)
   ``` 
* Open the TensorBoard UI via the link that is provided when the `tensorboard` program is started (usually: http://localhost:6006)

## Training on the chair's servers

To run the training on one of the chair's servers you need to:

* Clone the necessary repositories
* Create a new virtual environment. Note that the Python version needs to be at least 3.6 for everything to run. 
The default might be a lower version so if that is the case you must make sure that the correct version is used.
You can pass the correct python version to virtualenv via the `-p` parameter, for example
    ```bash
    $ virtualenv -p python3.6 <envname>
    ```
    (If there is no Python 3.6/3.7 installed you are out of luck because we don't have sudo access)
* Activate the environment via
    ```bash
    $ source <envname>/bin/activate
    ```
* Install the required packages (see section "Prerequisites"). Don't forget **cupy**, **tb_chainer** and **tensorflow**!
* Prepare the training data (either generate it using the [dataset-generator](https://github.com/hpi-www-midl-text-localization/dataset-generator)
or transfer existing data on the server)
* To avoid stopping the training after disconnecting from the server, you might want to use a terminal-multiplexer 
such as [tmux](https://wiki.ubuntuusers.de/tmux/) or [screen](https://wiki.ubuntuusers.de/Screen/)
* Set the CUDA_PATH and LD_LIBRARY_PATH variables if they are not already set. The command should be something like
    ```bash
    $ export CUDA_PATH=/usr/local/cuda
    $ export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
    ```
* Start training!

These instructions are for starting from scratch, for example if there is already a suitable virtual environment you 
obviously don't need to create a new one.

## Evaluating

* To evaluate a previously trained agent on a dataset, you may use the `evaluate` method available as a click CLI when executing:
    ```bash
    $ python evaluate_agent.py
    ```
    (Run `python evaluate_agent.py --help` to see the required parameters for the CLI)
* If you provide the `--save` flag in the CLI above, it creates `.npy` files which can be read by the `evaluate_from_files` CLI afterwards:
    ```bash
    $ python evaluate_from_files.py
    ```
    (Run `python evaluate_from_files.py --help` to see the required parameters for the CLI)
* The `evaluate_from_files` CLI allows defining an IoU threshold used for the calculation of the evaluation metrics. Furthermore, it does not only output the mean average precision (mAP) but also the precision and recall values.

## Creating image sequences/animations for visualization purposes

* To create an image sequence of a an already trained agent acting on a specific image, use:
    ```bash
    $ python generate_image_sequence.py
    ```
    (Run `python generate_image_sequence.py --help` to see the required parameters for the CLI and have a look into the `generate_image_sequence.py` file for instructions on creating a video out of the generated single frames using ffmpeg) 