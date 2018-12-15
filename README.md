# text-localization-agent

The code to train the agent.

## Prequisites

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
