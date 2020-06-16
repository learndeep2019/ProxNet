# Proximal Mapping for Deep Regularization

This repository provides the code for the paper 
<br>
[Proximal Mapping for Deep Regularization](http://)
<br>

<!-- ### Citation -->

## Dependencies
* [Python 3.6+ ](https://www.python.org)
* [PyTorch 1.2+](http://pytorch.org)
* [TorchVision](https://www.python.org)
* [scipy](https://www.scipy.org)
* [matplotlib](https://matplotlib.org/#)
* [TQDM](https://github.com/tqdm/tqdm)
* [YAML](https://pyyaml.org)
* [sklearn](http://scikit-learn.github.io/stable)

## Getting Started
---

### Installation & Data preparation

1. Install [PyTorch](http://pytorch.org) and other dependencies in the above list (e.g., torchvision, tqdm, Numpy).

2. Download the sketchy dataset via this [link](https://drive.google.com/file/d/1xugjAF0TeyHMjxhEtg2hb0tOURZWN9VF/view?usp=sharing). Download XRMB dataset via this [link](http://ttic.uchicago.edu/~qmtang/Data/Interspeech2017/XRMB_SEQ.mat). Then unzip and move them to `data` folder.

### Sketch-Photo Image Classification Task
+ To train MetaProx on Sketchy, you need frist navigate to `image_classify` folder then run the script with default arguments:
```bash
cd image_classify
python main.py
```
+ In this experiment, we mainly varied the number of classes over \{20, 50, 100, 125\}. You can test them by running the script with specified arguments:
```bash
python main.py -c CONFIG
```
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
replace CONFIG with \{'PROX_20', 'PROX_50', 'PROX_100', 'PROX_125'\}.

+ We mainly use the [YAML](https://pyyaml.org) syntax for configuration. This allow us to easily test the code with varying setting by editting configuration file `config.yaml` directly. 

> For instance, the model will save the best checkpoint on the fly. You can resume the training process from the checkpoint by assigning `True` value to argument `resume` in `config.yaml`.

+ For completeness we document some options which have a marked impact on the results:
    * `lr`: The learning rate of Adam optimizer. Default 0.001.
    * `lr_decay`: decay the learning rate by 0.1 every step_size epoch. This argument define step_size value. Default 200.
    * `proj_k`: project the latent representation into a K-dimensional subspace. Default 20. A large K may lead to the training time increases.
    * `batch_size`: the mini-batch size. Default 100.
    * `alpha`: the trade-off parameter between correlation and displacement. You can either fix this argument or adjust it as traning proceed. Default 10.
    * `hidden_size`: the number of units in each hidden layer. Default 512.

### Audio-visual speech recognition
+ To train MetaProx for speech recognition on XRMB, you need frist navigate to `speech_recog` folder then run the script with default arguments:
```bash
cd speech_recog
python main.py
```

+ To simulate the real-life scenarios and to improve the model’s robustness to noise, the acoustic features of a given speaker are corrupted by mixing with \{0.2, 0.5, 0.8\} level of another randomly picked speaker's acoustic features. You can see the difference, in the experiment, by running with a specified argument:
```bash
python main.py -n VALUE
``` 
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 
replace *VALUE* with \{0, 0.2, 0.5, 0.8\}.

+ For completeness we document some options which have a marked impact on the results:
    * `lr`: learning rate of Adam optimizer. Default 0.001.
    * `lr_decay`: decay the learning rate by 0.1 every step_size epoch. This argument define step_size value. Default 200.
    * `proj_k`: project the latent representation into a K-dimensional subspace. Default 20. A large K may lead to the training time increases.
    * `batch_size`: mini-batch size. Default 32.
    * `alpha`: trade-off parameter between correlation and displacement. You can either fix this argument or adjust it as traning proceed. Default 100.
    * `hidden_size`: number of units in each hidden layer. Default 256.
    * `num_layers`: number of recurrent layers of a stacked LSTM. Default 2.
    * `bidirectional`: if True, becomes a bidirectional LSTM. Default False.
    * `avg_logit`: if True, averaging the logits of two views before compute the loss. Default True.
    * `seq_len`: lenght of input sequence. Default 1000. 



