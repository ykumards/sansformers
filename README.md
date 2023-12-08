# sansformers
Code for "Self-Supervised Forecasting in Electronic Health Records with Attention-Free Models"


## Training

> python run_mimic_experiment.py --cfg "path-to-config-yaml"


#### Note
- Training is fully functional but repo is still WIP, lot of code needs to be cleaned up!
- Work on this repo began before the whole huggingface ecosystem on sequence processing was mature, so you could possible speed up the vectorizer related code using transformers tokenizers.
---

#### Credits
The following repos greatly helped code up our model
- [g-mlp-pytorch](https://github.com/lucidrains/g-mlp-pytorch) - for the mixer components
- [minGPT](https://github.com/karpathy/minGPT) - for the Trainer
- [pycls](https://github.com/facebookresearch/pycls) - for our repo structure and for introducing us to YACS
