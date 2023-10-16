# sansformers
Code for "Self-Supervised Forecasting in Electronic Health Records with Attention-Free Models"


## Training

> python run_mimic_experiment.py --cfg "path-to-config-yaml"


#### Note
- Training is fully functional but repo is still WIP, lot of code needs to be cleaned up!
- Work on this repo began before the whole huggingface ecosystem on sequence processing was mature, so you could possible speed up the vectorizer related code using transformers tokenizers.