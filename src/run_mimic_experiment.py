import sys

sys.path.append("../")

import os
import time

import dataset.loaders as loaders
import utils.common as common
from models.mimic_additive_sansformer import MimicAdditiveSansformerModel
from models.mimic_axial_sansformer import MimicAxialSansformerModel
from trainers.mimic_trainer import Trainer_MIMIC

MODEL_TYPE = {
    "add_SANSformer": MimicAdditiveSansformerModel,
    "axial_SANSformer": MimicAxialSansformerModel,
}


def run_experiment(cfg, train_dataloader, val_dataloader, test_dataloaders):

    assert (
        cfg.MODEL.TYPE in MODEL_TYPE.keys()
    ), f"Model type name should be among {MODEL_TYPE.keys()}"

    # define model
    model = MODEL_TYPE[cfg.MODEL.TYPE](cfg)
    # initialize the trainer
    trainer = Trainer_MIMIC(
        cfg, model, train_dataloader, val_dataloader, test_dataloaders
    )
    # fit on training data
    trainer.fit()
    # predict on valid and test data
    test_metrics_l = trainer.predict()

    common.log_test_results_to_csv(
        cfg,
        os.path.join(
            "/l/Work/aalto/sansformers/experiments/experiment_logs/",
            cfg.PATHS.RESULTS_LOG_FILENAME,
        ),
        test_metrics_l,
    )


def main():
    starttime = time.time()
    # Parse cmd line args
    args = common.parse_args()
    cfg = common.handle_config_and_log_paths(args)
    # set experiment seed
    common.seed_everything(cfg.RNG_SEED)

    (
        train_dataloader,
        val_dataloader,
        test_dataloaders,
    ) = loaders.get_mimic_dataloaders(cfg)

    cfg.defrost()

    cfg.OPTIM.STEPS_PER_EPOCH = len(train_dataloader) // cfg.MODEL.ACCU_GRAD_STEPS
    print(f"Number of steps per epoch: {len(train_dataloader)}")

    cfg.MODEL.VOCAB_SIZE = train_dataloader.dataset.vectorizer.seq_vocab_len
    print(f"Total vocab size: {train_dataloader.dataset.vectorizer.seq_vocab_len}")

    cfg.freeze()

    print("=" * 100)
    print(f"Running Experiment: {cfg.PATHS.EXPERIMENT_NAME}")
    print("=" * 100)

    # run experiment
    run_experiment(cfg, train_dataloader, val_dataloader, test_dataloaders)

    print(f"Done in {(time.time() - starttime)/60} minutes.")


if __name__ == "__main__":
    main()
