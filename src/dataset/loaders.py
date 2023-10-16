"""Data loader."""

import os
from pathlib import Path

import _pickle as pickle
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler

from core.config import cfg
from dataset.mimic_dataset import MIMICDataset
from dataset.vectorizer import EHRCountVectorizer
from utils import common


def _truncate_mimic_df(df, max_seq_length):
    "Truncate the sequences to max_seq_length"
    seq_columns = [
        "icd_all",
        "proc_all",
        "drug_all",
        "service_all",
        "admission_type",
        "insurance",
        "marital_status",
        "days_from_prev",
        "los",
    ]

    def _truncator(row):
        visits = row.split(";")
        trunc_visits = visits[-(max_seq_length - 1) :]
        return ";".join(trunc_visits)

    for col in seq_columns:
        print(col)
        df.loc[:, col] = df.loc[:, col].apply(lambda row: _truncator(row))

    return df


def apply_topcap(row, topcap):
    capped_outcome = []
    for i in range(len(row)):
        row_count = row[i]
        row_count = min(row_count, topcap)
        capped_outcome.append(row_count)
    capped_outcome = (
        [capped_outcome[0]]
        + [capped_outcome[1] + capped_outcome[2]]
        + capped_outcome[3:]
    )
    return np.array(capped_outcome)


def load_mimic_dataframe(
    datapath, filename, split, max_seq_length, topcap, is_sample=False, sample_pct=None
):
    assert split in ["train", "test"], "split must be either 'train' or 'test'"

    file_path = os.path.join(datapath, filename)
    df = pd.read_feather(file_path)

    # drop patients with < 3 visits
    df = df.query("seq_length >= 3")
    df = df.query(f"seq_length <= 1000")

    # if split == "test": max_seq_length = 3
    print(f"Using seqlength of {max_seq_length}")
    df = _truncate_mimic_df(df, max_seq_length)

    if split == "train":
        if is_sample:
            sample_frac = 1.0 if sample_pct is None else sample_pct
            print(f"Using sample proportion of {sample_frac} percent")
            df = df.sample(frac=sample_frac, random_state=100)
        print("Training: Stats")
        print(df.seq_length.describe())

    df.reset_index(inplace=True, drop=True)
    return df


def load_mimic_vectorizers(cfg, df_train):
    infile = open(
        cfg.PATHS.VECTORIZER_PATH, "rb"
    )
    vec_dict = pickle.load(infile)
    infile.close()
    return vec_dict


def get_mimic_dataloaders(cfg):
    common.seed_everything(cfg.RNG_SEED)

    df_train = load_mimic_dataframe(
        datapath=cfg.PATHS.DATAPATH,
        filename=cfg.TRAIN.FILENAME,
        split="train",
        max_seq_length=cfg.MODEL.MAX_SEQ_LENGTH,
        topcap=cfg.MODEL.TOP_CAP,
        is_sample=cfg.OVERFIT_ON_BATCH,
        sample_pct=cfg.OVERFIT_ON_BATCH_PCT,
    )

    df_test = load_mimic_dataframe(
        datapath=cfg.PATHS.DATAPATH,
        filename=cfg.TEST.FILENAME,
        split="test",
        max_seq_length=cfg.MODEL.MAX_SEQ_LENGTH,
        topcap=cfg.MODEL.TOP_CAP,
    )
    # optional second test set, by default copy of the first
    df_test2 = load_mimic_dataframe(
        datapath=cfg.PATHS.DATAPATH,
        filename=cfg.TEST.FILENAME2,
        split="test",
        max_seq_length=cfg.MODEL.MAX_SEQ_LENGTH,
        topcap=cfg.MODEL.TOP_CAP,
    )

    vec_dict = load_mimic_vectorizers(cfg, df_train)

    train_dataset = MIMICDataset(
        df_train,
        vec_dict,
    )

    test_dataset = MIMICDataset(
        df_test,
        vec_dict,
    )

    test_dataset2 = MIMICDataset(
        df_test2,
        vec_dict,
    )
    collate_fn = MIMICDataset.collate_fn

    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(cfg.TRAIN.VALIDATION_SPLIT * dataset_size))
    np.random.shuffle(indices)
    train_indices, valid_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        sampler=train_sampler,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=False,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        sampler=valid_sampler,
        batch_size=cfg.TEST.BATCH_SIZE,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=False,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=collate_fn,
        batch_size=cfg.TEST.BATCH_SIZE,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=False,
        pin_memory=True,
    )
    test_dataloader2 = DataLoader(
        test_dataset2,
        collate_fn=collate_fn,
        batch_size=cfg.TEST.BATCH_SIZE,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=False,
        pin_memory=True,
    )

    test_dataloaders = [test_dataloader, test_dataloader2]

    return train_dataloader, val_dataloader, test_dataloaders
