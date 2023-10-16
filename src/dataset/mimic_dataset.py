"""MIMIC dataset."""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from core.config import cfg
from dataset.vectorizer import EHRCountVectorizer


class MIMICDataset(torch.utils.data.Dataset):
    def __init__(self, df, vec_dict):
        self.df = df
        self.vectorizer = vec_dict["mimic_all"]

        self.y_outcome = np.array(df.loc[:, "hospital_expire_flag"])
        self.y_los = np.array(df.loc[:, "los"])
        self.y_next_v = np.array(df.loc[:, "days_from_prev"])

        self.icd = df.loc[:, "icd_all"]
        self.proc = df.loc[:, "proc_all"]
        self.drg = df.loc[:, "drug_all"]
        self.service = df.loc[:, "service_all"]
        self.admtype = df.loc[:, "admission_type"]
        self.insur = df.loc[:, "insurance"]
        self.marit = df.loc[:, "marital_status"]

        self.delta_t = df.loc[:, "days_from_prev"]
        self.language = df.loc[:, "encoded_language"]
        self.ethnicity = df.loc[:, "encoded_ethnicity"]
        self.language = df.loc[:, "encoded_language"]
        self.gender = df.loc[:, "encoded_gender"]
        self.age = df.loc[:, "anchor_age"]
        self.seq_length = df.loc[:, "seq_length"]

        self.length = self.df.shape[0]

    def __getitem__(self, idx):
        diag_seq, diag_n_tkns_per_visit = self.vectorizer.vectorize(self.icd[idx])
        proc_seq, proc_n_tkns_per_visit = self.vectorizer.vectorize(self.proc[idx])
        drg_seq, drg_n_tkns_per_visit = self.vectorizer.vectorize(self.drg[idx])
        service, service_n_tkns_per_visit = self.vectorizer.vectorize(self.service[idx])
        admtype, admtype_n_tkns_per_visit = self.vectorizer.vectorize(self.admtype[idx])

        insur, _ = self.vectorizer.vectorize(self.insur[idx])
        marit, _ = self.vectorizer.vectorize(self.marit[idx])

        delta_t = [float(x) for x in self.delta_t[idx].split(";")]
        y_los = [float(x) for x in self.y_los[idx].split(";")][1:]
        x_los = [float(x) for x in self.y_los[idx].split(";")][:-1]
        y_next_v = [float(x) for x in self.y_next_v[idx].split(";")][1:]
        x_next_v = [float(x) for x in self.y_next_v[idx].split(";")][:-1]

        return {
            "diag_seq": diag_seq,
            "proc_seq": proc_seq,
            "drg_seq": drg_seq,
            "seq_length": len(diag_n_tkns_per_visit),
            "diag_max_visit_items": min(50, max(diag_n_tkns_per_visit)),
            "proc_max_visit_items": min(40, max(proc_n_tkns_per_visit)),
            "drug_max_visit_items": min(2, max(drg_n_tkns_per_visit)),
            "service_max_visit_items": min(10, max(service_n_tkns_per_visit)),
            "admtype_max_visit_items": min(4, max(admtype_n_tkns_per_visit)),
            "delta_t": np.array(delta_t),
            "age": float(self.age[idx]),
            "gender": int(self.gender[idx]),
            "language": int(self.language[idx]),
            "ethnicity": int(self.ethnicity[idx]),
            "service": service,
            "admtype": admtype,
            "insurance": np.array(insur).ravel(),
            "marit": np.array(marit).ravel(),
            "y_outcome": int(self.y_outcome[idx]),
            "y_los": np.array(y_los),
            "y_next_v": np.array(y_next_v),
            "x_los": np.array(x_los),
            "x_next_v": np.array(x_next_v),
        }

    def __len__(self):
        return self.length

    @staticmethod
    def pad_sequence(seq, max_num_visits, max_max_visit_items):
        time_pad = max_num_visits - len(seq)
        if time_pad < 0:
            breakpoint()
        seq = [np.array(xi) for xi in seq]
        padded_seq = []
        for i, visit in enumerate(seq):
            # some visits are longer than hardcoded max visit lens,
            # we just ignore the other tokens
            if len(visit) < max_max_visit_items:
                visit_idx = np.arange(len(visit)).astype(np.int16)
            else:
                visit_idx = np.random.choice(
                    np.arange(len(visit)).astype(np.int16),
                    size=max_max_visit_items,
                    replace=False,
                )
            visit_tr = visit[visit_idx]
            intra_pad = max_max_visit_items - visit_tr.shape[0]
            padded_seq.append(np.pad(visit_tr, pad_width=(0, intra_pad)))

        # now pad along time axis
        padded_seq = np.pad(np.stack(padded_seq), pad_width=((0, time_pad), (0, 0)))
        return torch.tensor(padded_seq, dtype=torch.long)

    @staticmethod
    def collate_fn(batch, is_flat_seq=False):
        keys = list(batch[0].keys())
        processed_batch = {k: [] for k in keys}

        # all 1-D sequences that need padding
        single_pad_visit_cols = [
            "insurance",
            "marit",
            "delta_t",
            "y_los",
            "y_next_v",
            "x_los",
            "x_next_v",
        ]
        seq_keys = single_pad_visit_cols + [
            "diag_seq",
            "proc_seq",
            "drg_seq",
            "service",
            "admtype",
        ]
        float_cols = ["age", "delta_t", "y_los", "y_next_v", "x_los", "x_next_v"]

        # processing all keys except seq_keys
        for _, sample in enumerate(batch):
            for col, v in sample.items():
                if col not in seq_keys:
                    processed_batch[col].append(v)

        max_num_visits = max(processed_batch["seq_length"])
        diag_max_max_visit_items = max(processed_batch["diag_max_visit_items"])
        proc_max_max_visit_items = max(processed_batch["proc_max_visit_items"])
        drug_max_max_visit_items = max(processed_batch["drug_max_visit_items"])
        service_max_max_visit_items = max(processed_batch["service_max_visit_items"])
        admtype_max_max_visit_items = max(processed_batch["admtype_max_visit_items"])

        processed_batch.pop("diag_max_visit_items")
        processed_batch.pop("proc_max_visit_items")
        processed_batch.pop("drug_max_visit_items")
        processed_batch.pop("service_max_visit_items")
        processed_batch.pop("admtype_max_visit_items")

        # processing seq_keys
        for _, sample in enumerate(batch):
            padded_diag_seq = MIMICDataset.pad_sequence(
                sample["diag_seq"], max_num_visits, diag_max_max_visit_items
            )
            processed_batch["diag_seq"].append(padded_diag_seq.unsqueeze(0))
            padded_proc_seq = MIMICDataset.pad_sequence(
                sample["proc_seq"], max_num_visits, proc_max_max_visit_items
            )
            processed_batch["proc_seq"].append(padded_proc_seq.unsqueeze(0))
            padded_drg_seq = MIMICDataset.pad_sequence(
                sample["drg_seq"], max_num_visits, drug_max_max_visit_items
            )
            processed_batch["drg_seq"].append(padded_drg_seq.unsqueeze(0))
            padded_service_seq = MIMICDataset.pad_sequence(
                sample["service"], max_num_visits, service_max_max_visit_items
            )
            processed_batch["service"].append(padded_service_seq.unsqueeze(0))
            padded_admtype_seq = MIMICDataset.pad_sequence(
                sample["admtype"], max_num_visits, admtype_max_max_visit_items
            )
            processed_batch["admtype"].append(padded_admtype_seq.unsqueeze(0))

            for col in single_pad_visit_cols:
                if max_num_visits - sample[col].shape[0] < 0:
                    breakpoint()
                padded_col = np.pad(
                    sample[col], pad_width=(0, max_num_visits - sample[col].shape[0])
                )
                processed_batch[col].append(padded_col)

        processed_batch["y_outcome"] = torch.LongTensor(processed_batch["y_outcome"])
        # processed_batch["y_los"] = torch.FloatTensor(processed_batch["y_los"])
        processed_batch["gender"] = torch.FloatTensor(processed_batch["gender"])
        processed_batch["seq_length"] = torch.FloatTensor(processed_batch["seq_length"])
        processed_batch["ethnicity"] = torch.FloatTensor(processed_batch["ethnicity"])
        processed_batch["language"] = torch.FloatTensor(processed_batch["language"])
        processed_batch["age"] = torch.FloatTensor(processed_batch["age"])

        processed_batch["diag_seq"] = torch.cat(processed_batch["diag_seq"], dim=0)
        processed_batch["proc_seq"] = torch.cat(processed_batch["proc_seq"], dim=0)
        processed_batch["drg_seq"] = torch.cat(processed_batch["drg_seq"], dim=0)
        processed_batch["service"] = torch.cat(processed_batch["service"], dim=0)
        processed_batch["admtype"] = torch.cat(processed_batch["admtype"], dim=0)

        for col in single_pad_visit_cols:
            if col in float_cols:
                processed_batch[col] = torch.FloatTensor(processed_batch[col])
            else:
                processed_batch[col] = torch.LongTensor(processed_batch[col])
        return processed_batch
