#%%
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tqdm.auto import tqdm

MIMIC_PATH = "path/to/mimic"
HOSP_PATH = MIMIC_PATH / "hosp"
CORE_PATH = MIMIC_PATH / "core"

TARGET_OFFSET = 2

#%%
admits = pd.read_csv(
    CORE_PATH / "admissions.csv.gz",
    parse_dates=["admittime", "dischtime"],
    compression="gzip",
)
patients = pd.read_csv(CORE_PATH / "patients.csv.gz", compression="gzip")
patients = patients.merge(admits, on="subject_id", how="right")
patients["los"] = patients["dischtime"] - patients["admittime"]
patients["los"] = patients["los"].dt.total_seconds() / 86400.0
# topcap los at 180 days
patients["los"] = np.clip(patients["los"], 0, 180)
patients.drop(
    [
        "discharge_location",
        "anchor_year_group",
        "anchor_year",
        "admission_location",
        "discharge_location",
        "edregtime",
        "edouttime",
    ],
    axis=1,
    inplace=True,
)
patients.sort_values(["subject_id", "admittime", "dischtime"], inplace=True)
patients["days_from_prev"] = patients["admittime"] - patients["admittime"].shift(1)
patients["days_from_prev"] = patients["days_from_prev"].dt.total_seconds() / 86400.0
bin_edges = np.arange(0, np.max(patients["days_from_prev"]) + 16, 15)
patients["days_from_prev"] = np.digitize(patients["days_from_prev"].values, bin_edges)

# store the last recorded hospital_expire_flag as the flag for the patient
patients = patients.drop("hospital_expire_flag", axis=1).join(
    patients.groupby("subject_id")[["subject_id", "hospital_expire_flag"]]
    .tail(1)
    .set_index("subject_id"),
    on="subject_id",
    how="inner",
)

# drop last visits for each patient
hadm_to_drop = (
    patients.groupby("subject_id")[["dischtime", "hadm_id"]].tail(TARGET_OFFSET).hadm_id
)
patients = patients[~patients["hadm_id"].isin(hadm_to_drop)]

# processing diagnosis codes
diags = pd.read_csv(HOSP_PATH / "diagnoses_icd.csv.gz", compression="gzip")
diags["icd_code"] = diags["icd_code"].apply(lambda row: row[:3])
diags["icd_version"] = diags["icd_version"].apply(lambda row: f"IP{row}_")
diags["icd_all"] = diags["icd_version"] + diags["icd_code"]
print(diags.head())

# group diag codes according to admission id
n_unique_admid = diags.hadm_id.nunique()
diags.set_index("hadm_id", inplace=True)
diag_cols = ["subject_id", "hadm_id", "icd_all"]
grouped_dict = {k: [] for k in diag_cols}

for g, f in tqdm(diags.groupby(diags.index), total=n_unique_admid):
    grouped_dict["hadm_id"].append(g)
    grouped_dict["subject_id"].append(f.subject_id.values[0])
    for col in diag_cols:
        if col not in ["hadm_id", "subject_id"]:
            grouped_dict[col].append(" ".join(f[col].values))

grouped_diag = pd.DataFrame(grouped_dict)
print(grouped_diag.head())
grouped_diag.to_feather(MIMIC_PATH / "grouped_diag.feather")

# grouped_diag = pd.read_feather(MIMIC_PATH/"grouped_diag.feather")

# processing procedure codes
procs = pd.read_csv(HOSP_PATH / "procedures_icd.csv.gz", compression="gzip")
procs["icd_code"] = procs["icd_code"].apply(lambda row: row[:3])
procs["icd_version"] = procs["icd_version"].apply(lambda row: f"IP{row}_")
procs["icd_all"] = procs["icd_version"] + procs["icd_code"]
print(procs.head())

# group proc codes according to admit id
n_unique_admid = procs.hadm_id.nunique()
procs.set_index("hadm_id", inplace=True)
diag_cols = ["subject_id", "hadm_id", "icd_all"]
grouped_dict = {k: [] for k in diag_cols}

for g, f in tqdm(procs.groupby(procs.index), total=n_unique_admid):
    grouped_dict["hadm_id"].append(g)
    grouped_dict["subject_id"].append(f.subject_id.values[0])
    for col in diag_cols:
        if col not in ["hadm_id", "subject_id"]:
            grouped_dict[col].append(" ".join(f[col].values))

grouped_proc = pd.DataFrame(grouped_dict)
grouped_proc.columns = ["subject_id", "hadm_id", "proc_all"]
print(grouped_proc.head())
grouped_proc.to_feather(MIMIC_PATH / "grouped_proc.feather")

# process drg codes
drgs = pd.read_csv(HOSP_PATH / "drgcodes.csv.gz", compression="gzip")
drgs.drop(
    ["drg_type", "description", "drg_severity", "drg_mortality"], axis=1, inplace=True
)
drgs["drg_code"] = drgs["drg_code"].astype(str)
print(drgs.head())

n_unique_admid = drgs.hadm_id.nunique()
drgs.set_index("hadm_id", inplace=True)
diag_cols = ["subject_id", "hadm_id", "drg_code"]
grouped_dict = {k: [] for k in diag_cols}

for g, f in tqdm(drgs.groupby(drgs.index), total=n_unique_admid):
    grouped_dict["hadm_id"].append(g)
    grouped_dict["subject_id"].append(f.subject_id.values[0])
    for col in diag_cols:
        if col not in ["hadm_id", "subject_id"]:
            grouped_dict[col].append(" ".join(f[col].values))

grouped_drg = pd.DataFrame(grouped_dict)
grouped_drg.columns = ["subject_id", "hadm_id", "drg_all"]
print(grouped_drg.head())
grouped_drg.to_feather(MIMIC_PATH / "grouped_drg.feather")

# process services
services = pd.read_csv(HOSP_PATH / "services.csv.gz", compression="gzip")
services.drop(["transfertime", "prev_service"], axis=1, inplace=True)
print(services.head())

n_unique_admid = services.hadm_id.nunique()
services.set_index("hadm_id", inplace=True)
diag_cols = ["subject_id", "hadm_id", "curr_service"]
grouped_dict = {k: [] for k in diag_cols}

for g, f in tqdm(services.groupby(services.index), total=n_unique_admid):
    grouped_dict["hadm_id"].append(g)
    grouped_dict["subject_id"].append(f.subject_id.values[0])
    for col in diag_cols:
        if col not in ["hadm_id", "subject_id"]:
            grouped_dict[col].append(" ".join(f[col].values))

grouped_service = pd.DataFrame(grouped_dict)
grouped_service.columns = ["subject_id", "hadm_id", "service_all"]
print(grouped_service.head())
grouped_service.to_feather(MIMIC_PATH / "grouped_service.feather")

# merging everything together
patients = patients.merge(grouped_diag, on=["subject_id", "hadm_id"], how="left")
patients = patients.merge(grouped_proc, on=["subject_id", "hadm_id"], how="left")
patients = patients.merge(grouped_drg, on=["subject_id", "hadm_id"], how="left")
patients = patients.merge(grouped_service, on=["subject_id", "hadm_id"], how="left")

# handle missing rows
for col in patients.columns:
    patients[col].fillna("missing", inplace=True)

patients.reset_index(inplace=True)
n_unique_patid = patients.subject_id.nunique()
patients.set_index("subject_id", inplace=True)
pat_cols = patients.columns.tolist() + ["subject_id"]
for col in patients.columns:
    patients[col] = patients[col].astype(str)
grouped_dict = {k: [] for k in pat_cols}

for g, f in tqdm(patients.groupby(patients.index), total=n_unique_patid):
    grouped_dict["subject_id"].append(g)
    for col in pat_cols:
        if col not in ["subject_id"]:
            grouped_dict[col].append(";".join(f[col].values))

grouped_patient = pd.DataFrame(grouped_dict)
grouped_patient["days_from_prev"] = grouped_patient["days_from_prev"].apply(
    lambda row: ";".join(["0.0"] + row.split(";")[1:])
)
num_cols = [
    "gender",
    "anchor_age",
    "dod",
    "deathtime",
    "language",
    "ethnicity",
    "hospital_expire_flag",
]
for col in num_cols:
    grouped_patient[col] = grouped_patient[col].apply(lambda row: row.split(";")[-1])
print(grouped_patient.hospital_expire_flag.value_counts())
print(grouped_patient.head())
grouped_patient.to_feather(MIMIC_PATH / f"grouped_patient_rem{TARGET_OFFSET}.feather")

#%%
mimic_df = pd.read_feather(MIMIC_PATH / f"grouped_patient_rem{TARGET_OFFSET}.feather")
# %%
mimic_df["seq_length"] = mimic_df["hadm_id"].apply(lambda row: len(row.split(";")))


def remove_puncts(code):
    return re.sub("[^A-Za-z<>]+", "", code)


# encode speciality
all_gender = []
for idx in mimic_df.index:
    all_gender += mimic_df.loc[idx, "gender"].split(";")

all_gender = set(all_gender)
le_gen = {ch: i for i, ch in enumerate(all_gender)}

# encode lang
all_lang = []
for idx in mimic_df.index:
    all_lang += mimic_df.loc[idx, "language"].split(";")

all_lang = set(all_lang)
le_lang = {ch: i for i, ch in enumerate(all_lang)}

# encode ethn
all_ethn = []
for idx in mimic_df.index:
    all_ethn += mimic_df.loc[idx, "ethnicity"].split(";")

all_ethn = set(all_ethn)
le_ethn = {ch: i for i, ch in enumerate(all_ethn)}

#%%


print("Creating vectorizers...")
import sys

sys.path.append("../src/")
from dataset.vectorizer import EHRCountVectorizer


def apply_prefix(row, prefix):
    visits = row.split(";")
    out_visits = []
    for visit in visits:
        out_visits.append(" ".join([prefix + tok for tok in visit.split()]))
    return ";".join(out_visits)


mimic_df["icd_all"] = mimic_df["icd_all"].apply(lambda row: apply_prefix(row, "diag"))
mimic_df["proc_all"] = mimic_df["proc_all"].apply(lambda row: apply_prefix(row, "proc"))
mimic_df["drg_all"] = mimic_df["drg_all"].apply(lambda row: apply_prefix(row, "drg"))
mimic_df["admission_type"] = mimic_df["admission_type"].apply(
    lambda row: apply_prefix(row, "admtype")
)
mimic_df["insurance"] = mimic_df["insurance"].apply(
    lambda row: apply_prefix(row, "insur")
)
mimic_df["marital_status"] = mimic_df["marital_status"].apply(
    lambda row: apply_prefix(row, "marit")
)
mimic_df["service_all"] = mimic_df["service_all"].apply(
    lambda row: apply_prefix(row, "serv")
)
#%%
#%%
mimic_df["encoded_gender"] = mimic_df["gender"].apply(
    lambda row: le_gen[row.split(";")[0]]
)
mimic_df["encoded_ethnicity"] = mimic_df["ethnicity"].apply(
    lambda row: le_ethn[row.split(";")[0]]
)
mimic_df["encoded_language"] = mimic_df["language"].apply(
    lambda row: le_lang[row.split(";")[0]]
)
#%%
train, test = train_test_split(mimic_df, test_size=0.2, random_state=100)
train.reset_index(drop=True).to_feather(
    MIMIC_PATH / f"mimic_train_rem{TARGET_OFFSET}.feather"
)
test.reset_index(drop=True).to_feather(
    MIMIC_PATH / f"mimic_test_rem{TARGET_OFFSET}.feather"
)
