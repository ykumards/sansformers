import numpy as np

from dataset.vocab import SequenceVocabulary


class EHRCountVectorizer:
    """
    Vectorizer to convert a sequence of ICD10 code to vectorized arrays
    In this specific instance, output is not a sequence.
    """

    def __init__(self, seq_vocab: SequenceVocabulary):
        self.seq_vocab = seq_vocab
        self.seq_vocab_len = len(self.seq_vocab)

    def vectorize(self, patient_history_seq: str, sep=";"):
        """
        Convert the patient history sequence (diagnoses, procedures, etc) to a list of indeces based on the
        vocabulary object.
        This method also handles the padding of the "inner-list", i.e.,
        the number of diagnoses per visit. Padding the number of visits sequence
        is handled by the collate function in trainer.
        Args:
          diagnoses: String of diagnoses for a patient, each visit
                     separated by ';', diagnoses per
                     visit separated by space
        Returns:
          vectorized_history: list of vectorized visit codes
          n_tkns_per_visit: list of number of tokens per visit
        """
        # split
        visits = [visit for visit in patient_history_seq.split(sep)]
        n_tkns_per_visit = [
            len(visit.split(" ")) for visit in patient_history_seq.split(sep)
        ]

        vectorized_history = []
        max_visit_items = 0
        for visit in visits:
            items_per_visit_i = [
                self.seq_vocab.lookup_token(token) for token in visit.split(" ")
            ]
            items_per_visit_i_length = len(items_per_visit_i)

            if max_visit_items < items_per_visit_i_length:
                max_visit_items = items_per_visit_i_length
            vectorized_history.append(items_per_visit_i)

        return vectorized_history, n_tkns_per_visit

    @classmethod
    def from_dataframe_cols(cls, df, col_names):
        seq_vocab = SequenceVocabulary()
        for colname in col_names:
            df[colname] = df[colname].apply(
                lambda row: row.replace(";", " ").split(" ")
            )
            seq_vocab.add_tokens(np.concatenate(df[colname].values))
        print(f"Corpus has {len(seq_vocab)} unique tokens")
        return cls(seq_vocab)

    @classmethod
    def from_dataframe(cls, df, colname="icd10"):
        seq_vocab = SequenceVocabulary()
        df[colname] = df[colname].apply(lambda row: row.replace(";", " ").split(" "))
        seq_vocab.add_tokens(np.concatenate(df[colname].values))

        print(f"Corpus has {len(seq_vocab)} unique tokens")
        return cls(seq_vocab)
