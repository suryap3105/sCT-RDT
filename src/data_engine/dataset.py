import os
import gc
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from .masking import apply_synthetic_occlusion

REQUIRED_CSV_COLUMNS = {'object_id', 'mjd', 'passband', 'flux', 'flux_err'}

class AstroDataset(Dataset):
    def __init__(self, csv_path, max_seq_len, occlusion_level="0%", synthetic=False):
        """
        Loads CSV dataset if available, otherwise generates synthetic data if synthetic=True.
        Allows immediate testing of train.py / evaluate.py without needing Kaggle downloads.
        """
        self.max_seq_len = max_seq_len
        self.occlusion_level = occlusion_level
        self.synthetic = synthetic
        # FAILSAFE: always initialise mock_len so __len__ never raises AttributeError,
        # regardless of whether synthetic=True was passed directly or set via fallback.
        self.mock_len = 1000
        self.num_groups = 0

        if not synthetic:
            try:
                df = pd.read_csv(csv_path)
                # FAILSAFE: validate required columns exist before groupby
                missing_cols = REQUIRED_CSV_COLUMNS - set(df.columns)
                if missing_cols:
                    raise ValueError(
                        f"CSV '{csv_path}' is missing required columns: {missing_cols}. "
                        "Falling back to synthetic data."
                    )
                
                # BUGFIX: The actual PLAsTiCC time-series CSVs don't contain the 'target' column.
                # If missing, we must merge it from the metadata otherwise all labels default to 0!
                if 'target' not in df.columns:
                    base_dir = os.path.dirname(csv_path)
                    if 'training' in os.path.basename(csv_path):
                        meta_file = 'training_set_metadata.csv'
                    elif 'test' in os.path.basename(csv_path):
                        meta_file = 'test_set_metadata.csv'
                    else:
                        meta_file = os.path.basename(csv_path).replace('.csv', '_metadata.csv')
                    
                    meta_path = os.path.join(base_dir, meta_file)
                    if os.path.exists(meta_path):
                        df_meta = pd.read_csv(meta_path)
                        if 'target' in df_meta.columns:
                            # PyTorch CrossEntropy expects class indices in [0, num_classes-1].
                            # PLAsTiCC has raw astronomical IDs (e.g., 6, 15, 88). We must map them.
                            unique_classes = sorted(df_meta['target'].dropna().unique())
                            class_map = {val: idx for idx, val in enumerate(unique_classes)}
                            df_meta['mapped_target'] = df_meta['target'].map(class_map)
                            
                            df = df.merge(df_meta[['object_id', 'mapped_target']], on='object_id', how='left')
                            df.rename(columns={'mapped_target': 'target'}, inplace=True)
                            print(f"[AstroDataset] Merged targets from '{meta_file}' & mapped {len(unique_classes)} classes to [0-{len(unique_classes)-1}]")
                        else:
                            print(f"[WARN] '{meta_file}' found but lacks 'target' column. Labels will default to 0.")
                    else:
                        print(f"[WARN] Metadata '{meta_file}' not found. Labels will default to 0.")

                print(f"[AstroDataset] Compiling DataFrame to numpy arrays for memory efficiency...")
                df.sort_values(by=['object_id', 'mjd'], inplace=True)
                self.mjd = df['mjd'].values.astype(np.float32)
                self.flux = df['flux'].values.astype(np.float32)
                self.flux_err = df['flux_err'].values.astype(np.float32)
                self.passband = df['passband'].values.astype(np.int64)
                self.target = df['target'].values.astype(np.int64) if 'target' in df.columns else np.zeros(len(df), dtype=np.int64)
                
                _, start_indices, counts = np.unique(df['object_id'].values, return_index=True, return_counts=True)
                self.start_indices = start_indices
                self.counts = counts
                self.num_groups = len(self.start_indices)
                
                # Delete dataframe and invoke GC to free system memory immediately
                del df
                gc.collect()

                if self.num_groups == 0:
                    raise ValueError(f"CSV '{csv_path}' has no object groups. Falling back to synthetic data.")
            except FileNotFoundError:
                print(f"[AstroDataset] Warning: '{csv_path}' not found. Falling back to synthetic mock data.")
                self.synthetic = True
            except (ValueError, pd.errors.ParserError) as e:
                print(f"[AstroDataset] Warning: {e} Falling back to synthetic mock data.")
                self.synthetic = True

    def __len__(self):
        if self.synthetic:
            return self.mock_len
        return self.num_groups

    def __getitem__(self, index):
        if self.synthetic:
            # Generate mock object
            seq_len = np.random.randint(50, self.max_seq_len)
            raw_t = np.sort(np.random.uniform(0, 100, seq_len))
            raw_f = np.sin(raw_t) + np.random.normal(0, 1, seq_len)
            raw_e = np.random.uniform(0.1, 0.5, seq_len)
            p = np.random.randint(0, 6, seq_len)
            label = np.random.randint(0, 14)
        else:
            start = self.start_indices[index]
            end = start + self.counts[index]
            
            raw_t = self.mjd[start:end]
            raw_f = self.flux[start:end]
            raw_e = self.flux_err[start:end]
            p = self.passband[start:end]
            label = self.target[start]

        # 1. Apply Occlusion Benchmarks
        t, f, e, p = apply_synthetic_occlusion(raw_t, raw_f, raw_e, p, self.occlusion_level)
        
        # Handle empty sequences due to extreme masking
        if len(t) == 0:
            t, f, e, p = np.array([0]), np.array([0]), np.array([1]), np.array([0])

        # 2. Robust Normalization
        t_min, t_max = np.min(t), np.max(t)
        t_norm = (t - t_min) / (t_max - t_min + 1e-6)
        
        f_median = np.median(f)
        q75, q25 = np.percentile(f, [75 ,25])
        f_iqr = q75 - q25 + 1e-9
        
        f_norm = (f - f_median) / f_iqr
        e_norm = e / f_iqr
        
        # 3. Zero Padding to max_seq_len
        actual_length = len(t)
        pad_len = max(0, self.max_seq_len - actual_length)
        
        # Truncate if too long
        if actual_length > self.max_seq_len:
            t_norm = t_norm[:self.max_seq_len]
            f_norm = f_norm[:self.max_seq_len]
            e_norm = e_norm[:self.max_seq_len]
            p = p[:self.max_seq_len]
            actual_length = self.max_seq_len
            pad_len = 0
            
        padded_f = np.pad(f_norm, (0, pad_len), 'constant')
        padded_p = np.pad(p, (0, pad_len), 'constant')
        padded_t = np.pad(t_norm, (0, pad_len), 'constant')
        padded_e = np.pad(e_norm, (0, pad_len), 'constant', constant_values=1.0) # Error padded with 1.0 safely
        
        padding_mask = np.zeros(self.max_seq_len, dtype=bool)
        if pad_len > 0:
            padding_mask[actual_length:] = True
            
        return (
            torch.FloatTensor(padded_f),
            torch.LongTensor(padded_p),
            torch.FloatTensor(padded_t),
            torch.FloatTensor(padded_e),
            torch.BoolTensor(padding_mask),
            torch.tensor(label, dtype=torch.long)
        )

    def get_class_weights(self, num_classes):
        """Calculates inverse frequency weights to correct Severe Class Imbalance."""
        if self.synthetic:
            return torch.ones(num_classes)
            
        counts = np.bincount(self.target, minlength=num_classes)
        # Avoid division by zero for unrepresented classes
        counts = np.where(counts == 0, 1, counts)
        weights = 1.0 / counts
        
        # Normalize weights so they sum to num_classes (mean weight = 1.0)
        weights = weights / np.sum(weights) * num_classes
        return torch.FloatTensor(weights)
