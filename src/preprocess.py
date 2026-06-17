import os
import argparse
import numpy as np
import pandas as pd

LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax',
]


def expand_labels(df):
    """Add 14 binary columns from the pipe-separated 'Finding Labels' string."""
    finding = df['Finding Labels'].fillna('')
    for label in LABELS:
        df[label] = finding.str.contains(label, regex=False).astype('float32')
    return df


def undersample_no_finding(df, ratio, seed):
    has_finding = df[LABELS].sum(axis=1) > 0
    pos = df[has_finding]
    neg = df[~has_finding]

    keep_neg = int(len(pos) * ratio)
    if keep_neg < len(neg):
        neg = neg.sample(n=keep_neg, random_state=seed)

    out = pd.concat([pos, neg]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def pos_weight(df, cap):
    """neg/pos ratio per class, capped — passed to BCEWithLogitsLoss."""
    w = {}
    for label in LABELS:
        pos = float(df[label].sum())
        neg = float(len(df) - pos)
        w[label] = min(neg / max(pos, 1.0), cap)
    return w


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archive', required=True,help="Folder with Data_Entry_2017.csv, train_val_list.txt, test_list.txt")
    p.add_argument('--out', required=True, help="Where to write train/val/test CSVs")
    p.add_argument('--val-frac', type=float, default=0.10, help="Patient-wise val fraction of train_val")
    p.add_argument('--nf-ratio', type=float, default=0.5, help="No-Finding images per finding image in TRAIN")
    p.add_argument('--pos-cap', type=float, default=10.0, help="Cap on per-class pos_weight")
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # load + clean 
    df = pd.read_csv(os.path.join(args.archive, 'Data_Entry_2017.csv'))
    df = df[df['Patient Age'] <= 100].copy()
    df['Patient Gender'] = (df['Patient Gender'] == 'F').astype('float32')   # M=0, F=1
    df['Patient Age'] = df['Patient Age'].astype('float32') / 100.0
    df = expand_labels(df)

    cols = ['Image Index', 'Patient ID', 'Patient Age', 'Patient Gender'] + LABELS
    df = df[cols]

    # official split
    train_val_files = set(open(os.path.join(args.archive, 'train_val_list.txt')).read().split())
    test_files       = set(open(os.path.join(args.archive, 'test_list.txt')).read().split())

    trainval = df[df['Image Index'].isin(train_val_files)].copy()
    test     = df[df['Image Index'].isin(test_files)].copy()

    # patient-wise val split (no patient appears in both train and val) ───
    patients = trainval['Patient ID'].unique()
    rng.shuffle(patients)
    n_val = int(len(patients) * args.val_frac)
    val_patients = set(patients[:n_val])

    val   = trainval[trainval['Patient ID'].isin(val_patients)].copy()
    train = trainval[~trainval['Patient ID'].isin(val_patients)].copy()

    # undersample No-Finding in TRAIN only 
    train_bal = undersample_no_finding(train, ratio=args.nf_ratio, seed=args.seed)

    # save 
    drop_pid = lambda d: d.drop(columns=['Patient ID'])
    drop_pid(train_bal).to_csv(os.path.join(args.out, 'train.csv'), index=False)
    drop_pid(val).to_csv(os.path.join(args.out, 'val.csv'), index=False)
    drop_pid(test).to_csv(os.path.join(args.out, 'test.csv'), index=False)

    # report
    def healthy_frac(d):
        return float((d[LABELS].sum(axis=1) == 0).mean())

    print("=" * 64)
    print(f"  train (raw)       : {len(train):>7,}  | healthy {healthy_frac(train):.1%}")
    print(f"  train (balanced)  : {len(train_bal):>7,}  | healthy {healthy_frac(train_bal):.1%}")
    print(f"  val               : {len(val):>7,}  | healthy {healthy_frac(val):.1%}")
    print(f"  test              : {len(test):>7,}  | healthy {healthy_frac(test):.1%}")
    print("-" * 64)
    print("  per-class positives in balanced train  |  pos_weight (cap %.0f)" % args.pos_cap)
    w = pos_weight(train_bal, cap=args.pos_cap)
    for label in LABELS:
        print(f"    {label:20s} {int(train_bal[label].sum()):>6d}   ->   {w[label]:5.2f}")
    print("-" * 64)
    print("  POS_WEIGHT (paste into train.py if not auto-computed):")
    print("    [" + ", ".join(f"{w[l]:.2f}" for l in LABELS) + "]")
    print("=" * 64)
    print(f"  wrote train.csv / val.csv / test.csv -> {args.out}")


if __name__ == '__main__':
    main()
