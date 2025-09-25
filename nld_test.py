import os, json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def evaluate_noise_detection_auto(savedir1: str,
                                  savedir2: str,
                                  json_path: str,
                                  outfile_name: str = "noise_detection_eval.csv"):
    

    td = torch.load(os.path.join(savedir1, "train_data.pth"))
    train_indices = td["train_indices"].cpu().numpy().astype(int)  

    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    true_all  = np.array([int(r["True_Label"])  for r in records], dtype=int)
    after_all = np.array([int(r["After_Label"]) for r in records], dtype=int)
    true_targets    = true_all[train_indices]
    after_targets   = after_all[train_indices]
    actual_is_noisy = (true_targets != after_targets)   

    df1 = pd.read_csv(os.path.join(savedir1, "sei_values.csv"))
    df2 = pd.read_csv(os.path.join(savedir2, "sei_values.csv"))
    for df in (df1, df2):
        df["sample_id"] = pd.to_numeric(df["sample_id"], errors="raise")
        if "sei" not in df.columns:
            cand = [c for c in df.columns if c.lower().startswith("sei")]
            df.rename(columns={cand[0]: "sei"}, inplace=True)
    df = df1.merge(df2, on="sample_id", suffixes=("_1", "_2"))

    det1 = pd.read_csv(os.path.join(savedir1, "sei_details.csv"))
    det2 = pd.read_csv(os.path.join(savedir2, "sei_details.csv"))
    thr1 = set(pd.to_numeric(det1.loc[det1["Is Threshold Sample"], "Index"]).astype(int))
    thr2 = set(pd.to_numeric(det2.loc[det2["Is Threshold Sample"], "Index"]).astype(int))

    def pick_sei(row):
        sid = int(row["sample_id"])
        if (sid in thr1) and (sid not in thr2):
            return row["sei_2"]
        if (sid in thr2) and (sid not in thr1):
            return row["sei_1"]
        return 0.5 * (row["sei_1"] + row["sei_2"])
    df["sei_final"] = df.apply(pick_sei, axis=1)

    df["global_idx"]      = df["sample_id"].map(lambda i: int(train_indices[int(i)]))
    df["actual_is_noisy"] = df["sample_id"].map(lambda i: bool(actual_is_noisy[int(i)]))
    y_true = df["actual_is_noisy"].values

    thr_vals_1 = df.loc[df["sample_id"].isin(thr1), "sei_1"].to_numpy()
    thr_vals_2 = df.loc[df["sample_id"].isin(thr2), "sei_2"].to_numpy()
    thr_vals = np.concatenate([thr_vals_1, thr_vals_2], axis=0)
    thr_vals = thr_vals[np.isfinite(thr_vals)]
    if thr_vals.size == 0:
        all_vals = df["sei_final"].values
        all_vals = all_vals[np.isfinite(all_vals)]
        cutoff_star = float(all_vals.mean())
        thr_count = 0
    else:
        cutoff_star = float(thr_vals.mean())
        thr_count = thr_vals.size  

    
    y_pred = (df["sei_final"].values < cutoff_star)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    df["pred_is_noisy"] = y_pred

    print("\n—— Noise Detection————————")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  Precision  : {prec:.4f}")
    print(f"  Recall     : {rec:.4f}")
    print(f"  F1         : {f1:.4f}")
    print("———————————————————————————————————————————————")

    out_cols = ["sample_id", "global_idx", "sei_final", "pred_is_noisy", "actual_is_noisy"]
    df[out_cols].to_csv(os.path.join(savedir1, outfile_name), index=False)

    meta = {
        "acc": acc, "prec": prec, "rec": rec, "f1": f1,
        "num_threshold_entries": thr_count  
    }
    pd.Series(meta).to_json(os.path.join(savedir1, "noise_detection_eval_meta.json"), indent=2)
    return df[out_cols], meta


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()

    p.add_argument("--savedir1", required=True, help="Directory of the first run (threshold=1)")
    p.add_argument("--savedir2", required=True, help="Directory of the second run (threshold=2)")
    p.add_argument("--json_path", required=True, help="Path to the synthetic noise data JSON file.")
    p.add_argument("--outfile", default="noise_detection_eval.csv")
    args = p.parse_args()

    evaluate_noise_detection_auto(
        savedir1=args.savedir1,
        savedir2=args.savedir2,
        json_path=args.json_path,
        outfile_name=args.outfile
    )
