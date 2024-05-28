
import pandas as pd
import os
def compute_metrics_binary(pred_df):
    pred_df["predicted_error_type_binary"] = pred_df["predicted_error_type"].apply(lambda x: 1 if x == 'ok' else 0)
    pred_df["error_type_ok_binary"] = pred_df["error_type_ok"].apply(lambda x: 1 if x == 'ok' else 0)

    pred_df["TP"] = ((pred_df["predicted_error_type_binary"] == 1) & (pred_df["error_type_ok_binary"] == 1)).astype(int)
    TP = pred_df["TP"].sum()

    pred_df["FP"] = ((pred_df["predicted_error_type_binary"] == 1) & (pred_df["error_type_ok_binary"] == 0)).astype(int)
    FP = pred_df["FP"].sum()

    pred_df["FN"] = ((pred_df["predicted_error_type_binary"] == 0) & (pred_df["error_type_ok_binary"] == 1)).astype(int)
    FN = pred_df["FN"].sum()

    pred_df["TN"] = ((pred_df["predicted_error_type_binary"] == 0) & (pred_df["error_type_ok_binary"] == 0)).astype(int)
    TN = pred_df["TN"].sum()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    exact_match = (pred_df["predicted_error_type_binary"] == pred_df["error_type_ok_binary"]).mean()

    return {
        "exact_match": exact_match,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }



if __name__ == "__main__":
    df = pd.read_csv(os.path.join("../../","outputs/retriever_evaluation","binary_mini_mmlu_groundtruth_correctness_zeroshot_llama_virology.csv"))
    print(compute_metrics_binary(df))
        