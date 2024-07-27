import pandas as pd
import os


def compute_metrics_binary(pred_df):
    pred_df["predicted_error_type_binary"] = pred_df["predicted_error_type"].apply(
        lambda x: 1 if x == "ok" else 0
    )
    pred_df["error_type_ok_binary"] = pred_df["error_type_ok"].apply(
        lambda x: 1 if x == "ok" else 0
    )

    pred_df["TP"] = (
        (pred_df["predicted_error_type_binary"] == 1)
        & (pred_df["error_type_ok_binary"] == 1)
    ).astype(int)
    TP = pred_df["TP"].sum()

    pred_df["FP"] = (
        (pred_df["predicted_error_type_binary"] == 1)
        & (pred_df["error_type_ok_binary"] == 0)
    ).astype(int)
    FP = pred_df["FP"].sum()

    pred_df["FN"] = (
        (pred_df["predicted_error_type_binary"] == 0)
        & (pred_df["error_type_ok_binary"] == 1)
    ).astype(int)
    FN = pred_df["FN"].sum()

    pred_df["TN"] = (
        (pred_df["predicted_error_type_binary"] == 0)
        & (pred_df["error_type_ok_binary"] == 0)
    ).astype(int)
    TN = pred_df["TN"].sum()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    f1_score = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    exact_match = (
        pred_df["predicted_error_type_binary"] == pred_df["error_type_ok_binary"]
    ).mean()

    npv = TN / (TN + FN) if (TN + FN) > 0 else 0
    tnr = TN / (TN + FP) if (TN + FP) > 0 else 0
    neg_f1_score = (2 * npv * tnr) / (npv + tnr) if (npv + tnr) > 0 else 0
    neg_f2_score = (5 * npv * tnr) / (4 * npv + tnr) if (4 * npv + tnr) > 0 else 0

    return {
        "exact_match": exact_match,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
        "npv": round(npv, 4),
        "tnr": round(tnr, 4),
        "neg_f1": round(neg_f1_score, 4),
        "neg_f2": round(neg_f2_score, 4),
    }


if __name__ == "__main__":
    base_name = "binary_mini_mmlu_groundtruth_correctness_zeroshot"

    # models = ["llama","gpt-4-turbo","gpt4"]
    models = ["claude"]

    dataset = [
        "college_chemistry",
        "college_mathematics",
        "econometrics",
        "formal_logic",
        "global_facts",
        "high_school_physics",
        "machine_learning",
        "professional_law",
        "public_relations",
        "virology",
    ]

    # methods = ["","_cot","_simple_prompt"]
    methods = ["_simple_prompt"]
    index = ["msmarco-v1-passage", "enwiki-paragraphs"]

    with open("./results.txt", "w") as f:
        for m in models:
            for i in index:
                for met in methods:
                    # if m =="llama" and (met =="_cot"):
                    #     continue

                    print(m, met, i, file=f)
                    em_av = 0
                    f1_av = 0
                    for d in dataset:
                        df = pd.read_csv(
                            os.path.join(
                                "../../outputs",
                                "retriever_evaluation",
                                base_name + met + "_" + m + "_" + d + "_" + i + ".csv",
                            )
                        )
                        res = compute_metrics_binary(df)
                        em_av += res["exact_match"]
                        f1_av += round(res["f1_score"], 4)

                        # print(str(res["exact_match"])+"/"+str(round(res["f1_score"],2)),file=f)
                        print(
                            d,
                            ",",
                            res["TP"],
                            ",",
                            res["TN"],
                            ",",
                            res["FN"],
                            ",",
                            res["FP"],
                            ",",
                            res["precision"],
                            ",",
                            res["recall"],
                            ",",
                            res["f1_score"],
                            ",",
                            res["npv"],
                            ",",
                            res["tnr"],
                            ",",
                            res["neg_f1"],
                            ",",
                            res["neg_f2"],
                            file=f,
                        )

                    print("", file=f)
                    print(
                        round((em_av / len(dataset)), 4),
                        "/",
                        round((f1_av / len(dataset)), 4),
                        end="  ",
                        file=f,
                    )
                    print("\n", file=f)
