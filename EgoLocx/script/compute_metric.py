import json
import pandas as pd
import numpy as np

def evaluate_predictions(json_path, gt_excel_path, sheet_name):
    def evaluate_single_video(ground_truth, prediction, total_frame):
        gt_contact, gt_separation = ground_truth
        pred_contact, pred_separation = prediction

        if gt_contact == 0 and gt_separation == 0:
            return None
        if pred_contact == 0 and pred_separation == 0:
            return None

        if gt_contact == 0:
            contact_error = None
            contact_correct1 = contact_correct2 = contact_correct3 = None
        else:
            contact_error = abs(gt_contact - pred_contact)
            contact_correct1 = contact_error <= 1
            contact_correct2 = contact_error <= 3
            contact_correct3 = contact_error <= 5

        if gt_separation == 0:
            separation_error = None
            separation_correct1 = separation_correct2 = separation_correct3 = None
        else:
            separation_error = abs(gt_separation - pred_separation)
            separation_correct1 = separation_error <= 1
            separation_correct2 = separation_error <= 3
            separation_correct3 = separation_error <= 5

        total_checks = 0
        acc1 = acc2 = acc3 = 0
        if contact_error is not None:
            acc1 += contact_correct1
            acc2 += contact_correct2
            acc3 += contact_correct3
            total_checks += 1
        if separation_error is not None:
            acc1 += separation_correct1
            acc2 += separation_correct2
            acc3 += separation_correct3
            total_checks += 1
        accuracy1 = acc1 / total_checks if total_checks else 0
        accuracy2 = acc2 / total_checks if total_checks else 0
        accuracy3 = acc3 / total_checks if total_checks else 0

        errors = [e for e in (contact_error, separation_error) if e is not None]
        temporal_error = np.mean(errors) if errors else 0
        temporal_similarity = 1 / (1 + temporal_error) if errors else np.nan

        MoF = IoU = None
        if gt_contact and gt_separation:
            correct = 0
            # (0,0) from prediction means no event
            pred_has_event = pred_contact > 0 and pred_separation > 0
            for f in range(total_frame):
                in_gt = gt_contact <= f <= gt_separation
                in_pred = pred_has_event and (pred_contact <= f <= pred_separation)
                if in_gt == in_pred:
                    correct += 1
            MoF = correct / total_frame
            gt_set = set(range(gt_contact, gt_separation + 1))
            pred_set = set(range(pred_contact, pred_separation + 1)) if pred_has_event else set()
            union = gt_set | pred_set
            IoU = len(gt_set & pred_set) / len(union) if union else 0

        return {
            "Accuracy_1": accuracy1,
            "Accuracy_2": accuracy2,
            "Accuracy_3": accuracy3,
            "Temporal Error (MAE)": temporal_error,
            "Temporal Similarity": temporal_similarity,
            "MoF": MoF,
            "IoU": IoU
        }

    # load ground truth
    df = pd.read_excel(gt_excel_path, sheet_name=sheet_name)
    gt_list = []
    total_frames = []

    end = len(df) 
    for i in range(1, end):
        # print(i)
        # print(df.iloc[i,1])
        # print(df.iloc[i, 0])
        total_frames.append(int(df.iloc[i, 1]))
        gt_list.append((int(df.iloc[i, 2]), int(df.iloc[i, 3])))
    
    # load predictions
    with open(json_path, "r") as f:
        raw = json.load(f)

    predictions = []
    for name, lst in raw:
        c, s = lst[0]
        predictions.append((int(c), int(s)))

    metrics = []
    for (gt, total), pred in zip(zip(gt_list, total_frames), predictions):
        m = evaluate_single_video(gt, pred, total)
        if m is not None:
            metrics.append(m)

    avg = {}
    if metrics:
        for key in metrics[0].keys():
            vals = [m[key] for m in metrics if m[key] is not None]
            if vals:
                avg[key] = float(np.nanmean(vals))

    return avg