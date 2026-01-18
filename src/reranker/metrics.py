import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def sub_exact_match(prediction: str, golden_answers) -> float:

    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    normalized_prediction = normalize_answer(prediction)
    score = 0.0

    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1.0
            break

    return score


def batch_sub_exact_match(
    pred_list,
    golden_answers_list,
):
    score_list = [
        sub_exact_match(pred, golden_answers)
        for pred, golden_answers in zip(pred_list, golden_answers_list)
    ]
    # print(score_list)
    avg_score = sum(score_list) / len(score_list)
    return avg_score, score_list


def token_level_f1(prediction: str, ground_truths: list):

    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    final_metric = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    normalized_prediction = normalize_answer(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)

        # Special cases for yes/no/noanswer
        if (
            normalized_prediction in ["yes", "no", "noanswer"]
            and normalized_prediction != normalized_ground_truth
        ):
            continue
        if (
            normalized_ground_truth in ["yes", "no", "noanswer"]
            and normalized_prediction != normalized_ground_truth
        ):
            continue

        pred_tokens = normalized_prediction.split()
        gt_tokens = normalized_ground_truth.split()
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            continue

        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall)

        final_metric["precision"] = max(final_metric["precision"], precision)
        final_metric["recall"] = max(final_metric["recall"], recall)
        final_metric["f1"] = max(final_metric["f1"], f1)

    return final_metric


def dataset_level_f1(pred_list: list, golden_answers_list: list):

    metric_scores = []
    for pred, golds in zip(pred_list, golden_answers_list):
        score = token_level_f1(pred, golds)["f1"]
        metric_scores.append(score)

    avg_f1 = sum(metric_scores) / len(metric_scores) if metric_scores else 0.0
    # return {"f1": avg_f1}, metric_scores
    return avg_f1, metric_scores
