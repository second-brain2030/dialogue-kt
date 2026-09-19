from typing import List, Union
import json
import re
from ast import literal_eval
import pandas as pd

COMTA_SUBJECTS = ["Elementary", "Algebra", "Trigonometry", "Geometry"]

def add_content(cur: str, new: str):
    new = new.strip()
    if not cur:
        return new
    if cur == new: # Sometimes turns are repeated in MathDial
        return cur
    if not cur.endswith((".", "!", "?")):
        cur += "."
    return cur + " " + new

def process_dialogue(turns: List[dict]):
    cur_role = turns[0]["role"]
    cur_turn = {
        "turn": 0 if cur_role == "student" else 1,
        "teacher": "",
        "student": ""
    }
    result = []
    for turn in turns:
        if turn["role"] == "teacher" and cur_role == "student":
            result.append(cur_turn)
            cur_turn = {
                "turn": cur_turn["turn"] + 1,
                "teacher": "",
                "student": ""
            }
        cur_role = turn["role"]
        cur_turn[cur_role] = add_content(cur_turn[cur_role], turn["content"])
    # Only include final turn if there was a student response (not always the case in MathDial)
    if cur_turn["student"]:
        result.append(cur_turn)
    return result

def correct_from_str(correct: str):
    return True if correct == "true" else False if correct == "false" else None

def correct_to_str(correct: Union[bool, None]):
    return "na" if correct is None else "true" if correct else False

def standards_to_str(standards: List[str], sep: str):
    return "None" if not standards else sep.join([f"{idx + 1}) {kc}" for idx, kc in enumerate(standards)])

def load_comta_src_data():
    with open("data/src/CoMTA_dataset.json") as file:
        data = json.load(file)
    proc_data = []
    for index, sample in enumerate(data):
        # Skip calculus since not in ATC
        if sample["math_level"] == "Calculus":
            continue
        # Add dialogue and meta data
        proc_data.append({
            "index": index,
            "dialogue": process_dialogue([
                {"role": "student" if turn["role"] == "user" else "teacher", "content": turn["content"]}
                for turn in sample["data"]
            ]),
            "meta_data": {
                "expected_result": sample["expected_result"],
                "math_level": sample["math_level"]
            }
        })
    return pd.DataFrame(proc_data)

def load_mathdial_src_data(split: str):
    turn_prefix_re = re.compile(r"^[a-zA-Z]+: (\([a-z]+\))?")
    with open(f"data/src/mathdial/data/{split}.jsonl") as file:
        data = [json.loads(line) for line in file]
    proc_data = []
    for index, sample in enumerate(data):
        if not sample["self-typical-confusion"] or not sample["self-typical-interactions"]:
            continue
        # Add dialogue and meta data
        proc_data.append({
            "index": index,
            **sample,
            "dialogue": process_dialogue([
                {"role": "teacher" if turn.startswith("Teacher") else "student", "content": turn_prefix_re.sub("", turn)}
                for turn in sample["conversation"].split("|EOM|")
            ]),
            "meta_data": {
                "question": sample["question"],
                "correct_solution": sample["ground_truth"],
                "incorrect_solution": sample["student_incorrect_solution"],
                "self_correctness": sample["self-correctness"],
                "self_typical_confusion": sample["self-typical-confusion"],
                "self_typical_interactions": sample["self-typical-interactions"]
            }
        })
    return pd.DataFrame(proc_data)

def load_src_data(args, split: str = ""):
    if args.dataset == "comta":
        return load_comta_src_data()
    elif args.dataset == "mathdial":
        return load_mathdial_src_data(split)
    raise Exception(f"Loading not supported for {args.dataset}")

def get_annotated_data_filename(args, split: str = ""):
    return f"data/annotated/{args.dataset}{f'_{split}' if split else ''}_{args.tag_src}.csv"

def get_kc_dict_filename(args):
    return f"data/annotated/kc_dict_{args.dataset}_{args.tag_src}.json"

def load_kc_dict(args):
    with open(get_kc_dict_filename(args)) as file:
        return json.load(file)

def get_default_fold(args):
    if args.dataset == "comta":
        if args.split_by_subject:
            return "Elementary"
        return 1
    else:
        return None

def load_annotated_data(args, fold: Union[int, str, None] = 1):
    if args.dataset == "comta":
        df = pd.read_csv(get_annotated_data_filename(args), converters={col: literal_eval for col in ["dialogue", "meta_data", "annotation"]})
        if args.split_by_subject:
            assert fold in COMTA_SUBJECTS
            subj_mask = df.apply(lambda row: row["meta_data"]["math_level"] == fold, axis=1)
            test_df = df[subj_mask]
            train_df = df[~subj_mask].sample(frac=1, random_state=221)
            return (
                train_df[:int(.8 * len(train_df))],
                train_df[int(.8 * len(train_df)):],
                test_df
            )
        else:
            assert fold in range(1, 6)
            df = df.sample(frac=1, random_state=221)
            split_point = int(len(df) * ((fold - 1) / 5))
            df = pd.concat([df[split_point:], df[:split_point]])
            return (
                df[:int(len(df) * .65)],
                df[int(len(df) * .65) : int(len(df) * .8)],
                df[int(len(df) * .8):],
            )
    elif args.dataset == "mathdial":
        def pass_typical_threshold(row):
            return (row["meta_data"]["self_typical_confusion"] >= args.typical_cutoff and
                    row["meta_data"]["self_typical_interactions"] >= args.typical_cutoff)

        train_df = pd.read_csv(get_annotated_data_filename(args, "train"), converters={col: literal_eval for col in ["dialogue", "meta_data", "annotation"]})
        train_df = train_df.sample(frac=1, random_state=221)
        train_df = train_df[train_df.apply(pass_typical_threshold, axis=1)]
        test_df = pd.read_csv(get_annotated_data_filename(args, "test"), converters={col: literal_eval for col in ["dialogue", "meta_data", "annotation"]})
        test_df = test_df[test_df.apply(pass_typical_threshold, axis=1)]
        return (
            train_df[:int(.8 * len(train_df))],
            train_df[int(.8 * len(train_df)):],
            test_df
        )
    elif args.dataset == "aiqgen":
        # 8-sequence sparse dataset; simple 70/15/15 split, no cross-validation
        df = pd.read_csv(get_annotated_data_filename(args),
                         converters={col: literal_eval for col in ["dialogue", "meta_data", "annotation"]})
        df = df.sample(frac=1, random_state=221)
        n = len(df)
        return df[:int(.7 * n)], df[int(.7 * n):int(.85 * n)], df[int(.85 * n):]
    raise Exception(f"Loading not supported for {args.dataset}")

def get_model_file_suffix(args, fold = None):
    suffix = "_incfirst" if args.inc_first_label else ""
    suffix += f"_{fold}" if fold else ""
    if args.model_name:
        return args.model_name + suffix
    if args.model_type == "lmkt": # Zero-shot LLMKT
        model_name = args.base_model.replace("/", "-")
    else:
        model_name = args.model_type
    return f"{args.dataset}_{model_name}_agg{args.agg}" + suffix

def get_kc_result_filename(args, fold):
    return f"results/kcs_{get_model_file_suffix(args, fold)}.json"

def get_qual_result_filename(args, fold = None):
    return f"results/qual_{get_model_file_suffix(args, fold)}.csv"

def load_atc():
    with open("data/src/ATC/domain_groups.json") as file:
        domain_groups = json.load(file)

    with open("data/src/ATC/standards.jsonl") as file:
        standards = [json.loads(line) for line in file]

    for stand in standards:
        stand["description"] = stand["description"].split("\nGrade")[0] # Remove grade-level descriptions
        stand["description"] = stand["description"].replace("\n", " ") # Remove newlines for easier LM prompting

    return {
        "domain_groups": domain_groups,
        "standards": {tag["id"]: tag for tag in standards}
    }
