import argparse
import torch
import functools


def parse_al_args():
    parser = argparse.ArgumentParser("common parameters for active learning")
    parser.add_argument("--MODEL_NAME", default="facebook/contriever")
    parser.add_argument("--DATA_NAME", help="DATA_NAME is required. Try msmarco or nq")
    parser.add_argument(
        "--RESULTS_DIR", help="RESULTS_DIR is required. e.g.: msmarco_new_human"
    )
    parser.add_argument(
        "--path_to_ranking_profile_bm25",
        default="",
        help=(
            "path_to_ranking_profile_bm25 is required,"
            " which can be obtained by BM25 with k1=0.9 and b=0.4."
        ),
    )
    parser.add_argument("--STEP", help="STEP is required, starting with 0")
    parser.add_argument("--N_REPEATS", default=3, type=int)
    parser.add_argument("--REPEAT_SEED", default=42, type=int)
    parser.add_argument(
        "--path_to_splits",
        default="",
        help="find qids_split by train_data.pt or train_data_human_response.pt",
    )
    parser.add_argument("--number_of_qid_split_batch", default=4, type=int)
    parser.add_argument("--NUM_EPOCHS", default=10, type=int)
    parser.add_argument("--DRYRUN", default=0, type=int)

    args = parser.parse_args()
    print(args)

    try:
        STEP = int(args.STEP)
    except ValueError:
        STEP = args.STEP

    qids_split = []
    if len(args.path_to_splits):
        for i in range(args.number_of_qid_split_batch):
            training_data = torch.load(
                f"{args.path_to_splits}/data_iteration_{i}/training_data.pt"
            )
            qids_split.append(
                list(
                    set(training_data.keys())
                    - set(functools.reduce(list.__add__, qids_split, []))
                )
            )
        qids_split = [[str(x) for x in s] for s in qids_split]

    if len(args.path_to_ranking_profile_bm25):
        ranking_profile_bm25 = torch.load(args.path_to_ranking_profile_bm25)
    else:
        ranking_profile_bm25 = None

    return (
        args.MODEL_NAME,
        args.DATA_NAME,
        args.RESULTS_DIR,
        STEP,
        ranking_profile_bm25,
        qids_split,
        args.N_REPEATS,
        args.REPEAT_SEED,
        args.number_of_qid_split_batch,
        args.NUM_EPOCHS,
        args.DRYRUN,
    )
