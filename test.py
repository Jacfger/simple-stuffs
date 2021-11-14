import torch
import argparse
import os
import pandas as pd
from tqdm import tqdm

import glob
from data_helper import TaskManager, BenchmarkTaskManager, all_normal_form
from typing import List

from fol import (
    BetaEstimator4V,
    order_bounds,
)
from utils.util import (
    Writer,
    load_data_with_indexing,
    load_task_manager,
    read_from_yaml,
    set_global_seed,
)

from collections import defaultdict


def eval_step(model, eval_iterator, device, mode, allowed_easy_ans=False):
    logs = defaultdict(lambda: defaultdict(float))
    with torch.no_grad():
        for data in tqdm(eval_iterator):
            for key in data:
                pred = data[key]["emb"]
                all_logit = model.compute_all_entity_logit(
                    pred, union=("u" in key or "U" in key)
                )  # batch*nentity
                argsort = torch.argsort(all_logit, dim=1, descending=True)
                ranking = argsort.clone().to(torch.float)
                #  create a new torch Tensor for batch_entity_range
                if device != torch.device("cpu"):
                    ranking = ranking.scatter_(
                        1,
                        argsort,
                        torch.arange(model.n_entity)
                        .to(torch.float)
                        .repeat(argsort.shape[0], 1)
                        .to(device),
                    )
                else:
                    ranking = ranking.scatter_(
                        1,
                        argsort,
                        torch.arange(model.n_entity)
                        .to(torch.float)
                        .repeat(argsort.shape[0], 1),
                    )
                # achieve the ranking of all entities
                for i in range(all_logit.shape[0]):
                    if mode == "train":
                        easy_ans = []
                        hard_ans = data[key]["answer_set"][i]
                    else:
                        if allowed_easy_ans:
                            easy_ans = []
                            hard_ans = list(
                                set(data[key]["hard_answer_set"][i]).union(
                                    set(data[key]["easy_answer_set"][i])
                                )
                            )
                        else:
                            easy_ans = data[key]["easy_answer_set"][i]
                            hard_ans = data[key]["hard_answer_set"][i]

                    num_hard = len(hard_ans)
                    num_easy = len(easy_ans)
                    assert len(set(hard_ans).intersection(set(easy_ans))) == 0
                    # only take those answers' rank
                    cur_ranking = ranking[i, list(easy_ans) + list(hard_ans)]
                    cur_ranking, indices = torch.sort(cur_ranking)
                    masks = indices >= num_easy
                    if device != torch.device("cpu"):
                        answer_list = (
                            torch.arange(num_hard + num_easy).to(torch.float).to(device)
                        )
                    else:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float)

                    cur_ranking = cur_ranking - answer_list + 1
                    # filtered setting: +1 for start at 0, -answer_list for ignore other answers

                    cur_ranking = cur_ranking[masks]
                    # only take indices that belong to the hard answers
                    mrr = torch.mean(1.0 / cur_ranking).item()
                    h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                    h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                    h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()
                    add_hard_list = torch.arange(num_hard).to(torch.float).to(device)
                    hard_ranking = (
                        cur_ranking + add_hard_list
                    )  # for all hard answer, consider other hard answer
                    logs[key]["retrieval_accuracy"] += torch.mean(
                        (hard_ranking <= num_hard).to(torch.float)
                    ).item()
                    logs[key]["MRR"] += mrr
                    logs[key]["HITS1"] += h1
                    logs[key]["HITS3"] += h3
                    logs[key]["HITS10"] += h10
                num_query = all_logit.shape[0]
                logs[key]["num_queries"] += num_query
        for key in logs.keys():
            for metric in logs[key].keys():
                if metric != "num_queries":
                    logs[key][metric] /= logs[key]["num_queries"]
            # print(key, logs[key]['MRR'])
    # torch.cuda.empty_cache()

    return logs


def save_benchmark(log, writer, taskmanger: BenchmarkTaskManager):
    form_log = defaultdict(lambda: defaultdict(float))
    for normal_form in all_normal_form:
        formula = taskmanger.form2formula[normal_form]
        if formula in log:
            form_log[normal_form] = log[formula]
            print(normal_form, log[formula]['MRR'])
    # writer.save_dataframe(form_log, f"eval_{taskmanger.type_str}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/benchmark_BetaE.yaml", type=str)
    parser.add_argument("--prefix", default="dev", type=str)
    parser.add_argument("--checkpoint_path", default=None, type=str)
    parser.add_argument("--load_step", default=0, type=int)
    parser.add_argument("--test", default=None, type=str)
    args = parser.parse_args()

    assert args.test, 'enter test file'
    """
        Load config
    """
    configure = read_from_yaml(args.config)
    print("loaded configuration")
    """
        Load model
    """

    if configure.get("cuda", -1) >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(configure["cuda"]))
        # logging.info('Device use cuda: %s' % configure['cuda'])
    else:
        device = torch.device("cpu")

    print(device)
    data_folder = configure["data"]["data_folder"]
    (
        entity_dict,
        relation_dict,
        projection_train,
        reverse_projection_train,
        projection_valid,
        reverse_projection_valid,
        projection_test,
        reverse_projection_test,
    ) = load_data_with_indexing(data_folder)

    print("loaded data")
    n_entity, n_relation = len(entity_dict), len(relation_dict)

    case_name = f'dev/{args.checkpoint_path.split("/")[-1]}'
    writer = Writer(case_name=case_name, config=configure, log_path="benchmark_log")

    model_name = configure["estimator"]["embedding"]
    model_params = configure["estimator"][model_name]
    model_params["n_entity"], model_params["n_relation"] = n_entity, n_relation
    model_params["negative_sample_size"] = configure["train"]["negative_sample_size"]
    model_params["device"] = device
    if model_name == "beta":
        model = BetaEstimator4V(**model_params)
        allowed_norm = ["DeMorgan", "DNF+MultiIU"]
    else:
        assert False, "Not valid model name!"
    model.to(device)

    print("loaded model")
    if args.checkpoint_path:
        torch.load(args.checkpoint_path, map_location="cpu")
    """
        Load data
    """

    test_tm_list = []
    # query_files = glob.glob(os.path.join(configure["data"]["data_folder"], "*.csv"))
    query_files = [args.test] # Training code have to be used
    print(f"Queries files: {query_files}")
    assert query_files, "Probably wrong dataset folder"
    for query_file in query_files:
        formula_id_data = pd.read_csv(configure["evaluate"]["formula_id_file"])
        type_str = os.path.splitext(os.path.split(query_file)[1])[0].split("-")[1]
        test_tm = BenchmarkTaskManager(
            formula_id_data, data_folder, type_str, device, model
        )
        test_iterator = test_tm.build_iterators(
            model, batch_size=configure["evaluate"]["batch_size"]
        )
        test_tm_list.append(test_tm)
    # print(configure)
    print("data processed. Start inferencing")
    # print(test_tm_list)
    for test_tm in test_tm_list:
        test_iterator = test_tm.build_iterators(
            model, batch_size=configure["evaluate"]["batch_size"]
        )
        _log = eval_step(model, test_iterator, device, mode="test")
        """
        test_iterator = test_tm.build_iterators(model, batch_size=configure['evaluate']['batch_size'])
        _log_easy = eval_step(model, test_iterator, device, mode='test', allowed_easy_ans=True)
        for formula in _log_easy:
            for metrics in _log_easy[formula]:
                _log[formula][f'easy_{metrics}'] = _log_easy[formula][metrics]
        """
        save_benchmark(_log, writer, test_tm)

    """
        MRR
    """
