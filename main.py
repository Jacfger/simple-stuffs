import argparse
import collections
import os
from pprint import pprint
import pandas as pd
import torch
from tqdm.std import trange, tqdm

from fol.appfoq import compute_final_loss
from data_helper import TaskManager, BenchmarkTaskManager, all_normal_form
from fol import BetaEstimator, BoxEstimator, LogicEstimator, NLKEstimator, BetaEstimator4V, order_bounds
from utils.util import (Writer, load_data_with_indexing, load_task_manager, read_from_yaml,
                        set_global_seed)

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config/default.yaml', type=str)
parser.add_argument('--prefix', default='dev', type=str)
parser.add_argument('--checkpoint_path', default=None, type=str)
parser.add_argument('--load_step', default=0, type=int)


def train_step(model, opt, iterator):
    opt.zero_grad()
    data = next(iterator)
    emb_list, answer_list = [], []
    union_emb_list, union_answer_list = [], []
    for key in data:
        if 'u' in key or 'U' in key:  # TODO: consider 'evaluate_union' in the future
            union_emb_list.append(data[key]['emb'])
            union_answer_list.append(data[key]['answer_set'])
        else:
            emb_list.append(data[key]['emb'])
            answer_list.extend(data[key]['answer_set'])
    pred_embedding = torch.cat(emb_list, dim=0)
    all_positive_logit, all_negative_logit, all_subsampling_weight = model.criterion(pred_embedding, answer_list)
    for i in range(len(union_emb_list)):
        union_positive_logit, union_negative_logit, union_subsampling_weight = \
            model.criterion(union_emb_list[i], union_answer_list[i], union=True)
        all_positive_logit = torch.cat([all_positive_logit, union_positive_logit], dim=0)
        all_negative_logit = torch.cat([all_negative_logit, union_negative_logit], dim=0)
        all_subsampling_weight = torch.cat([all_subsampling_weight, union_subsampling_weight], dim=0)
    positive_loss, negative_loss = compute_final_loss(all_positive_logit, all_negative_logit, all_subsampling_weight)
    loss = (positive_loss + negative_loss) / 2
    loss.backward()
    opt.step()
    log = {
        'po': positive_loss.item(),
        'ne': negative_loss.item(),
        'loss': loss.item()
    }
    if model.name == 'logic':
        entity_embedding = model.entity_embeddings.weight.data
        if model.bounded:
            model.entity_embeddings.weight.data = order_bounds(entity_embedding)
        else:
            model.entity_embeddings.weight.data = torch.clamp(entity_embedding, 0, 1)
    return log


def eval_step(model, eval_iterator, device, mode, allowed_easy_ans=False):
    logs = collections.defaultdict(lambda: collections.defaultdict(float))
    with torch.no_grad():
        for data in tqdm(eval_iterator):
            for key in data:
                pred = data[key]['emb']
                all_logit = model.compute_all_entity_logit(pred, union=('u' in key or 'U' in key))  # batch*nentity
                argsort = torch.argsort(all_logit, dim=1, descending=True)
                ranking = argsort.clone().to(torch.float)
                #  create a new torch Tensor for batch_entity_range
                if device != torch.device('cpu'):
                    ranking = ranking.scatter_(
                        1, argsort, torch.arange(model.n_entity).to(torch.float).repeat(argsort.shape[0], 1).to(
                            device))
                else:
                    ranking = ranking.scatter_(
                        1, argsort, torch.arange(model.n_entity).to(torch.float).repeat(argsort.shape[0], 1))
                # achieve the ranking of all entities
                for i in range(all_logit.shape[0]):
                    if mode == 'train':
                        easy_ans = []
                        hard_ans = data[key]['answer_set'][i]
                    else:
                        if allowed_easy_ans:
                            easy_ans = []
                            hard_ans = list(set(data[key]['hard_answer_set'][i]).union
                                            (set(data[key]['easy_answer_set'][i])))
                        else:
                            easy_ans = data[key]['easy_answer_set'][i]
                            hard_ans = data[key]['hard_answer_set'][i]

                    num_hard = len(hard_ans)
                    num_easy = len(easy_ans)
                    assert len(set(hard_ans).intersection(set(easy_ans))) == 0
                    # only take those answers' rank
                    cur_ranking = ranking[i, list(easy_ans) + list(hard_ans)]
                    cur_ranking, indices = torch.sort(cur_ranking)
                    masks = indices >= num_easy
                    if device != torch.device('cpu'):
                        answer_list = torch.arange(
                            num_hard + num_easy).to(torch.float).to(device)
                    else:
                        answer_list = torch.arange(
                            num_hard + num_easy).to(torch.float)

                    cur_ranking = cur_ranking - answer_list + 1
                    # filtered setting: +1 for start at 0, -answer_list for ignore other answers

                    cur_ranking = cur_ranking[masks]
                    # only take indices that belong to the hard answers
                    mrr = torch.mean(1. / cur_ranking).item()
                    h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                    h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                    h10 = torch.mean(
                        (cur_ranking <= 10).to(torch.float)).item()
                    add_hard_list = torch.arange(num_hard).to(torch.float).to(device)
                    hard_ranking = cur_ranking + add_hard_list  # for all hard answer, consider other hard answer
                    logs[key]['retrieval_accuracy'] += torch.mean(
                        (hard_ranking <= num_hard).to(torch.float)).item()
                    logs[key]['MRR'] += mrr
                    logs[key]['HITS1'] += h1
                    logs[key]['HITS3'] += h3
                    logs[key]['HITS10'] += h10
                num_query = all_logit.shape[0]
                logs[key]['num_queries'] += num_query
        for key in logs.keys():
            for metric in logs[key].keys():
                if metric != 'num_queries':
                    logs[key][metric] /= logs[key]['num_queries']
    # torch.cuda.empty_cache()
    return logs


# def training(model, opt, train_iterator, valid_iterator, test_iterator, writer, **train_cfg):
#     lr = train_cfg['learning_rate']
#     with tqdm.trange(train_cfg['steps']) as t:
#         for step in t:
#             log = train_step(model, opt, train_iterator, writer)
#             t.set_postfix({'loss': log['loss']})
#             if step % train_cfg['evaluate_every_steps'] and step > 0:
#                 eval_step(model, valid_iterator, 'valid', writer, **train_cfg)
#                 eval_step(model, test_iterator, 'test', writer, **train_cfg)

#             if step >= train_cfg['warm_up_steps']:
#                 lr /= 5
#                 # logging
#                 opt = torch.optim.Adam(
#                     filter(lambda p: p.requires_grad, model.parameters()),
#                     lr=lr
#                 )
#                 train_cfg['warm_up_steps'] *= 1.5
#             if step % train_cfg['save_every_steps']:
#                 pass
#             if step % train_cfg['log_every_steps']:
#                 pass

def save_eval(log, mode, step, writer):
    for t in log:
        logt = log[t]
        logt['step'] = step
        writer.append_trace(f'eval_{mode}_{t}', logt)


def save_benchmark(log, writer, taskmanger: BenchmarkTaskManager):
    form_log = collections.defaultdict(lambda: collections.defaultdict(float))
    for normal_form in all_normal_form:
        formula = taskmanger.form2formula[normal_form]
        if formula in log:
            form_log[normal_form] = log[formula]
    writer.save_dataframe(form_log, f'eval_{taskmanger.type_str}.csv')


def load_beta_model(checkpoint_path, model, optimizer):
    print('Loading checkpoint %s...' % checkpoint_path)
    checkpoint = torch.load(os.path.join(
        args.checkpoint_path, 'checkpoint'))
    init_step = checkpoint['step']
    model.load_state_dict(checkpoint['model_state_dict'])
    current_learning_rate = checkpoint['current_learning_rate']
    warm_up_steps = checkpoint['warm_up_steps']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return current_learning_rate, warm_up_steps, init_step


def load_model(step, checkpoint_path, model, opt):
    print('Loading checkpoint %s...' % checkpoint_path)
    checkpoint = torch.load(os.path.join(
        checkpoint_path, f'{step}.ckpt'))
    model.load_state_dict(checkpoint['model_parameter'])
    opt.load_state_dict(checkpoint['optimizer_parameter'])
    learning_rate = checkpoint['learning_rate']
    warm_up_steps = checkpoint['warm_up_steps']
    return learning_rate, warm_up_steps


if __name__ == "__main__":

    args = parser.parse_args()
    # parse args and load config
    # configure = read_from_yaml('config/default.yaml')
    configure = read_from_yaml(args.config)
    print("[main] config loaded")
    pprint(configure)
    # initialize my log writer
    if configure['data']['type'] == 'beta':
        case_name = f'{args.prefix}/{args.config.split("/")[-1].split(".")[0]}'
        # case_name = 'dev/default'
        writer = Writer(case_name=case_name, config=configure, log_path='log')
        # writer = SummaryWriter('./logs-debug/unused-tb')
    else:
        case_name = f'{args.prefix}/{args.checkpoint_path.split("/")[-1]}'
        writer = Writer(case_name=case_name, config=configure, log_path='benchmark_log')


    # initialize environments
    set_global_seed(configure.get('seed', 0))
    if configure.get('cuda', -1) >= 0 and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(configure['cuda']))
        # logging.info('Device use cuda: %s' % configure['cuda'])
    else:
        device = torch.device('cpu')

    # prepare the procedure configs
    train_config = configure['train']
    train_config['device'] = device
    eval_config = configure['evaluate']
    eval_config['device'] = device

    # load the data
    print("[main] loading the data")
    data_folder = configure['data']['data_folder']
    entity_dict, relation_dict, projection_train, reverse_projection_train, projection_valid, reverse_projection_valid, \
        projection_test, reverse_projection_test = load_data_with_indexing(data_folder)
    n_entity, n_relation = len(entity_dict), len(relation_dict)

    # get model
    model_name = configure['estimator']['embedding']
    model_params = configure['estimator'][model_name]
    model_params['n_entity'], model_params['n_relation'] = n_entity, n_relation
    model_params['negative_sample_size'] = train_config['negative_sample_size']
    model_params['device'] = device
    if model_name == 'beta':
        model = BetaEstimator4V(**model_params)
        allowed_norm = ['DeMorgan', 'DNF+MultiIU']
    elif model_name == 'box':
        model = BoxEstimator(**model_params)
        allowed_norm = ['DNF+MultiIU']
    elif model_name == 'logic':
        model = LogicEstimator(**model_params)
        allowed_norm = ['DeMorgan', 'DNF+MultiIU']
    elif model_name == 'NewLook':
        model = NLKEstimator(**model_params)
        model.setup_relation_tensor(projection_train)
        allowed_norm = ['DNF+MultiIUD']
    elif model_name == 'CQD':
        pass
    else:
        assert False, 'Not valid model name!'
    model.to(device)
    if configure['data']['type'] == 'beta':
        if 'train' in configure['action']:
            print("[main] load training data")
            beta_path_tasks, beta_other_tasks = [], []
            for task in train_config['meta_queries']:
                if task in ['1p', '2p', '3p']:
                    beta_path_tasks.append(task)
                else:
                    beta_other_tasks.append(task)

            path_tasks = load_task_manager(
                configure['data']['data_folder'], 'train', task_names=beta_path_tasks)
            other_tasks = load_task_manager(
                configure['data']['data_folder'], 'train', task_names=beta_other_tasks)
            if len(beta_path_tasks) > 0:
                train_path_tm = TaskManager('train', path_tasks, device)
                train_path_iterator = train_path_tm.build_iterators(model, batch_size=train_config['batch_size'])
            else:
                train_path_tm, train_path_iterator = None, None
            if len(beta_other_tasks) > 0:
                train_other_tm = TaskManager('train', other_tasks, device)
                train_other_iterator = train_other_tm.build_iterators(model, batch_size=train_config['batch_size'])
            else:
                train_other_tm, train_other_iterator = None, None
            all_tasks = load_task_manager(
                configure['data']['data_folder'], 'train', task_names=train_config['meta_queries'])
            train_tm = TaskManager('train', all_tasks, device)
            train_iterator = train_tm.build_iterators(model, batch_size=configure['evaluate']['batch_size'])
        else:
            train_path_iterator = None
            train_other_iterator = None
            train_iterator = None
            train_tm, train_path_tm, train_other_tm = None, None, None

        if 'valid' in configure['action']:
            print("[main] load valid data")
            tasks = load_task_manager(configure['data']['data_folder'], 'valid',
                                      task_names=configure['evaluate']['meta_queries'])
            valid_tm = TaskManager('valid', tasks, device)
            valid_iterator = valid_tm.build_iterators(model, batch_size=configure['evaluate']['batch_size'])
        else:
            valid_iterator = None
            valid_tm = None

        if 'test' in configure['action']:
            print("[main] load test data")
            tasks = load_task_manager(configure['data']['data_folder'], 'test',
                                      task_names=configure['evaluate']['meta_queries'])
            test_tm = TaskManager('test', tasks, device)
            test_iterator = test_tm.build_iterators(model, batch_size=configure['evaluate']['batch_size'])
        else:
            test_iterator = None
            test_tm = None
    elif configure['data']['type'] == 'benchmark':
        test_tm_list = []
        if 'test' in configure['action']:
            formula_id_file = configure['evaluate']['formula_id_file']
            formula_id_data = pd.read_csv(formula_id_file)
            query_id_str_list = formula_id_data['formula_id']
            for type_str in query_id_str_list:
                filename = os.path.join(data_folder, f'data-{type_str}.csv')
                test_tm = BenchmarkTaskManager(formula_id_data, data_folder, type_str, device, model)
                test_iterator = test_tm.build_iterators(model, batch_size=configure['evaluate']['batch_size'])
                test_tm_list.append(test_tm)
            train_path_iterator = None
    else:
        assert False, 'Not valid data type!'

    lr = train_config['learning_rate']
    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    init_step = 1
    # exit()
    assert 2 * train_config['warm_up_steps'] == train_config['steps']
    if args.checkpoint_path is not None:
        if args.load_step != 0:
            lr, train_config['warm_up_steps'] = load_model(args.load_step, args.checkpoint_path, model, opt)
            init_step = args.load_step
        else:
            lr, train_config['warm_up_steps'], init_step = load_beta_model(args.checkpoint_path, model, opt)

    training_logs = []
    with trange(init_step, train_config['steps'] + 1) as t:
        for step in t:
            # basic training step
            if train_path_iterator:
                if step >= train_config['warm_up_steps']:
                    lr /= 5
                    # logging
                    opt = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        lr=lr
                    )
                    train_config['warm_up_steps'] *= 1.5
                try:
                    _log = train_step(model, opt, train_path_iterator)
                except StopIteration:
                    print("new epoch for path meta-query")
                    train_path_iterator = train_path_tm.build_iterators(model, batch_size=train_config['batch_size'])
                    _log = train_step(model, opt, train_path_iterator)
                if train_other_iterator:
                    try:
                        _log_other = train_step(model, opt, train_other_iterator)
                        try:
                            _log_second = train_step(model, opt, train_path_iterator)
                        except StopIteration:
                            print("new epoch for path meta-query")
                            train_path_iterator = train_path_tm.build_iterators(model,
                                                                                batch_size=train_config['batch_size'])
                            _log_second = train_step(model, opt, train_path_iterator)
                    except StopIteration:
                        print("new epoch for other meta-query")
                        train_other_iterator = \
                            train_other_tm.build_iterators(model, batch_size=train_config['batch_size'])
                        _log_other = train_step(model, opt, train_other_iterator)
                        try:
                            _log_second = train_step(model, opt, train_path_iterator)
                        except StopIteration:
                            print("new epoch for path meta-query")
                            train_path_iterator = train_path_tm.build_iterators(model,
                                                                                batch_size=train_config['batch_size'])
                            _log_second = train_step(model, opt, train_path_iterator)
                    _alllog = {}
                    for key in _log:
                        _alllog[f'all_{key}'] = (_log[key] + _log_other[key]) / 2
                        _alllog[key] = _log[key]
                    _log = _alllog
                t.set_postfix(_log)
                training_logs.append(_log)
                if step % train_config['log_every_steps'] == 0:
                    for metric in training_logs[0].keys():
                        _log[metric] = sum(log[metric] for log in training_logs) / len(training_logs)
                    training_logs = []
                    writer.append_trace('train', _log)

            if step % train_config['evaluate_every_steps'] == 0 or step == train_config['steps']:
                if configure['data']['type'] == 'beta':
                    '''
                    if train_iterator:
                        train_iterator = train_tm.build_iterators(model, batch_size=configure['evaluate']['batch_size'])
                        _log = eval_step(model, train_iterator, device, mode='train')
                        save_eval(_log, 'train', step, writer)
                    '''

                    if valid_iterator:
                        valid_iterator = valid_tm.build_iterators(model, batch_size=configure['evaluate']['batch_size'])
                        _log = eval_step(model, valid_iterator, device, mode='valid')
                        save_eval(_log, 'valid', step, writer)

                    if test_iterator:
                        test_iterator = test_tm.build_iterators(model, batch_size=configure['evaluate']['batch_size'])
                        _log = eval_step(model, test_iterator, device, mode='test')
                        save_eval(_log, 'test', step, writer)
                elif configure['data']['type'] == 'benchmark':
                    for test_tm in test_tm_list:
                        test_iterator = test_tm.build_iterators(model, batch_size=configure['evaluate']['batch_size'])
                        _log = eval_step(model, test_iterator, device, mode='test')
                        '''
                        test_iterator = test_tm.build_iterators(model, batch_size=configure['evaluate']['batch_size'])
                        _log_easy = eval_step(model, test_iterator, device, mode='test', allowed_easy_ans=True)
                        for formula in _log_easy:
                            for metrics in _log_easy[formula]:
                                _log[formula][f'easy_{metrics}'] = _log_easy[formula][metrics]
                        '''
                        save_benchmark(_log, writer, test_tm)

            if step % train_config['save_every_steps'] == 0 and train_path_iterator:
                writer.save_model(model, opt, step, train_config['warm_up_steps'], lr)
