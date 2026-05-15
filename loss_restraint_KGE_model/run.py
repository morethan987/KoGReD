import torch
from torch.utils.data import DataLoader
import argparse
import numpy as np
from ordered_set import OrderedSet
from collections import defaultdict as ddict
from pprint import pprint
import os
import json
from helper import *
from data_loader import *
from models import *
from metrics_collector import MetricsCollector


class Runner(object):
    def __init__(self, params):
        """
        Constructor of the runner class

        Parameters
        ----------
        params: List of hyper-parameters of the model

        Returns
        -------
        Creates computational graph and optimizer
        """
        self.p = params
        self.entity_mrr_totals = {}
        self.entity_count = {}
        self.entity_mrr_average = {}
        self.logger = get_logger(self.p.name)

        self.logger.info(vars(self.p))
        pprint(vars(self.p))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        torch.cuda.manual_seed(self.p.seed)

        self.logger.info(f"Using device: {self.device}")

        self.load_data()
        self.model = self.add_model(self.p.model, self.p.score_func)
        self.optimizer = self.add_optimizer(self.model.parameters())

    def load_data(self):
        """
        Reading in raw triples and converts it into a standard format.

        Parameters
        ----------
        self.p.dataset:         Takes in the name of the dataset (FB15k-237)

        Returns
        -------
        self.ent2id:            Entity to unique identifier mapping
        self.id2rel:            Inverse mapping of self.rel2id
        self.rel2id:            Relation to unique identifier mapping
        self.num_ent:           Number of entities in the Knowledge graph
        self.num_rel:           Number of relations in the Knowledge graph
        self.embed_dim:         Embedding dimension used
        self.data['train']:     Stores the triples corresponding to training dataset
        self.data['valid']:     Stores the triples corresponding to validation dataset
        self.data['test']:      Stores the triples corresponding to test dataset
        self.data_iter:			The dataloader for different data splits

        """

        ent_set, rel_set = OrderedSet(), OrderedSet()
        for split in ['train', 'test', 'valid']:
            for line in open(f'data/{self.p.dataset}/{split}.txt'):
                sub, rel, obj = map(str.lower, line.strip().split('\t'))
                ent_set.add(sub)
                rel_set.add(rel)
                ent_set.add(obj)

        self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
        self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
        self.rel2id.update({rel+'_reverse': idx+len(self.rel2id)
                        for idx, rel in enumerate(rel_set)})
        self.save_id_config()

        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

        self.p.num_ent = len(self.ent2id)
        self.p.num_rel = len(self.rel2id) // 2
        self.p.embed_dim = self.p.k_w * \
            self.p.k_h if self.p.embed_dim is None else self.p.embed_dim

        self.data = ddict(list)
        sr2o = ddict(set)
        sr2observed = ddict(set)

        for split in ['train', 'test', 'valid']:
            if split == "train":
                for line in open(f'data/{self.p.dataset}/{split}.txt'):
                    sub, rel, obj = map(str.lower, line.strip().split('\t'))
                    sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
                    self.data[split].append((sub, rel, obj))

                    sr2o[(sub, rel)].add(obj)
                    sr2o[(obj, rel+self.p.num_rel)].add(sub)
                    if self.p.loss_delta > 0:
                        sr2observed[(sub, rel)].add(obj)
                        sr2observed[(obj, rel+self.p.num_rel)].add(sub)

                aux_path = self.p.aux_triples
                if self.p.loss_delta > 0 or self.p.keep_aux:
                    if not os.path.isfile(aux_path):
                        self.logger.error(f'Auxiliary triples file not found: {aux_path}')
                        sys.exit(1)
                    aux_cnt = 0
                    self.logger.info(f'Loading auxiliary triples from: {aux_path}')
                    for line in open(aux_path):
                        sub, rel, obj = map(
                            str.lower, line.strip().split('\t'))
                        try:
                            sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
                            self.data[split].append((sub, rel, obj))
                            sr2o[(sub, rel)].add(obj)
                            sr2o[(obj, rel+self.p.num_rel)].add(sub)
                            aux_cnt += 1
                        except:
                            continue
                    self.logger.info(f'Number of auxiliary triples added: {aux_cnt}')
            else:
                for line in open(f'data/{self.p.dataset}/{split}.txt'):
                    sub, rel, obj = map(str.lower, line.strip().split('\t'))
                    sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
                    self.data[split].append((sub, rel, obj))

        self.aux_confidence_map = self._load_confidence_map()
        self.aux_triple_set = self._build_aux_triple_set()

        self.data = dict(self.data)

        self.sr2o = {k: list(v) for k, v in sr2o.items()}

        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                sr2o[(sub, rel)].add(obj)
                sr2o[(obj, rel+self.p.num_rel)].add(sub)
        if self.p.loss_delta > 0:
            self.sr2observed = {k: list(v) for k, v in sr2observed.items()}
            keys = set(self.sr2o.keys()) - set(self.sr2observed.keys())

            for key in keys:
                self.sr2observed[key] = []

            sr2newadd = ddict(set)
            total_keys = self.sr2o.keys()
            for key in total_keys:
                newadd_label = set(self.sr2o[key]) - set(self.sr2observed[key])
                sr2newadd[key] = newadd_label
            self.sr2newadd = {k: list(v) for k, v in sr2newadd.items()}

            self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
            self.triples = ddict(list)

            for (sub, rel), obj in self.sr2o.items():
                self.triples['train'].append({
                    'triple': (sub, rel, -1),
                    'label': self.sr2o[(sub, rel)],
                    'observed_label': self.sr2observed[(sub, rel)],
                    'newadd_label': self.sr2newadd[(sub, rel)], 'sub_samp': 1
                })
        else:
            self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
            self.triples = ddict(list)

            for (sub, rel), obj in self.sr2o.items():
                self.triples['train'].append({
                    'triple': (sub, rel, -1),
                    'label': self.sr2o[(sub, rel)],
                    'sub_samp': 1
                })

        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                rel_inv = rel + self.p.num_rel
                self.triples['{}_{}'.format(split, 'tail')].append({
                    'triple': (sub, rel, obj),
                    'label': self.sr2o_all[(sub, rel)]
                })
                self.triples['{}_{}'.format(split, 'head')].append({
                    'triple': (obj, rel_inv, sub),
                    'label': self.sr2o_all[(obj, rel_inv)]
                })

        self.triples = dict(self.triples)

        self.data_iter = {
            'train':    	self.get_data_loader(TrainDataset, 'train', 	    self.p.batch_size) if (self.p.loss_delta < 0) else self.get_data_loader(TrainDataset_addLoss, 'train', self.p.batch_size),
            'valid_head':   self.get_data_loader(TestDataset,  'valid_head', self.p.batch_size),
            'valid_tail':   self.get_data_loader(TestDataset,  'valid_tail', self.p.batch_size),
            'test_head':   	self.get_data_loader(TestDataset,  'test_head',  self.p.batch_size),
            'test_tail':   	self.get_data_loader(TestDataset,  'test_tail',  self.p.batch_size),
        }

        self.edge_index, self.edge_type = self.construct_adj()

    def get_data_loader(self, dataset_class, split, batch_size, shuffle=True):
        num_workers = max(0, self.p.num_workers)
        return DataLoader(
            dataset_class(self.triples[split], self.p),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=dataset_class.collate_fn
        )

    def _load_confidence_map(self):
        """加载辅助三元组置信度映射，返回 {(h_id, r_id, t_id): confidence}"""
        conf_path = self.p.aux_confidence
        if not conf_path or not os.path.isfile(conf_path):
            self.logger.info('Confidence map not provided or not found, metrics collection disabled')
            return {}
        self.logger.info(f'Loading confidence map from: {conf_path}')
        with open(conf_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        conf_map = {}
        for key_str, conf in raw.items():
            parts = key_str.strip().split('\t')
            if len(parts) == 3:
                h, r, t = parts[0].lower(), parts[1].lower(), parts[2].lower()
                h_id = self.ent2id.get(h)
                r_id = self.rel2id.get(r)
                t_id = self.ent2id.get(t)
                if h_id is not None and r_id is not None and t_id is not None:
                    conf_map[(h_id, r_id, t_id)] = conf
                    r_inv_id = r_id + self.p.num_rel
                    conf_map[(t_id, r_inv_id, h_id)] = conf
        self.logger.info(f'Loaded confidence map with {len(conf_map)} entries')
        return conf_map

    def _build_aux_triple_set(self):
        """构建辅助三元组 ID 集合，包含原始边和反向边"""
        aux_set = set()
        aux_path = self.p.aux_triples
        if not os.path.isfile(aux_path):
            return aux_set
        for line in open(aux_path):
            parts = line.strip().split('\t')
            if len(parts) == 3:
                sub, rel, obj = parts[0].lower(), parts[1].lower(), parts[2].lower()
                s_id = self.ent2id.get(sub)
                r_id = self.rel2id.get(rel)
                o_id = self.ent2id.get(obj)
                if s_id is not None and r_id is not None and o_id is not None:
                    aux_set.add((s_id, r_id, o_id))
                    r_inv_id = r_id + self.p.num_rel
                    aux_set.add((o_id, r_inv_id, s_id))
        self.logger.info(f'Built aux triple set with {len(aux_set)} entries')
        return aux_set

    def construct_adj(self):
        """
        Constructor of the adjacency matrix

        Parameters
        ----------

        Returns
        -------
        Constructs the adjacency matrix for GCN
        """
        edge_index, edge_type = [], []

        for sub, rel, obj in self.data['train']:
            edge_index.append((sub, obj))
            edge_type.append(rel)

        # Adding inverse edges
        for sub, rel, obj in self.data['train']:
            edge_index.append((obj, sub))
            edge_type.append(rel + self.p.num_rel)

        edge_index = torch.LongTensor(edge_index).to(self.device).t()
        edge_type = torch.LongTensor(edge_type). to(self.device)

        return edge_index, edge_type

    def add_model(self, model, score_func):
        """
        Creates the computational graph

        Parameters
        ----------
        model_name:     Contains the model name to be created

        Returns
        -------
        Creates the computational graph for model and initializes it

        """
        model_name = f'{model}_{score_func}'

        if model_name.lower() == 'compgcn_transe':
            model = CompGCN_TransE(
                self.edge_index, self.edge_type, params=self.p)
        elif model_name.lower() == 'compgcn_distmult':
            model = CompGCN_DistMult(
                self.edge_index, self.edge_type, params=self.p)
        elif model_name.lower() == 'compgcn_conve':
            model = CompGCN_ConvE(
                self.edge_index, self.edge_type, params=self.p)
        else:
            raise NotImplementedError

        model.to(self.device)

        return model

    def add_optimizer(self, parameters):
        """
        Creates an optimizer for training the parameters

        Parameters
        ----------
        parameters:         The parameters of the model

        Returns
        -------
        Returns an optimizer for learning the parameters of the model

        """
        return torch.optim.Adam(parameters, lr=self.p.lr, weight_decay=self.p.l2)

    def read_batch(self, batch, split):
        """
        Function to read a batch of data and move the tensors in batch to CPU/GPU

        Parameters
        ----------
        batch: 		the batch to process
        split: (string) If split == 'train', 'valid' or 'test' split

        Returns
        -------
        Head, Relation, Tails, labels
        """
        if split == 'train':
            if self.p.loss_delta > 0:
                triple, label, observed_label, newadd_label = [
                    _.to(self.device) for _ in batch]
                return triple[:, 0], triple[:, 1], triple[:, 2], label, observed_label, newadd_label
            else:
                triple, label = [_.to(self.device) for _ in batch]
                return triple[:, 0], triple[:, 1], triple[:, 2], label
        else:
            triple, label = [_.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label

    def save_id_config(self):
        """将id配置保存到txt文件中"""
        os.makedirs(self.p.save_dir, exist_ok=True)
        with open(f'{self.p.save_dir}/entity2id.txt', 'w') as f:
            for ent, idx in self.ent2id.items():
                f.write(f'{ent}\t{idx}\n')
        with open(f'{self.p.save_dir}/relation2id.txt', 'w') as f:
            for rel, idx in self.rel2id.items():
                f.write(f'{rel}\t{idx}\n')

    def save_model(self, save_path):
        """
        Function to save a model. It saves the model parameters, best validation scores,
        best epoch corresponding to best validation, state of the optimizer and all arguments for the run.

        Parameters
        ----------
        save_path: path where the model is saved

        Returns
        -------
        """
        state = {
            'state_dict': self.model.state_dict(),
            'best_val': self.best_val,
            'best_epoch': self.best_epoch,
            'optimizer': self.optimizer.state_dict(),
            'args': vars(self.p)
        }
        torch.save(state, save_path)

    def load_model(self, load_path):
        """
        Function to load a saved model

        Parameters
        ----------
        load_path: path to the saved model

        Returns
        -------
        """
        state = torch.load(load_path)
        # 1. 获取保存的参数 (args) 并用它们更新当前的 self.p
        saved_args = state['args']
        vars(self.p).update(saved_args)

        # 2. 根据刚刚恢复的、正确的参数重新构建模型和优化器
        # 这一步是必需的，因为在 __init__ 中可能已经用错误的参数构建了一个旧模型
        self.model = self.add_model(self.p.model, self.p.score_func)
        self.optimizer = self.add_optimizer(self.model.parameters())

        # 3. 将保存的权重和优化器状态加载到新创建的实例中
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])

        # 4. 恢复最佳验证结果等元数据
        self.best_val = state['best_val']
        self.best_val_mrr = self.best_val.get('mrr', 0) # 使用 .get() 增加兼容性

    def evaluate(self, split, epoch):
        """
        Function to evaluate the model on validation or test set

        Parameters
        ----------
        split: (string) If split == 'valid' then evaluate on the validation set, else the test set
        epoch: (int) Current epoch count

        Returns
        -------
        resutls:			The evaluation results containing the following:
            results['mr']:         	Average of ranks_left and ranks_right
            results['mrr']:         Mean Reciprocal Rank
            results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

        """
        # 清空先前的数据
        self.entity_mrr_totals = {}
        self.entity_count = {}

        left_results = self.predict(split=split, mode='tail_batch')
        right_results = self.predict(split=split, mode='head_batch')
        results = get_combined_results(left_results, right_results)
        # self.logger.info(
        #     f'[Epoch {epoch} {split}]: MRR: Tail : {results["left_mrr"]:.5}, Head : {results["right_mrr"]:.5}, Avg : {results["mrr"]:.5}')
        return results

    def _process_single_prediction(self, pred, obj, label, results):
        """
        Processes a single prediction to calculate rank and update evaluation metrics.

        Parameters
        ----------
        pred: (torch.Tensor)    The model's prediction scores for a single subject and relation.
        obj: (torch.Tensor)     The ground truth object for the single triple.
        label: (torch.Tensor)   The label tensor for the single triple.
        results: (dict)         The dictionary to update with evaluation metrics.

        Returns
        -------
        results: (dict)         The updated evaluation results.
        """
        b_range = torch.arange(pred.size()[0], device=self.device)
        target_pred = pred[b_range, obj]
        pred = torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
        pred[b_range, obj] = target_pred
        ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]
        ranks = ranks.float()

        ranks_ = torch.unsqueeze(ranks, dim=1)
        label_ = label.argmax(dim=1)
        label_ = torch.unsqueeze(label_, dim=1)

        entity_ranks = torch.cat((label_, ranks_), dim=1)

        for row in entity_ranks:
            entity_id = row[0].item()
            rrank = 1.0 / (row[1].item())

            if entity_id in self.entity_mrr_totals:
                self.entity_mrr_totals[entity_id] += rrank
                self.entity_count[entity_id] += 1
            else:
                self.entity_mrr_totals[entity_id] = rrank
                self.entity_count[entity_id] = 1

        results['count'] = torch.numel(ranks) + results.get('count', 0.0)
        results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
        results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
        for k in range(10):
            results['hits@{}'.format(k + 1)] = torch.numel(
                ranks[ranks <= (k + 1)]) + results.get('hits@{}'.format(k + 1), 0.0)

        return results

    def predict(self, split='valid', mode='tail_batch'):
        """
        Function to run model evaluation for a given mode.

        Parameters
        ----------
        split: (string)        If split == 'valid' then evaluate on the validation set, else the test set.
        mode: (string):        Can be 'head_batch' or 'tail_batch'.

        Returns
        -------
        results:            The evaluation results containing the following:
            results['mr']:         Average of ranks_left and ranks_right
            results['mrr']:        Mean Reciprocal Rank
            results['hits@k']:     Probability of getting the correct prediction in top-k ranks based on predicted score.
        """
        self.model.eval()

        with torch.no_grad():
            results = {}
            train_iter = iter(self.data_iter[f'{split}_{mode.split("_")[0]}'])

            for step, batch in enumerate(train_iter):
                sub, rel, obj, label = self.read_batch(batch, split)
                pred = self.model.forward(sub, rel)

                # The core logic for processing each prediction is now in a separate function
                results = self._process_single_prediction(pred, obj, label, results)

                if (step + 1) % 100 == 0:
                    self.logger.info('[{}, {} Step {}]\t{}'.format(
                        split.title(), mode.title(), step, self.p.name))

        return results

    def run_epoch(self, epoch, val_mrr=0, clean_rate=1, metrics_collector=None):
        """
        Function to run one epoch of training

        Parameters
        ----------
        epoch: current epoch count

        Returns
        -------
        loss: The loss value after the completion of one epoch
        """
        self.model.train()
        losses = []
        train_iter = iter(self.data_iter['train'])
        clean_rate = clean_rate

        for step, batch in enumerate(train_iter):
            self.optimizer.zero_grad()
            if self.p.loss_delta < 0:
                sub, rel, obj, label = self.read_batch(batch, 'train')
                pred = self.model.forward(sub, rel)
                loss = self.model.loss(pred, label)
            elif self.p.loss_only_new > 0:
                sub, rel, obj, label, obeserved_label, newadd_label = self.read_batch(batch, 'train')
                pred = self.model.forward(sub, rel)
                loss = self.model.modify_loss_only_add(
                    pred, label, newadd_label, clean_rate,
                    metrics_collector=metrics_collector,
                    epoch=epoch, sub_ids=sub, rel_ids=rel)
            else:
                sub, rel, obj, label, obeserved_label, newadd_label = self.read_batch(batch, 'train')
                pred = self.model.forward(sub, rel)
                loss = self.model.modify_loss(pred, label, obeserved_label, clean_rate)

            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            if step % 100 == 0:
                self.logger.info(
                    f'[E:{epoch}| {step}]: Train Loss:{np.mean(losses):.5},  Val MRR:{self.best_val_mrr:.5}\t{self.p.name}')

        loss = np.mean(losses)
        self.logger.info(f'[Epoch:{epoch}]:  Training Loss:{loss:.4}\n')
        return loss

    def fit(self):
        """
        Function to run training and evaluation of model

        Parameters
        ----------

        Returns
        -------
        """
        self.best_val_mrr, self.best_val, self.best_epoch, val_mrr = 0., {}, 0, 0.
        save_path = os.path.join(self.p.save_dir, self.p.name + '.pth')

        metrics_collector = None
        if self.aux_confidence_map and self.aux_triple_set and self.p.loss_only_new > 0:
            metrics_output_dir = os.path.join(self.p.save_dir, 'visualization_metrics')
            metrics_collector = MetricsCollector(
                aux_confidence_map=self.aux_confidence_map,
                aux_triple_set=self.aux_triple_set,
                output_dir=metrics_output_dir,
            )
            self.logger.info(f'Metrics collection enabled, output: {metrics_output_dir}')

        if self.p.restore:
            self.load_model(save_path)
            self.logger.info('Successfully Loaded previous model')

        kill_cnt = 0
        epoch = -1
        clean_rate = 1  # init

        for epoch in range(self.p.max_epochs):
            train_loss = self.run_epoch(epoch, val_mrr, clean_rate, metrics_collector=metrics_collector)

            if metrics_collector is not None:
                metrics_collector.finalize_epoch_loss(epoch)
                if hasattr(self.model, 'conv1') and hasattr(self.model.conv1, 'last_alpha'):
                    metrics_collector.record_epoch_alpha(
                        epoch,
                        self.model.conv1.last_alpha,
                        self.model.conv1.last_edge_index,
                        self.model.conv1.last_edge_type,
                    )
                if epoch % 30 == 0:
                    metrics_collector.save_interim(epoch)

            val_results = self.evaluate('valid', epoch)
            if epoch % 30 == 0:
                test_results = self.evaluate('test', epoch)
                self.logger.info('\nTest set results:')
                self.logger.info(
                    f'Epoch {epoch}: MRR: Tail : {test_results["left_mrr"]:.5}, Head : {test_results["right_mrr"]:.5}, Avg : {test_results["mrr"]:.5}')
                self.logger.info(
                    f'Epoch {epoch}: MR: Tail : {test_results["left_mr"]:.5}, Head : {test_results["right_mr"]:.5}, Avg : {test_results["mr"]:.5}')
                self.logger.info(
                    f'Epoch {epoch}: left_hits@1: Tail : {test_results["left_hits@1"]:.5}, Head : {test_results["right_hits@1"]:.5}, Avg : {test_results["hits@1"]:.5}')
                self.logger.info(
                    f'Epoch {epoch}: left_hits@3: Tail : {test_results["left_hits@3"]:.5}, Head : {test_results["right_hits@3"]:.5}, Avg : {test_results["hits@3"]:.5}')
                self.logger.info(
                    f'Epoch {epoch}: left_hits@10: Tail : {test_results["left_hits@10"]:.5}, Head : {test_results["right_hits@10"]:.5}, Avg : {test_results["hits@10"]:.5}\n')

            if val_results['mrr'] > self.best_val_mrr:
                self.best_val = val_results
                self.best_val_mrr = val_results['mrr']
                self.best_epoch = epoch
                self.save_model(save_path)
                kill_cnt = 0
            else:
                kill_cnt += 1
                if kill_cnt % 10 == 0 and self.p.gamma > 5:
                    self.p.gamma -= 5
                    self.logger.info(
                        'Gamma decay on saturation, updated value of gamma: {}'.format(self.p.gamma))
                if kill_cnt > 45:
                    self.logger.info("Early Stopping!!")
                    break

            self.logger.info(
                f'[Epoch {epoch}]: Training Loss: {train_loss:.5}, Valid MRR: {self.best_val_mrr:.5}\n\n')

            if (epoch % 30 == 0) and (self.p.loss_delta > 0):
                # update clean_rate
                clean_rate -= self.p.loss_delta

        self.logger.info('Loading best model, Evaluating on Test data')
        self.load_model(save_path)

        if metrics_collector is not None:
            metrics_path = metrics_collector.save()
            self.logger.info(f'Suppression metrics saved to {metrics_path}')

        test_results = self.evaluate('test', epoch)
        self.logger.info('\nFinal Test set results:')
        self.logger.info(
            f'Final results: MRR: Tail : {test_results["left_mrr"]:.5}, Head : {test_results["right_mrr"]:.5}, Avg : {test_results["mrr"]:.5}')
        self.logger.info(
            f'Final results: MR: Tail : {test_results["left_mr"]:.5}, Head : {test_results["right_mr"]:.5}, Avg : {test_results["mr"]:.5}')
        self.logger.info(
            f'Final results: left_hits@1: Tail : {test_results["left_hits@1"]:.5}, Head : {test_results["right_hits@1"]:.5}, Avg : {test_results["hits@1"]:.5}')
        self.logger.info(
            f'Final results: left_hits@3: Tail : {test_results["left_hits@3"]:.5}, Head : {test_results["right_hits@3"]:.5}, Avg : {test_results["hits@3"]:.5}')
        self.logger.info(
            f'Final results: left_hits@10: Tail : {test_results["left_hits@10"]:.5}, Head : {test_results["right_hits@10"]:.5}, Avg : {test_results["hits@10"]:.5}')

    def load_relation_type_data(self, relation_type):
        """
        读取特定关系类型（如 '1-1'）的三元组文件，并将其格式化存储到 self.triples 中。
        文件中的三元组(h, r, t)是字符串编码,制表符分割

        Parameters
        ----------
        relation_type:  '1-1', '1-n', 'n-1', 'n-n'
        """

        # 构造文件名
        filename = f'data/{self.p.dataset}/{relation_type}.txt'

        self.logger.info(f'Loading relation type data from: {filename}')

        triples_list = []
        try:
            with open(filename, 'r') as f:
                for line in f:
                    # 使用 int() 确保读取到的是整数
                    try:
                        sub, rel, obj = line.strip().split('\t')
                        triples_list.append((self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]))
                    except ValueError:
                        self.logger.warning(f"Skipping line with non-integer IDs in {filename}: {line.strip()}")
                        continue
        except FileNotFoundError:
            self.logger.error(f'Relation type file not found: {filename}')
            return

        split_prefix = 'test_' + relation_type # 使用 test_ 作为前缀
        tail_key = f'{split_prefix}_tail'
        head_key = f'{split_prefix}_head'

        if tail_key not in self.triples:
            self.triples[tail_key] = []
        if head_key not in self.triples:
            self.triples[head_key] = []

        # 格式化数据并存入 self.triples
        for sub, rel, obj in triples_list:
            rel_inv = rel + self.p.num_rel

            # tail prediction: (sub, rel, ?) -> obj
            # label 仍然使用 self.sr2o_all 来获取所有正确的答案
            if (sub, rel) in self.sr2o_all:
                self.triples[f'{split_prefix}_tail'].append(
                    {'triple': (sub, rel, obj), 'label': self.sr2o_all[(sub, rel)]})
            else:
                self.logger.warning(f"Triple ({sub}, {rel}, {obj}) not found in sr2o_all for tail prediction. Skipping.")

            # head prediction: (?, rel_inv, obj) -> sub
            if (obj, rel_inv) in self.sr2o_all:
                self.triples[f'{split_prefix}_head'].append(
                    {'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})
            else:
                self.logger.warning(f"Triple ({obj}, {rel_inv}, {sub}) not found in sr2o_all for head prediction. Skipping.")

        self.logger.info(f'Loaded {len(triples_list)} base triples and generated {len(self.triples.get(f"{split_prefix}_tail", []))} tail evaluation triples for {relation_type}')

    def test_overall(self):
        self.best_val_mrr, self.best_val, self.best_epoch, val_mrr = 0., {}, 0, 0.
        save_path = os.path.join(self.p.save_dir, self.p.name + '.pth')

        if self.p.restore:
            self.load_model(save_path)
            self.logger.info('Successfully Loaded previous model')

        test_results = self.evaluate('test', -1)
        self.logger.info('\nFinal Test set results:')
        self.logger.info(
            f'Final results: MRR: Tail : {test_results["left_mrr"]:.5}, Head : {test_results["right_mrr"]:.5}, Avg : {test_results["mrr"]:.5}')
        self.logger.info(
            f'Final results: MR: Tail : {test_results["left_mr"]:.5}, Head : {test_results["right_mr"]:.5}, Avg : {test_results["mr"]:.5}')
        self.logger.info(
            f'Final results: left_hits@1: Tail : {test_results["left_hits@1"]:.5}, Head : {test_results["right_hits@1"]:.5}, Avg : {test_results["hits@1"]:.5}')
        self.logger.info(
            f'Final results: left_hits@3: Tail : {test_results["left_hits@3"]:.5}, Head : {test_results["right_hits@3"]:.5}, Avg : {test_results["hits@3"]:.5}')
        self.logger.info(
            f'Final results: left_hits@10: Tail : {test_results["left_hits@10"]:.5}, Head : {test_results["right_hits@10"]:.5}, Avg : {test_results["hits@10"]:.5}')

    def test_relation_type(self):
        """
        Function to get performance on various relation types
        """
        # load data
        relation_types = ['1-1', '1-n', 'n-1', 'n-n']

        # --- 加载特定关系类型的数据集 ---
        for rel_type in relation_types:
            self.load_relation_type_data(rel_type)

        # --- 构造 Data Loader ---
        for rel_type in relation_types:
            split_prefix = 'test_' + rel_type
            tail_key = f'{split_prefix}_tail'
            head_key = f'{split_prefix}_head'

            # 检查是否有数据被加载
            if tail_key in self.triples and self.triples[tail_key]:
                self.data_iter[tail_key] = self.get_data_loader(
                    TestDataset, tail_key, self.p.batch_size, shuffle=False)

            if head_key in self.triples and self.triples[head_key]:
                self.data_iter[head_key] = self.get_data_loader(
                    TestDataset, head_key, self.p.batch_size, shuffle=False)

        self.best_val_mrr, self.best_val, self.best_epoch, val_mrr = 0., {}, 0, 0.
        save_path = os.path.join(self.p.save_dir, self.p.name + '.pth')

        if self.p.restore:
            self.load_model(save_path)
            self.logger.info('Successfully Loaded previous model')

        # --- 评估特定关系类型 ---
        self.logger.info('\nResults on Relation Types:')
        for rel_type in relation_types:
            split_prefix = 'test_' + rel_type
            tail_key = f'{split_prefix}_tail'
            head_key = f'{split_prefix}_head'

            # 如果没有数据加载器，跳过
            if tail_key not in self.data_iter and head_key not in self.data_iter:
                self.logger.info(f'[{rel_type}]: No data loaded or data loader created. Skipping.')
                continue

            results = self.evaluate(split_prefix, -1)
            self.logger.info(
                f'[{rel_type}]:\n MRR: Tail: {results["left_mrr"]:.5}, Head: {results["right_mrr"]:.5}, Avg: {results["mrr"]:.5}\n'
                f'Hits@1: Tail: {results["left_hits@1"]:.5}, Head: {results["right_hits@1"]:.5}, Avg: {results["hits@1"]:.5}, \n'
                f'Hits@3: Tail: {results["left_hits@3"]:.5}, Head: {results["right_hits@3"]:.5}, Avg: {results["hits@3"]:.5}, \n'
                f'Hits@10: Tail: {results["left_hits@10"]:.5}, Head: {results["right_hits@10"]:.5}, Avg: {results["hits@10"]:.5}, \n'
            )

    def test_entity_degree(self):
        self.best_val_mrr, self.best_val, self.best_epoch, val_mrr = 0., {}, 0, 0.
        save_path = os.path.join(self.p.save_dir, self.p.name + '.pth')

        if self.p.restore:
            self.load_model(save_path)
            self.logger.info('Successfully Loaded previous model')

        # 触发计算
        results = self.evaluate('test', -1)

        area_num = 8
        entity_performance = [0] * area_num
        rr_sum = [0] * area_num
        cnt_sum = [0] * area_num
        entity2degree = self._get_entity2degree()
        for entity in self.entity_mrr_totals.keys():
            deg = entity2degree[entity]
            if 0 <= deg < 10:
                rr_sum[0] += self.entity_mrr_totals[entity]
                cnt_sum[0] += self.entity_count[entity]
            elif deg <15:
                rr_sum[1] += self.entity_mrr_totals[entity]
                cnt_sum[1] += self.entity_count[entity]
            elif deg <20:
                rr_sum[2] += self.entity_mrr_totals[entity]
                cnt_sum[2] += self.entity_count[entity]
            elif deg <50:
                rr_sum[3] += self.entity_mrr_totals[entity]
                cnt_sum[3] += self.entity_count[entity]
            elif deg <100:
                rr_sum[4] += self.entity_mrr_totals[entity]
                cnt_sum[4] += self.entity_count[entity]
            elif deg <200:
                rr_sum[5] += self.entity_mrr_totals[entity]
                cnt_sum[5] += self.entity_count[entity]
            elif deg <350:
                rr_sum[6] += self.entity_mrr_totals[entity]
                cnt_sum[6] += self.entity_count[entity]
            else:
                rr_sum[7] += self.entity_mrr_totals[entity]
                cnt_sum[7] += self.entity_count[entity]

        for i in range(len(entity_performance)):
            entity_performance[i] = rr_sum[i] / cnt_sum[i]

        print(f"Entity performance: {entity_performance}")

    def _get_entity2degree(self):
        entt2deg = ddict(int)
        train_data = f'data/{self.p.dataset}/train.txt'
        with open(train_data, 'r') as f:
            for line in f:
                h, r, t = line.strip().split('\t')
                entt2deg[self.ent2id[h]] += 1
                entt2deg[self.ent2id[t]] += 1
        return entt2deg

    def test_inference_time(self):
        """
        测试模型在测试集上的推理时间消耗，计算平均每条查询的耗时（毫秒）
        """
        import time

        save_path = os.path.join(self.p.save_dir, self.p.name + '.pth')
        if self.p.restore:
            self.load_model(save_path)
            self.logger.info('Successfully Loaded previous model')

        self.model.eval()

        total_queries = 0
        total_ms = 0.0

        with torch.no_grad():
            for mode in ['tail_batch', 'head_batch']:
                split_key = f'test_{mode.split("_")[0]}'
                train_iter = iter(self.data_iter[split_key])

                for step, batch in enumerate(train_iter):
                    sub, rel, obj, label = self.read_batch(batch, 'test')

                    torch.cuda.synchronize(self.device) if self.device.type == 'cuda' else None
                    t0 = time.perf_counter()

                    _ = self.model.forward(sub, rel)

                    torch.cuda.synchronize(self.device) if self.device.type == 'cuda' else None
                    elapsed_ms = (time.perf_counter() - t0) * 1000

                    batch_size = sub.size(0)
                    total_queries += batch_size
                    total_ms += elapsed_ms

                self.logger.info(f'[{mode}] total: {total_ms:.2f} ms, queries: {total_queries}')

        avg_ms = total_ms / total_queries if total_queries > 0 else 0
        self.logger.info(f'\nInference Time Benchmark:')
        self.logger.info(f'  Total queries: {total_queries}')
        self.logger.info(f'  Total time: {total_ms:.2f} ms')
        self.logger.info(f'  Avg per query: {avg_ms:.4f} ms')

    def case_study(self):
        """
        对具体的三元组进行案例分析，展示模型的预测能力
        此版本实现了过滤排名（Filtered Ranking）以提供更公平的评估
        """
        # 1. 加载最佳模型
        save_path = os.path.join(self.p.save_dir, self.p.name + '.pth')
        if not os.path.exists(save_path):
            self.logger.error(f"Model file not found at {save_path}. Please train a model first.")
            return

        self.load_model(save_path)
        self.logger.info('Successfully Loaded previous model for Case Study')
        self.model.eval()

        # 2. 定义测试的案例
        case_studies = [
            { # (/m/0m0bj, /location/location/contains, /m/01tzfz)
                'triple': ('/m/0m0bj', '/location/location/contains', None),
                'label': '/m/01tzfz',
                'mode': 'tail'
            },
            { # ('/m/0n85g', '/music/record_label/artist', '/m/01vrkdt')
                'triple': ('/m/0n85g', '/music/record_label/artist', None),
                'label': '/m/01vrkdt',
                'mode': 'tail' # 预测尾实体
            },
            { # (/m/0bxbr, /location/hud_county_place/county, /m/0bx9y)
                'triple': ('/m/0bxbr', '/location/hud_county_place/county', None),
                'label': '/m/0bx9y',
                'mode': 'tail'
            },
            { # (/m/03x6m, /sports/sports_team/colors, /m/01g5v)
                'triple': (None, '/sports/sports_team/colors', '/m/01g5v'),
                'label': '/m/03x6m',
                'mode': 'head' # 预测头实体
            },
            { # (/m/03rs8y, /people/person/profession, /m/02krf9)
                'triple': (None, '/people/person/profession', '/m/02krf9'),
                'label': '/m/03rs8y',
                'mode': 'head'
            }
        ]

        with torch.no_grad():
            for i, case in enumerate(case_studies):
                print(f"\n--- Case Study {i+1} ---")

                h_str, r_str, t_str = case['triple']
                target_entity_str = case['label']
                mode = case['mode']

                try:
                    # 3. 将字符串转换为ID
                    r_id = self.rel2id[r_str]
                    target_entity_id = self.ent2id[target_entity_str]

                    if mode == 'tail':
                        h_id = self.ent2id[h_str]
                        print(f"Query: ({h_str}, {r_str}, ?)")
                        print(f"Target: {target_entity_str}")

                        sub = torch.LongTensor([h_id]).to(self.device)
                        rel = torch.LongTensor([r_id]).to(self.device)
                        pred = self.model.forward(sub, rel)

                        # 获取用于过滤的所有正确答案
                        true_entities = self.sr2o_all.get((h_id, r_id), [])

                    elif mode == 'head':
                        t_id = self.ent2id[t_str]
                        r_inv_id = r_id + self.p.num_rel
                        print(f"Query: (?, {r_str}, {t_str})")
                        print(f"Target: {target_entity_str}")

                        sub = torch.LongTensor([t_id]).to(self.device)
                        rel = torch.LongTensor([r_inv_id]).to(self.device)
                        pred = self.model.forward(sub, rel)

                        # 获取用于过滤的所有正确答案
                        true_entities = self.sr2o_all.get((t_id, r_inv_id), [])

                    else:
                        print(f"Invalid mode: {mode}")
                        continue

                    # 5. 分析并打印结果
                    scores = pred.squeeze(0)

                    # 5a. 计算原始排名 (Raw Rank)
                    sorted_scores_raw, sorted_indices_raw = torch.sort(scores, descending=True)
                    raw_rank = (sorted_indices_raw == target_entity_id).nonzero().item() + 1
                    print(f"Raw Rank of '{target_entity_str}': {raw_rank}")

                    # 5b. 计算过滤后排名 (Filtered Rank)
                    # 关键步骤：复制一份分数，并将所有“其他”正确答案的分数设为极小值
                    filtered_scores = scores.clone()
                    for true_id in true_entities:
                        # 只过滤掉不是当前目标实体的其他正确答案
                        if true_id != target_entity_id:
                            filtered_scores[true_id] = -float('Inf')

                    sorted_scores_filtered, sorted_indices_filtered = torch.sort(filtered_scores, descending=True)
                    filtered_rank = (sorted_indices_filtered == target_entity_id).nonzero().item() + 1
                    print(f"Filtered Rank of '{target_entity_str}': {filtered_rank}")

                    # 5c. 打印过滤后的前5名预测结果
                    print("Top 5 Predictions (Filtered):")
                    for j in range(5):
                        top_entity_id = sorted_indices_filtered[j].item()
                        top_entity_str = self.id2ent[top_entity_id]
                        # 从原始分数中获取分数，以避免显示 -inf
                        original_score = scores[top_entity_id].item()
                        print(f"  {j+1}. {top_entity_str} (Score: {original_score:.4f})")

                except KeyError as e:
                    print(f"Error: Entity or relation '{e}' not found in the dataset's vocabulary. Skipping this case.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', dest='name', default='testrun', help='Set run name for saving/restoring models')
    parser.add_argument('--mode', dest='mode', default=None, choices=['train', 'overall', 'test_relation_type', 'test_entity_degree', 'case_study', 'inference_time'], help='Set the mode for runner')
    parser.add_argument('--data', dest='dataset', default='FB15k-237', help='Dataset to use, default: FB15k-237')
    parser.add_argument('--model', dest='model', default='compgcn', help='Model Name')
    parser.add_argument('--score_func', dest='score_func', default='conve', help='Score Function for Link prediction')
    parser.add_argument('--opn', dest='opn', default='corr', help='Composition Operation to be used in CompGCN')

    parser.add_argument('--batch', dest='batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--gamma', default=40.0, type=float, help='Margin')
    parser.add_argument('--epoch',dest='max_epochs', default=500,type=int,help='Number of epochs')
    parser.add_argument('--l2',default=0.0, type=float,help='L2 Regularization for Optimizer')
    parser.add_argument('--lr',default=0.001, type=float, help='Starting Learning Rate')
    parser.add_argument('--lbl_smooth', dest='lbl_smooth', default=0.1, type=float, help='Label Smoothing')
    parser.add_argument('--num_workers', dest='num_workers', default=10, type=int,help='Number of processes to construct batches')
    parser.add_argument('--seed',dest='seed', default=41504,type=int,help='Seed for randomization')

    parser.add_argument('--restore',dest='restore',action='store_true', help='Restore from the previously saved model')
    parser.add_argument('--bias',dest='bias', action='store_true', help='Whether to use bias in the model')

    parser.add_argument('--num_bases', dest='num_bases', default=-1, type=int, help='Number of basis relation vectors to use')
    parser.add_argument('--init_dim', dest='init_dim', default=100, type=int, help='Initial dimension size for entities and relations')
    parser.add_argument('--gcn_dim',dest='gcn_dim',default=200, type=int,help='Number of hidden units in GCN')
    parser.add_argument('--embed_dim',dest='embed_dim',default=None, type=int,help='Embedding dimension to give as input to score function')
    parser.add_argument('--gcn_layer',dest='gcn_layer',default=1, type=int,help='Number of GCN Layers to use')
    parser.add_argument('--gcn_drop',dest='dropout', default=0.1, type=float,	help='Dropout to use in GCN Layer')
    parser.add_argument('--hid_drop', dest='hid_drop', default=0.3, type=float, help='Dropout after GCN')

    # ConvE specific hyperparameters
    parser.add_argument('--hid_drop2', dest='hid_drop2', default=0.3, type=float, help='ConvE: Hidden dropout')
    parser.add_argument('--feat_drop', dest='feat_drop', default=0.3, type=float, help='ConvE: Feature Dropout')
    parser.add_argument('--k_w', dest='k_w', default=10, type=int, help='ConvE: k_w')
    parser.add_argument('--k_h',dest='k_h', default=20, type=int, help='ConvE: k_h')
    parser.add_argument('--num_filt', dest='num_filt', default=200, type=int, help='ConvE: Number of filters in convolution')
    parser.add_argument('--ker_sz', dest='ker_sz', default=7, type=int, help='ConvE: Kernel size to use')

    parser.add_argument('--save', dest='save_dir', required=True, help='The directory to save the model checkpoints')
    parser.add_argument('--adapt_aggr', dest='adapt_aggr', default=-1, type=int, help='use adaptive message aggregator or not')
    parser.add_argument('--time_string', dest='time_string', default='', type=str, help='restore adaptive message aggregator from a file')

    # Modify Loss-
    parser.add_argument('--loss_delta',	dest='loss_delta', default=-1, type=float, help='hyperparameter that determines the speed of increase of rejection rate')
    parser.add_argument('--keep_aux',	dest='keep_aux', default=True, type=lambda x: x.lower() in ('true', '1', 'yes'), help='Whether to keep the auxiliary triples')
    parser.add_argument('--loss_only_new', dest='loss_only_new', default=1, type=float, help='only modify the loss of the added auxiliary triples')
    parser.add_argument('--aux_triples', dest='aux_triples', default=None, type=str, help='Path to auxiliary_triples.txt')
    parser.add_argument('--aux_confidence', dest='aux_confidence', default=None, type=str, help='Path to auxiliary_triples_confidence.json')

    args = parser.parse_args()

    if args.restore:
        ckpt_path = os.path.join(args.save_dir, args.name + '.pth')
        if os.path.isfile(ckpt_path):
            ckpt_args = torch.load(ckpt_path, map_location='cpu').get('args', {})
            restore_only_modes = {'overall', 'test_relation_type', 'test_entity_degree', 'case_study', 'inference_time'}
            if args.mode in restore_only_modes:
                for k, v in ckpt_args.items():
                    if not hasattr(args, k):
                        setattr(args, k, v)
                    elif k == 'name' or k == 'mode' or k == 'restore' or k == 'save_dir':
                        continue
                    else:
                        setattr(args, k, v)
                print(f'\n[Auto-restore] Parameters loaded from checkpoint: {ckpt_path}')
        else:
            print(f'Warning: Checkpoint not found at {ckpt_path}')

    if not args.restore:
        args.name = args.name + '_' + args.time_string

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    runner = Runner(args)
    if args.mode == 'train':
        runner.fit()
    elif args.mode == 'overall':
        runner.test_overall()
    elif args.mode == 'test_relation_type':
        runner.test_relation_type()
    elif args.mode == 'test_entity_degree':
        runner.test_entity_degree()
    elif args.mode == 'case_study':
        runner.case_study()
    elif args.mode == 'inference_time':
        runner.test_inference_time()
    else:
        print("You should set a mode for runner")
