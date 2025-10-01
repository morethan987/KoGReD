from helper import *
from data_loader import *

# sys.path.append('./')
from loss_restraint_KGE_model.models import *

class Runner(object):

	entity_mrr_totals = {}
	entity_count = {}
	entity_mrr_average = {}
	def load_data(self):
		"""
		Reading in raw triples and converts it into a standard format. 

		Parameters
		----------
		self.p.dataset:         Takes in the name of the dataset (FB15k-237)
		
		Returns
		-------
		self.ent2id:            Entity to unique identifier mapping
		self.id2rel:            Inverse mapping of self.ent2id
		self.rel2id:            Relation to unique identifier mapping
		self.num_ent:           Number of entities in the Knowledge graph
		self.num_rel:           Number of relations in the Knowledge graph
		self.embed_dim:         Embedding dimension used
		self.data['train']:     Stores the triples corresponding to training dataset
		self.data['valid']:     Stores the triples corresponding to validation dataset
		self.data['test']:      Stores the triples corresponding to test dataset
		self.data_iter:		The dataloader for different data splits

		"""

		ent_set, rel_set = OrderedSet(), OrderedSet()
		for split in ['train', 'test', 'valid']:
			for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
				sub, rel, obj = map(str.lower, line.strip().split('\t'))
				ent_set.add(sub)
				rel_set.add(rel)
				ent_set.add(obj)

		self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
		self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
		self.rel2id.update({rel+'_reverse': idx+len(self.rel2id) for idx, rel in enumerate(rel_set)})

		self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
		self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

		self.p.num_ent		= len(self.ent2id)
		self.p.num_rel		= len(self.rel2id) // 2
		self.p.embed_dim	= self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim

		self.data = ddict(list)
		sr2o = ddict(set)
		sr2observed = ddict(set)

		for split in ['train', 'test', 'valid']:
			if split == "train":
				for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
					sub, rel, obj = map(str.lower, line.strip().split('\t'))
					sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
					self.data[split].append((sub, rel, obj))

					sr2o[(sub, rel)].add(obj)
					sr2o[(obj, rel+self.p.num_rel)].add(sub)
					if self.p.loss_delta > 0: 
						sr2observed[(sub, rel)].add(obj)
						sr2observed[(obj, rel+self.p.num_rel)].add(sub)
				
				if self.p.loss_delta > 0:
					for line in open('./data/{}/auxiliary_triples.txt'.format(self.p.dataset)):
						sub, rel, obj = map(str.lower, line.strip().split('\t'))
						sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
						self.data[split].append((sub, rel, obj))
						sr2o[(sub, rel)].add(obj)
						sr2o[(obj, rel+self.p.num_rel)].add(sub)
			else:
				for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
					sub, rel, obj = map(str.lower, line.strip().split('\t'))
					sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
					self.data[split].append((sub, rel, obj))

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
			self.triples  = ddict(list)

			for (sub, rel), obj in self.sr2o.items():
				self.triples['train'].append({'triple':(sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'observed_label': self.sr2observed[(sub, rel)], 'newadd_label': self.sr2newadd[(sub, rel)], 'sub_samp': 1})
		else:
			self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
			self.triples  = ddict(list)

			for (sub, rel), obj in self.sr2o.items():
				self.triples['train'].append({'triple':(sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})

		for split in ['test', 'valid']:
			for sub, rel, obj in self.data[split]:
				rel_inv = rel + self.p.num_rel
				self.triples['{}_{}'.format(split, 'tail')].append({'triple': (sub, rel, obj), 	   'label': self.sr2o_all[(sub, rel)]})
				self.triples['{}_{}'.format(split, 'head')].append({'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})

		self.triples = dict(self.triples)

		def get_data_loader(dataset_class, split, batch_size, shuffle=True):
			return  DataLoader(
					dataset_class(self.triples[split], self.p),
					batch_size      = batch_size,
					shuffle         = shuffle,
					num_workers     = max(0, self.p.num_workers),
					collate_fn      = dataset_class.collate_fn
				)

		self.data_iter = {
			'train':    	get_data_loader(TrainDataset, 'train', 	    self.p.batch_size) if (self.p.loss_delta<0) else get_data_loader(TrainDataset_addLoss, 'train', self.p.batch_size),
			'valid_head':   get_data_loader(TestDataset,  'valid_head', self.p.batch_size),
			'valid_tail':   get_data_loader(TestDataset,  'valid_tail', self.p.batch_size),
			'test_head':   	get_data_loader(TestDataset,  'test_head',  self.p.batch_size),
			'test_tail':   	get_data_loader(TestDataset,  'test_tail',  self.p.batch_size),
		}

		self.edge_index, self.edge_type = self.construct_adj()


	def construct_adj(self):
		"""
		Constructor of the runner class

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

		edge_index	= torch.LongTensor(edge_index).to(self.device).t()
		edge_type	= torch.LongTensor(edge_type). to(self.device)

		return edge_index, edge_type

	def __init__(self, params):
		"""
		Constructor of the runner class

		Parameters
		----------
		params:         List of hyper-parameters of the model
		
		Returns
		-------
		Creates computational graph and optimizer
		
		"""
		self.p			= params
		self.logger		= get_logger(self.p.name, self.p.log_dir, self.p.config_dir)

		self.logger.info(vars(self.p))
		pprint(vars(self.p))

		if self.p.gpu != '-1' and torch.cuda.is_available():
			self.device = torch.device('cuda')
			torch.cuda.set_rng_state(torch.cuda.get_rng_state())
			torch.backends.cudnn.deterministic = True
		else:
			self.device = torch.device('cpu')

		self.load_data()
		self.model        = self.add_model(self.p.model, self.p.score_func)
		self.optimizer    = self.add_optimizer(self.model.parameters())


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
		model_name = '{}_{}'.format(model, score_func)

		if   model_name.lower()	== 'compgcn_transe': 	model = CompGCN_TransE(self.edge_index, self.edge_type, params=self.p)
		elif model_name.lower()	== 'compgcn_distmult': 	model = CompGCN_DistMult(self.edge_index, self.edge_type, params=self.p)
		elif model_name.lower()	== 'compgcn_conve': 	model = CompGCN_ConvE(self.edge_index, self.edge_type, params=self.p)
		else: raise NotImplementedError

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
				triple, label, observed_label, newadd_label = [ _.to(self.device) for _ in batch]
				return triple[:, 0], triple[:, 1], triple[:, 2], label, observed_label, newadd_label
			else:
				triple, label = [ _.to(self.device) for _ in batch]
				return triple[:, 0], triple[:, 1], triple[:, 2], label
		else:
			triple, label = [ _.to(self.device) for _ in batch]
			return triple[:, 0], triple[:, 1], triple[:, 2], label

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
			'state_dict'	: self.model.state_dict(),
			'best_val'	: self.best_val,
			'best_epoch'	: self.best_epoch,
			'optimizer'	: self.optimizer.state_dict(),
			'args'		: vars(self.p)
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
		state			= torch.load(load_path)
		state_dict		= state['state_dict']
		self.best_val		= state['best_val']
		self.best_val_mrr	= self.best_val['mrr'] 

		self.model.load_state_dict(state_dict)
		self.optimizer.load_state_dict(state['optimizer'])

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
		left_results  = self.predict(split=split, mode='tail_batch')
		right_results = self.predict(split=split, mode='head_batch')
		results       = get_combined_results(left_results, right_results)
		self.logger.info('[Epoch {} {}]: MRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split, results['left_mrr'], results['right_mrr'], results['mrr']))
		return results

	def predict(self, split='valid', mode='tail_batch'):
		"""
		Function to run model evaluation for a given mode

		Parameters
		----------
		split: (string) 	If split == 'valid' then evaluate on the validation set, else the test set
		mode: (string):		Can be 'head_batch' or 'tail_batch'
		
		Returns
		-------
		resutls:			The evaluation results containing the following:
			results['mr']:         	Average of ranks_left and ranks_right
			results['mrr']:         Mean Reciprocal Rank
			results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

		"""
		self.model.eval()

		with torch.no_grad():
			results = {}
			train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])

			for step, batch in enumerate(train_iter):
				sub, rel, obj, label	= self.read_batch(batch, split)
				pred			= self.model.forward(sub, rel)
				b_range			= torch.arange(pred.size()[0], device=self.device)
				target_pred		= pred[b_range, obj]
				pred 			= torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
				pred[b_range, obj] 	= target_pred
				ranks			= 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]			
				ranks 			= ranks.float()

				ranks_ = torch.unsqueeze(ranks, dim=1) 
				label_ = label.argmax(dim=1) 
				label_ = torch.unsqueeze(label_, dim=1) 

				entity_ranks = torch.cat((label_, ranks_), dim=1)

				for row in entity_ranks:
					entity_id = row[0].item()
					rank = 1.0/(row[1].item())

					if entity_id in self.entity_mrr_totals:
						self.entity_mrr_totals[entity_id] += rank
						self.entity_count[entity_id] += 1

					else:
						self.entity_mrr_totals[entity_id] = rank
						self.entity_count[entity_id] = 1

				results['count']	= torch.numel(ranks) 		+ results.get('count', 0.0)
				results['mr']		= torch.sum(ranks).item() 	+ results.get('mr',    0.0)
				results['mrr']		= torch.sum(1.0/ranks).item()   + results.get('mrr',   0.0)
				for k in range(10):
					results['hits@{}'.format(k+1)] = torch.numel(ranks[ranks <= (k+1)]) + results.get('hits@{}'.format(k+1), 0.0)

				if step % 100 == 0:
					self.logger.info('[{}, {} Step {}]\t{}'.format(split.title(), mode.title(), step, self.p.name))

			for entity_id in self.entity_mrr_totals.keys():
				self.entity_mrr_average[entity_id] = self.entity_mrr_totals[entity_id]/self.entity_count[entity_id]
			sorted_dict = dict(sorted(self.entity_mrr_average.items(), key=lambda x: x[0]))
			sorted_dict = {int(float(key)): value for key, value in sorted_dict.items()}

		return results


	def run_epoch(self, epoch, val_mrr = 0, clean_rate = 1):
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

				pred	= self.model.forward(sub, rel)
				loss	= self.model.loss(pred, label)
			elif self.p.loss_only_new > 0:
				sub, rel, obj, label, obeserved_label, newadd_label = self.read_batch(batch, 'train')
				pred	= self.model.forward(sub, rel)
				loss = self.model.modify_loss_only_add(pred, label, newadd_label, clean_rate)
			else:
				sub, rel, obj, label, obeserved_label, newadd_label = self.read_batch(batch, 'train')
				pred    = self.model.forward(sub, rel)
				loss = self.model.modify_loss(pred, label, obeserved_label, clean_rate)

			loss.backward()
			self.optimizer.step()
			losses.append(loss.item())

			if step % 100 == 0:
				self.logger.info('[E:{}| {}]: Train Loss:{:.5},  Val MRR:{:.5}\t{}'.format(epoch, step, np.mean(losses), self.best_val_mrr, self.p.name))

		loss = np.mean(losses)
		self.logger.info('[Epoch:{}]:  Training Loss:{:.4}\n'.format(epoch, loss))
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
		save_path = os.path.join('./checkpoints', self.p.name)

		if self.p.restore:
			self.load_model(save_path)
			self.logger.info('Successfully Loaded previous model')

		kill_cnt = 0

		clean_rate = 1 # init

		for epoch in range(self.p.max_epochs):
			train_loss  = self.run_epoch(epoch, val_mrr, clean_rate)
			val_results = self.evaluate('valid', epoch)

			if val_results['mrr'] > self.best_val_mrr:
				self.best_val	   = val_results
				self.best_val_mrr  = val_results['mrr']
				self.best_epoch	   = epoch
				self.save_model(save_path)
				kill_cnt = 0
			else:
				kill_cnt += 1
				if kill_cnt % 10 == 0 and self.p.gamma > 5:
					self.p.gamma -= 5 
					self.logger.info('Gamma decay on saturation, updated value of gamma: {}'.format(self.p.gamma))
				if kill_cnt > 25: 
					self.logger.info("Early Stopping!!")
					break

			self.logger.info('[Epoch {}]: Training Loss: {:.5}, Valid MRR: {:.5}\n\n'.format(epoch, train_loss, self.best_val_mrr))

			if (epoch % 30 == 0) and (self.p.loss_delta > 0):
				# update clean_rate
				clean_rate -= self.p.loss_delta

		self.logger.info('Loading best model, Evaluating on Test data')
		self.load_model(save_path)
		test_results = self.evaluate('test', epoch)
		self.logger.info('[Epoch {} {}]: MRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(1, "test", test_results['left_mrr'], test_results['right_mrr'], test_results['mrr']))
		self.logger.info('[Epoch {} {}]: MR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(1, "test", test_results['left_mr'], test_results['right_mr'], test_results['mr']))
		self.logger.info('[Epoch {} {}]: left_hits@1: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(1, "test", test_results['left_hits@1'], test_results['right_hits@1'], test_results['hits@1']))
		self.logger.info('[Epoch {} {}]: left_hits@3: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(1, "test", test_results['left_hits@3'], test_results['right_hits@3'], test_results['hits@3']))
		self.logger.info('[Epoch {} {}]: left_hits@10: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(1, "test", test_results['left_hits@10'], test_results['right_hits@10'], test_results['hits@10']))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument('-name',		default='testrun',					help='Set run name for saving/restoring models')
	parser.add_argument('-data',		dest='dataset',         default='FB15k-237',            help='Dataset to use, default: FB15k-237')
	parser.add_argument('-model',		dest='model',		default='compgcn',		help='Model Name')
	parser.add_argument('-score_func',	dest='score_func',	default='conve',		help='Score Function for Link prediction')
	parser.add_argument('-opn',             dest='opn',             default='corr',                 help='Composition Operation to be used in CompGCN')

	parser.add_argument('-batch',           dest='batch_size',      default=128,    type=int,       help='Batch size')
	parser.add_argument('-gamma',		type=float,             default=40.0,			help='Margin')
	parser.add_argument('-gpu',		type=str,               default='1',			help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
	parser.add_argument('-epoch',		dest='max_epochs', 	type=int,       default=500,  	help='Number of epochs')
	parser.add_argument('-l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
	parser.add_argument('-lr',		type=float,             default=0.001,			help='Starting Learning Rate')
	parser.add_argument('-lbl_smooth',      dest='lbl_smooth',	type=float,     default=0.1,	help='Label Smoothing')
	parser.add_argument('-num_workers',	type=int,               default=10,                     help='Number of processes to construct batches')
	parser.add_argument('-seed',            dest='seed',            default=41504,  type=int,     	help='Seed for randomization')

	parser.add_argument('-restore',         dest='restore',         action='store_true',            help='Restore from the previously saved model')
	parser.add_argument('-bias',            dest='bias',            action='store_true',            help='Whether to use bias in the model')

	parser.add_argument('-num_bases',	dest='num_bases', 	default=-1,   	type=int, 	help='Number of basis relation vectors to use')
	parser.add_argument('-init_dim',	dest='init_dim',	default=100,	type=int,	help='Initial dimension size for entities and relations')
	parser.add_argument('-gcn_dim',	  	dest='gcn_dim', 	default=200,   	type=int, 	help='Number of hidden units in GCN')
	parser.add_argument('-embed_dim',	dest='embed_dim', 	default=None,   type=int, 	help='Embedding dimension to give as input to score function')
	parser.add_argument('-gcn_layer',	dest='gcn_layer', 	default=1,   	type=int, 	help='Number of GCN Layers to use')
	parser.add_argument('-gcn_drop',	dest='dropout', 	default=0.1,  	type=float,	help='Dropout to use in GCN Layer')
	parser.add_argument('-hid_drop',  	dest='hid_drop', 	default=0.3,  	type=float,	help='Dropout after GCN')

	# ConvE specific hyperparameters
	parser.add_argument('-hid_drop2',  	dest='hid_drop2', 	default=0.3,  	type=float,	help='ConvE: Hidden dropout')
	parser.add_argument('-feat_drop', 	dest='feat_drop', 	default=0.3,  	type=float,	help='ConvE: Feature Dropout')
	parser.add_argument('-k_w',	  	dest='k_w', 		default=10,   	type=int, 	help='ConvE: k_w')
	parser.add_argument('-k_h',	  	dest='k_h', 		default=20,   	type=int, 	help='ConvE: k_h')
	parser.add_argument('-num_filt',  	dest='num_filt', 	default=200,   	type=int, 	help='ConvE: Number of filters in convolution')
	parser.add_argument('-ker_sz',    	dest='ker_sz', 		default=7,   	type=int, 	help='ConvE: Kernel size to use')

	parser.add_argument('-logdir',          dest='log_dir',         default='./log/',               help='Log directory')
	parser.add_argument('-config',          dest='config_dir',      default='./config/',            help='Config directory')

	parser.add_argument('-adapt_aggr',	dest='adapt_aggr', 	default=-1,   	type=int, 	help='use adaptive message aggregator or not')
	# Modify Loss
	parser.add_argument('-loss_delta',		dest='loss_delta',		type=float,         default=-1,            help='hyperparameter that determines the speed of increase of rejection rate')
	parser.add_argument('-loss_only_new',		dest='loss_only_new',		type=float,         default=1,            help='only modify the loss of the added auxiliary triples')

	args = parser.parse_args()

	if not args.restore: args.name = args.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')

	set_gpu(args.gpu)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	model = Runner(args)
	model.fit()
