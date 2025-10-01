from helper import *
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax

class CompGCNConv_adapt(MessagePassing):
	def __init__(self, in_channels, out_channels, num_rels, act=lambda x:x, params=None):
		super(self.__class__, self).__init__()

		self.p 			= params
		self.in_channels	= in_channels
		self.out_channels	= out_channels
		self.num_rels 		= num_rels
		self.act 		= act
		self.device		= None

		self.w_loop		= get_param((in_channels, out_channels))
		self.w_in		= get_param((in_channels, out_channels))
		self.w_out		= get_param((in_channels, out_channels))
		self.w_rel 		= get_param((in_channels, out_channels))
		self.loop_rel 		= get_param((1, in_channels))

		self.w_attn     = get_param((3*in_channels, out_channels))

		self.drop		= torch.nn.Dropout(self.p.dropout)
		self.bn			= torch.nn.BatchNorm1d(out_channels)

		self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
		self.a = torch.nn.Linear(out_channels, 1, bias=False).cuda()
		self.drop_ratio = self.p.dropout

		if self.p.bias: self.register_parameter('bias', Parameter(torch.zeros(out_channels)))

	def forward(self, x, edge_index, edge_type, rel_embed): 
		if self.device is None:
			self.device = edge_index.device

		num_edges = edge_index.size(1) // 2
		num_ent   = x.size(0)

		self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]

		self.in_norm     = self.compute_norm(self.in_index,  num_ent)
		self.out_norm    = self.compute_norm(self.out_index, num_ent)
		
		in_res = self.propagate(edge_index=edge_index, x=x, edge_type=edge_type, rel_emb=rel_embed, in_norm = self.in_norm, out_norm = self.out_norm)
		loop_res = torch.mm(x, self.w_loop)
		out = self.drop(in_res) + self.drop(loop_res)
		out = self.bn(out)

		return self.act(out), torch.matmul(rel_embed, self.w_rel)

	def rel_transform(self, ent_embed, rel_embed):
		if   self.p.opn == 'corr': 	trans_embed  = ccorr(ent_embed, rel_embed)
		elif self.p.opn == 'sub': 	trans_embed  = ent_embed - rel_embed
		elif self.p.opn == 'mult': 	trans_embed  = ent_embed * rel_embed
		else: raise NotImplementedError

		return trans_embed

	# def message(self, x_j, edge_type, rel_embed, edge_norm, mode):
	# 	weight 	= getattr(self, 'w_{}'.format(mode))
	# 	rel_emb = torch.index_select(rel_embed, 0, edge_type)
	# 	xj_rel  = self.rel_transform(x_j, rel_emb)
	# 	out	= torch.mm(xj_rel, weight)

	# 	return out if edge_norm is None else out * edge_norm.view(-1, 1)
	
	def message(self,x_i, x_j, edge_type, rel_emb, ptr, index, size_i, in_norm, out_norm):
		rel_emb = torch.index_select(rel_emb, 0, edge_type)
		xj_rel = self.rel_transform(x_j, rel_emb)
		num_edge = xj_rel.size(0)//2
		in_message = xj_rel[:num_edge]
		out_message = xj_rel[num_edge:]
		trans_in = torch.mm(in_message, self.w_in)
		trans_out = torch.mm(out_message, self.w_out)
		# out = torch.cat((trans_in, trans_out), dim=0)
		b = self.leaky_relu(torch.mm((torch.cat((x_i, rel_emb, x_j), dim=1)), self.w_attn)).cuda()
		b = self.a(b).float()
		alpha = softmax(b, index, ptr, size_i)
		alpha = F.dropout(alpha, p=self.drop_ratio)
		self.alpha = alpha
		trans_in_norm = trans_in * in_norm.view(-1, 1)
		trans_out_norm = trans_out * out_norm.view(-1, 1)
		out = torch.concat((trans_in_norm, trans_out_norm), dim=0)
		out = out * alpha.view(-1,1)  # # [num_edges*2, 200]
		return out

	def update(self, aggr_out):
		return aggr_out

	def compute_norm(self, edge_index, num_ent):
		row, col	= edge_index
		edge_weight 	= torch.ones_like(row).float()
		deg		= scatter_add( edge_weight, row, dim=0, dim_size=num_ent)	# Summing number of weights of the edges
		deg_inv		= deg.pow(-0.5)							
		deg_inv[deg_inv	== float('inf')] = 0
		norm		= deg_inv[row] * edge_weight * deg_inv[col]

		return norm

	def __repr__(self):
		return '{}({}, {}, num_rels={})'.format(
			self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)
