import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import radius_graph, knn_graph, JumpingKnowledge
from torch_scatter import scatter_softmax, scatter_sum
from models.common import GaussianSmearing, MLP, batch_hybrid_edge_connection, outer_product


class BaseX2HAttLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, ew_net_type='r', out_fc=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.act_fn = act_fn
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.ew_net_type = ew_net_type
        self.out_fc = out_fc

        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim

        # attention key func
        self.hk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        # attention value func
        self.hv_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        # attention query func
        self.hq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        # edge weight network
        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())
        elif ew_net_type == 'm':
            self.ew_net = nn.Sequential(nn.Linear(output_dim, 1), nn.Sigmoid())
        else:
            self.ew_net = None

        # edge gate
        self.edge_gate = nn.Sequential(
            nn.Linear(kv_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_heads),
            nn.Sigmoid()
        )

        if self.out_fc:
            self.node_output = MLP(2 * hidden_dim, hidden_dim, hidden_dim, norm=norm, act_fn=act_fn)



    def forward(self, h, r_feat, edge_feat, edge_index, e_w=None):
        N = h.size(0)
        src, dst = edge_index
        hi, hj = h[dst], h[src]

        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1)

        # K, V, Q
        k = self.hk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)  # [E,H,d]
        v = self.hv_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)  # [E,H,d]
        q = self.hq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)         # [N,H,d]

        # edge weight
        if self.ew_net is not None:
            if self.ew_net_type == 'r':
                e_w = self.ew_net(r_feat)
            else:  # 'm'
                e_w = self.ew_net(v.reshape(v.size(0), -1))
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.
        v = v * e_w.unsqueeze(-1)  # [E,H,d]

        # Gated attention：ReLU^2(logits) * gate
        gate = self.edge_gate(kv_input)  # [E,H]

        logits = (q[dst] * k).sum(-1) / np.sqrt(k.size(-1))  # [E,H]
        # att_raw = F.softmax(logits,dim=0)

        att_raw = F.relu(logits) ** 2                     # [E,H]
        att_raw = att_raw * gate                             # gated

        # sink-free 
        denom = scatter_sum(att_raw, dst, dim=0, dim_size=N) + 1e-12  # [N,H]
        alpha = att_raw / denom[dst]                                  # [E,H]

        m = alpha.unsqueeze(-1) * v                                   # [E,H,d]
        m = scatter_sum(m, dst, dim=0, dim_size=N)                    # [N,H,d]
        m = m.view(-1, self.output_dim)                               # [N,D]

        if self.out_fc:
            output = self.node_output(torch.cat([m, h], -1))
        else:
            output = m

        # Res + LayerNorm
        output = output + h
        return output



class BaseH2XAttLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, ew_net_type='r'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.act_fn = act_fn
        self.ew_net_type = ew_net_type

        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim

        self.xk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.xv_func = MLP(kv_input_dim, self.n_heads, hidden_dim, norm=norm, act_fn=act_fn)
        self.xq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())
        else:
            self.ew_net = None

        # edge gate
        self.edge_gate = nn.Sequential(
            nn.Linear(kv_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_heads),
            nn.Sigmoid()
        )

    def forward(self, h, rel_x, r_feat, edge_feat, edge_index, e_w=None):
        N = h.size(0)
        src, dst = edge_index
        hi, hj = h[dst], h[src]

        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1)

        k = self.xk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)  # [E,H,d]
        v = self.xv_func(kv_input)                                                           # [E,H]
        q = self.xq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)         # [N,H,d]

        if self.ew_net_type == 'r' and self.ew_net is not None:
            e_w = self.ew_net(r_feat)
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.
        v = v * e_w  # [E,H]

        # edge gate
        gate = self.edge_gate(kv_input)  # [E,H]

        logits = (q[dst] * k).sum(-1) / np.sqrt(k.size(-1))  # [E,H]
        att_raw = F.relu(logits) ** 2
        # att_raw = F.softmax(logits,dim=0)
        att_raw = att_raw * gate

        denom = scatter_sum(att_raw, dst, dim=0, dim_size=N) + 1e-12
        alpha = att_raw / denom[dst]  # [E,H]

        v_expand = v.unsqueeze(-1)        # [E,H,1]
        rel_x_h = rel_x.unsqueeze(1)      # [E,1,3]
        m = alpha.unsqueeze(-1) * v_expand * rel_x_h  # [E,H,3]
        output = scatter_sum(m, dst, dim=0, dim_size=N)      # [N,H,3]

        return output.mean(1)  # [N,3]


class AttentionLayerO2TwoUpdateNodeGeneral(nn.Module):
    def __init__(self, hidden_dim, n_heads, num_r_gaussian, edge_feat_dim,
                 act_fn='relu', norm=True,
                 num_x2h=1, num_h2x=1, r_max=10., num_node_types=8,
                 ew_net_type='r', x2h_out_fc=True, sync_twoup=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.num_r_gaussian = num_r_gaussian
        self.edge_feat_dim = edge_feat_dim
        self.act_fn = act_fn
        self.norm = norm
        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.r_max = r_max
        self.num_node_types = num_node_types
        self.ew_net_type = ew_net_type
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup


        self.distance_expansion = GaussianSmearing(0., r_max, num_gaussians=num_r_gaussian)

        # X2H 层
        self.x2h_layers = nn.ModuleList()
        for i in range(self.num_x2h):
            self.x2h_layers.append(
                BaseX2HAttLayer(
                    hidden_dim, hidden_dim, hidden_dim, n_heads,
                    edge_feat_dim,
                    r_feat_dim=num_r_gaussian * 4,   # 4 种 edge_type * num_r_gaussian
                    act_fn=act_fn, norm=norm,
                    ew_net_type=self.ew_net_type, out_fc=self.x2h_out_fc
                )
            )

        # H2X 层
        self.h2x_layers = nn.ModuleList()
        for i in range(self.num_h2x):
            self.h2x_layers.append(
                BaseH2XAttLayer(
                    hidden_dim, hidden_dim, hidden_dim, n_heads,
                    edge_feat_dim,
                    r_feat_dim=num_r_gaussian * 4,
                    act_fn=act_fn, norm=norm,
                    ew_net_type=self.ew_net_type
                )
            )

    def forward(self, h, x, edge_attr, edge_index, mask_ligand,
                e_w=None, fix_x=False):
        """
        h: [N, D]
        x: [N, 3]
        edge_attr: [E, 4] （one-hot edge type）
        edge_index: [2, E]
        mask_ligand: [N]
        e_w: [E,1] or None
        """
        src, dst = edge_index
        if self.edge_feat_dim > 0:
            edge_feat = edge_attr
        else:
            edge_feat = None

        rel_x = x[dst] - x[src]
        dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)


        h_in = h
        for i in range(self.num_x2h):
            dist_feat = self.distance_expansion(dist)         # [E, G]
            dist_feat = outer_product(edge_attr, dist_feat)   # [E, 4G]
            h_out = self.x2h_layers[i](h_in, dist_feat, edge_feat, edge_index, e_w=e_w)
            h_in = h_out
        x2h_out = h_in                                        # [N, D]

        new_h = h if self.sync_twoup else x2h_out
        for i in range(self.num_h2x):
            dist_feat = self.distance_expansion(dist)
            dist_feat = outer_product(edge_attr, dist_feat)
            delta_x = self.h2x_layers[i](new_h, rel_x, dist_feat, edge_feat, edge_index, e_w=e_w)
            if not fix_x:
                x = x + delta_x * mask_ligand[:, None]
            rel_x = x[dst] - x[src]
            dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)

        h_out_final = x2h_out

        return h_out_final, x







class UniTransformerO2TwoUpdateGeneral(nn.Module):
    def __init__(self, num_blocks, num_layers, hidden_dim, n_heads=1, k=32,
                 num_r_gaussian=50, edge_feat_dim=0, num_node_types=8, act_fn='relu', norm=True,
                 cutoff_mode='radius', ew_net_type='r',
                 num_init_x2h=1, num_init_h2x=0, num_x2h=1, num_h2x=1,
                 r_max=10., x2h_out_fc=True, sync_twoup=False,
                 jk_mode='max',        # 'last' | 'max' | 'sum' | 'concat'
                 jk_x_mode='last'       # 'last' | 'mean' | 'max'
                 ):
        super().__init__()

        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.num_r_gaussian = num_r_gaussian
        self.edge_feat_dim = edge_feat_dim
        self.act_fn = act_fn
        self.norm = norm
        self.num_node_types = num_node_types
        self.cutoff_mode = cutoff_mode
        self.k = k
        self.ew_net_type = ew_net_type
        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.num_init_x2h = num_init_x2h
        self.num_init_h2x = num_init_h2x
        self.r_max = r_max
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup


        self.jk_mode = jk_mode
        self.jk_x_mode = jk_x_mode

        # in UniTransformerO2TwoUpdateGeneral.__init__
        self.use_bigru = True  


        # BiGRU
        assert hidden_dim % 2 == 0, "BiGRU requires hidden_dim even when hidden_size=hidden_dim//2"

        self.bigru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=False,  # 输入用 [T, N, D]
        )


        self.bigru_ln = nn.LayerNorm(hidden_dim)

        self.distance_expansion = GaussianSmearing(0., r_max, num_gaussians=num_r_gaussian)
        if self.ew_net_type == 'global':
            self.edge_pred_layer = MLP(num_r_gaussian, 1, hidden_dim)

        self.init_h_emb_layer = self._build_init_h_layer()
        self.base_block = self._build_share_blocks()


        self.num_jk_layers = 1 + num_blocks * num_layers

    

        self.h2x_layers = nn.ModuleList()
        for i in range(self.num_h2x):
            self.h2x_layers.append(
                BaseH2XAttLayer(
                    hidden_dim, hidden_dim, hidden_dim, n_heads,
                    edge_feat_dim,
                    r_feat_dim=num_r_gaussian * 4,
                    act_fn=act_fn, norm=norm,
                    ew_net_type=self.ew_net_type
                )
            )

    def _bigru_aggregate_h(self, h_list):
        """
        h_list: List[Tensor [N, D]]
        return: Tensor [N, D]
        """
        h_seq = torch.stack(h_list, dim=0)  # [T, N, D]
        out_seq, h_n = self.bigru(h_seq)  # out_seq: [T, N, D]
        h_out = out_seq[-1]  # [N, D]
        h_out = self.bigru_ln(h_out)
        return h_out

    def _build_init_h_layer(self):
        layer = AttentionLayerO2TwoUpdateNodeGeneral(
            self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim,
            act_fn=self.act_fn, norm=self.norm,
            num_x2h=self.num_init_x2h, num_h2x=self.num_init_h2x,
            r_max=self.r_max, num_node_types=self.num_node_types,
            ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc,
            sync_twoup=self.sync_twoup,
        )
        return layer

    def _build_share_blocks(self):
        base_block = []
        for l_idx in range(self.num_layers):
            layer = AttentionLayerO2TwoUpdateNodeGeneral(
                self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim,
                act_fn=self.act_fn, norm=self.norm,
                num_x2h=self.num_x2h, num_h2x=self.num_h2x,
                r_max=self.r_max, num_node_types=self.num_node_types,
                ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc,
                sync_twoup=self.sync_twoup,
            )
            base_block.append(layer)
        return nn.ModuleList(base_block)



    def _connect_edge(self, x, mask_ligand, batch):
        if self.cutoff_mode == 'radius':
            edge_index = radius_graph(x, r=self.r_max, batch=batch, flow='source_to_target')
        elif self.cutoff_mode == 'knn':
            edge_index = knn_graph(x, k=self.k, batch=batch, flow='source_to_target')
        elif self.cutoff_mode == 'hybrid':
            edge_index = batch_hybrid_edge_connection(
                x, k=self.k, mask_ligand=mask_ligand, batch=batch, add_p_index=True)
        else:
            raise ValueError(f'Not supported cutoff mode: {self.cutoff_mode}')
        return edge_index

    @staticmethod
    def _build_edge_type(edge_index, mask_ligand):
        src, dst = edge_index
        edge_type = torch.zeros(len(src), dtype=torch.long, device=src.device)
        n_src = mask_ligand[src] == 1
        n_dst = mask_ligand[dst] == 1
        edge_type[n_src & n_dst] = 0
        edge_type[n_src & ~n_dst] = 1
        edge_type[~n_src & n_dst] = 2
        edge_type[~n_src & ~n_dst] = 3
        edge_type = F.one_hot(edge_type, num_classes=4).float()
        return edge_type

    def _jk_aggregate_h(self, h_list):
        """
        h_list: List[Tensor [N, D]]
        """
        if self.jk_mode == 'concat':
            h_cat = torch.cat(h_list, dim=-1)  # [N, D * L]
            return self.jk_linear(h_cat)
        elif self.jk_mode == 'max':
            h_stack = torch.stack(h_list, dim=0)  # [L, N, D]
            return h_stack.max(dim=0).values
        elif self.jk_mode == 'sum':
            h_stack = torch.stack(h_list, dim=0)
            return h_stack.sum(dim=0)
        else:  # 'last' 或 None
            return h_list[-1]

    def _jk_aggregate_x(self, x_list):
        """
        x_list: List[Tensor [N, 3]]
        """
        if self.jk_x_mode == 'mean':
            x_stack = torch.stack(x_list, dim=0)  # [L, N, 3]
            return x_stack.mean(dim=0)
        elif self.jk_x_mode == 'max':
            x_stack = torch.stack(x_list, dim=0)
            return x_stack.max(dim=0).values
        else:  # 'last'
            return x_list[-1]



    def forward(self, h, x, mask_ligand, batch, return_all=False, fix_x=False):
        """
        h: [N, D] 
        x: [N, 3]
        mask_ligand: [N]
        batch: [N]
        """

        h_list = [h]
        x_list = [x]

        # all_x = [x]
        # all_h = [h]

        for b_idx in range(self.num_blocks):
            edge_index = self._connect_edge(x, mask_ligand, batch)
            src, dst = edge_index

            edge_type = self._build_edge_type(edge_index, mask_ligand)

            if self.ew_net_type == 'global':
                dist = torch.norm(x[dst] - x[src], p=2, dim=-1, keepdim=True)
                dist_feat = self.distance_expansion(dist)
                logits = self.edge_pred_layer(dist_feat)
                e_w = torch.sigmoid(logits)
            else:
                e_w = None



            for l_idx, layer in enumerate(self.base_block):
                h, x = layer(h, x, edge_type, edge_index,
                             mask_ligand, e_w=e_w, fix_x=fix_x)
                h_list.append(h)
                # x_list.append(x)

            if getattr(self, "use_bigru", False):
                h_jk = self._bigru_aggregate_h(h_list)
            else:
                h_jk = self._jk_aggregate_h(h_list)

            # x_jk = self._jk_aggregate_x(x_list)


            rel_x = x[dst] - x[src]
            dist = torch.norm(rel_x, dim=-1, keepdim=True)

            dist_feat = self.distance_expansion(dist)
            r_feat = outer_product(edge_type, dist_feat)

            delta_x = self.h2x_layers[-1](
                h_jk,
                rel_x,
                r_feat,
                edge_type,
                edge_index
            )
            x_out = x + delta_x * mask_ligand[:, None]
            # all_x.append(x_out)
            # all_h.append(h_jk)

        outputs = {'x': x_out, 'h': h_jk}
        if return_all:
            outputs.update({
                # 'all_x': all_x, 'all_h': all_h,
                            'h_jk': h_jk, 'x': x})
        return outputs

