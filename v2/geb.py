from typing import Optional, Tuple, Union
import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.nn.inits import zeros
from timm.models.layers import trunc_normal_

class EGBBlock(MessagePassing):
    
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int,int]],
        mid_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        score_dim = 128,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'mean')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.score_dim = score_dim

        self.lin_src = Linear(in_channels[0], heads * mid_channels, False,weight_initializer='glorot')
        self.lin_dst = Linear(in_channels[1], heads * mid_channels, False,weight_initializer='glorot')
        self.lin_y = Linear(in_channels[2], heads * mid_channels, False,weight_initializer='glorot')
        self.att = torch.nn.Sequential(
            torch.nn.LeakyReLU(0.2,True),
            Linear(mid_channels, score_dim, False,weight_initializer='glorot')  
            )
        self.score = torch.nn.Sequential(
            torch.nn.LeakyReLU(0.2,True),
            Linear(score_dim, score_dim * 2,weight_initializer='glorot')
            )

        self.lin_edge = None
        self.register_parameter('att_edge', None)

        self.bias_src = Parameter(torch.Tensor(mid_channels))
        self.bias_dist = Parameter(torch.Tensor(mid_channels))
        self.out_src = Linear(mid_channels,out_channels, True, weight_initializer='glorot')
        self.out_dist = Linear(mid_channels,out_channels, True, weight_initializer='glorot')
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        self.lin_y.reset_parameters()
        self.score[1].reset_parameters()
        self.att[1].reset_parameters()
        self.out_src.reset_parameters()
        self.out_dist.reset_parameters()
        
        trunc_normal_(self.lin_src.weight,std=.02)
        trunc_normal_(self.lin_dst.weight,std=.02)
        trunc_normal_(self.lin_y.weight,std=.02)
        trunc_normal_(self.score[1].weight,std=.02)
        trunc_normal_(self.att[1].weight,std=.02)
        trunc_normal_(self.out_src.weight,std=.02)
        trunc_normal_(self.out_dist.weight,std=.02)
        
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        zeros(self.bias_src)
        zeros(self.bias_dist)
        zeros(self.out_src.bias)
        zeros(self.out_dist.bias)



    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights=None):

        x_src, x_dst, y_src = x        
        x_src = self.lin_src(x_src)
        x_dst = self.lin_dst(x_dst)
        y_src = self.lin_y(y_src)

        x = (x_src, None)
    
        alpha_src = self.att(x_src.squeeze(1))
        alpha_dst = self.att(x_dst.squeeze(1)) 
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            raise SystemExit

        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)
        alpha_dist, alpha_src = alpha.split(self.score_dim,-1)
        
        out_dist = self.propagate(edge_index, x=(torch.cat((x_src,y_src),-1), x_dst), alpha=alpha_dist, size=size)
        out_src =  self.propagate(edge_index.clone().flip(0), x=(x_dst, torch.cat((x_src,y_src),-1)), alpha=alpha_src, size=size) 

        out_dist = out_dist + self.bias_dist
        out_src = out_src + self.bias_src
        return self.out_src(out_src), self.out_dist(out_dist)


    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    size_i: Optional[int]) -> Tensor:
        alpha = alpha_j - alpha_i
        return self.score(alpha).sigmoid()


    def message(self, x_j: Tensor, x_i:Tensor, alpha: Tensor) -> Tensor:
        x = torch.cat((x_j,x_i),-1)
        R = self.mid_channels * 3 // self.score_dim
        x = alpha.repeat_interleave(R,-1) * x
        out = x.view(-1,3,self.mid_channels).mean(1)
        return out
