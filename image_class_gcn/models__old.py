from copy import deepcopy
import torch
from torch.nn import Linear, PReLU
from torch_geometric.nn import GCNConv,  SAGEConv, GINConv, GATConv
from torchvision.ops import MLP

torch.manual_seed(42)


class GNNBase(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, pooling):
        super(GNNBase, self).__init__()

        self.activations = torch.nn.ModuleList(
            [PReLU() for _ in range(num_layers - 1)]
        )

        self.lin = Linear(hidden_dim, output_dim)

        self.pooling = pooling

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.pooling_name = pooling.__name__

    def forward(self, x, edge_index, batch):
        for gcn, act in zip(self.convs[:-1], self.activations):
            x = gcn(x, edge_index)
            x = act(x)

        x = self.convs[-1](x, edge_index)

        # 2. Readout layer
        x = self.pooling(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = self.lin(x)

        return x


class GenericGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, layer, pooling):
        super(GenericGNN, self).__init__()
        self.convs = torch.nn.ModuleList(
            [layer(in_channels=input_dim, out_channels=hidden_dim)] +
            [layer(in_channels=hidden_dim, out_channels=hidden_dim) for _ in range(num_layers - 1)]
        )

        self.activations = torch.nn.ModuleList(
            [PReLU() for _ in range(num_layers - 1)]
        )

        self.lin = Linear(hidden_dim, output_dim)

        self.pooling = pooling

        self.model_name = layer.__name__
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.pooling_name = pooling.__name__

    def forward(self, x, edge_index, batch):
        for gcn, act in zip(self.convs[:-1], self.activations):
            x = gcn(x, edge_index)
            x = act(x)

        x = self.convs[-1](x, edge_index)

        # 2. Readout layer
        x = self.pooling(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = self.lin(x)

        return x


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, pooling):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList(
            [GCNConv(in_channels=input_dim, out_channels=hidden_dim)] +
            [GCNConv(in_channels=hidden_dim, out_channels=hidden_dim) for _ in range(num_layers - 1)]
        )

        self.activations = torch.nn.ModuleList(
            [PReLU() for _ in range(num_layers - 1)]
        )

        self.lin = Linear(hidden_dim, output_dim)

        self.pooling = pooling

        self.model_name = 'GCN'
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.pooling_name = pooling.__name__

    def forward(self, x, edge_index, batch):
        for gcn, act in zip(self.convs[:-1], self.activations):
            x = gcn(x, edge_index)
            x = act(x)

        x = self.convs[-1](x, edge_index)

        # 2. Readout layer
        x = self.pooling(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = self.lin(x)

        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, pooling):
        super(GraphSAGE, self).__init__()
        self.convs = torch.nn.ModuleList(
            [SAGEConv(in_channels=input_dim, out_channels=hidden_dim)] +
            [SAGEConv(in_channels=hidden_dim, out_channels=hidden_dim) for _ in range(num_layers - 1)]
        )

        self.activations = torch.nn.ModuleList(
            [PReLU() for _ in range(num_layers - 1)]
        )

        self.lin = Linear(hidden_dim, output_dim)

        self.pooling = pooling

        self.model_name = 'GraphSAGE'
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.pooling_name = pooling.__name__

    def forward(self, x, edge_index, batch):
        for gcn, act in zip(self.convs[:-1], self.activations):
            x = gcn(x, edge_index)
            x = act(x)

        x = self.convs[-1](x, edge_index)

        # 2. Readout layer
        x = self.pooling(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = self.lin(x)

        return x


class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, pooling):
        super(GAT, self).__init__()
        self.convs = torch.nn.ModuleList(
            [GATConv(in_channels=input_dim, out_channels=hidden_dim, heads=4)] +
            [GATConv(in_channels=hidden_dim, out_channels=hidden_dim, heads=4) for _ in range(num_layers - 1)]
        )

        self.activations = torch.nn.ModuleList(
            [PReLU() for _ in range(num_layers - 1)]
        )

        self.lin = Linear(hidden_dim, output_dim)

        self.pooling = pooling

        self.model_name = GATConv.__name__
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.pooling_name = pooling.__name__

    def forward(self, x, edge_index, batch):
        for gcn, act in zip(self.convs[:-1], self.activations):
            x = gcn(x, edge_index)
            x = act(x)

        x = self.convs[-1](x, edge_index)

        # 2. Readout layer
        x = self.pooling(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = self.lin(x)

        return x


class GIN(GNNBase):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, pooling):
        super(GIN, self).__init__(hidden_dim, output_dim, num_layers, pooling)

        mlp_input = MLP(in_channels=input_dim, hidden_channels=[hidden_dim, hidden_dim])
        mlp_hidden = MLP(in_channels=hidden_dim, hidden_channels=[hidden_dim, hidden_dim])

        self.convs = torch.nn.ModuleList(
            [GINConv(nn=mlp_input)] +
            [GINConv(nn=deepcopy(mlp_hidden)) for _ in range(num_layers - 1)]
        )

        self.model_name = GINConv.__name__
