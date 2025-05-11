import torch.nn as nn
from torch_geometric.nn import MLP, PointGNNConv

class PointGNNCon(nn.Module):
    def __init__(self, hparams, encoder, decoder):
        super(PointGNNCon, self).__init__()

        # Hyperparameters
        self.nb_hidden_layers = hparams['nb_hidden_layers']
        self.size_hidden_layers = hparams['size_hidden_layers']
        self.bn_bool = hparams['bn_bool']
        self.activation = nn.ReLU()

        # Encoder and decoder networks
        self.encoder = encoder  # Maps input features to a latent spaces
        self.decoder = decoder  # Maps the final output to target dimensions

        # Input transform to match hidden layer size
        self.input_transform = nn.Linear(hparams['encoder'][-1], self.size_hidden_layers)

        # Define the initial PointGNNConv layer
        self.in_layer = PointGNNConv(
            mlp_h=MLP([self.size_hidden_layers, self.size_hidden_layers, 3]),
            mlp_f=MLP([self.size_hidden_layers + 3, self.size_hidden_layers, self.size_hidden_layers]),
            mlp_g=MLP([self.size_hidden_layers, self.size_hidden_layers, self.size_hidden_layers])
        )

        # Define hidden layers with PointGNNConv
        self.hidden_layers = nn.ModuleList([
            PointGNNConv(
                mlp_h=MLP([self.size_hidden_layers, self.size_hidden_layers, 3]),
                mlp_f=MLP([self.size_hidden_layers + 3, self.size_hidden_layers, self.size_hidden_layers]),
                mlp_g=MLP([self.size_hidden_layers, self.size_hidden_layers, self.size_hidden_layers])
            ) for _ in range(self.nb_hidden_layers - 1)
        ])

        # Projection layer to align with output layer input size
        self.out_projection = nn.Linear(self.size_hidden_layers, hparams['decoder'][0])

        # Define the output PointGNNConv layer
        self.out_layer = PointGNNConv(
            mlp_h=MLP([hparams['decoder'][0], hparams['decoder'][0], 3]),
            mlp_f=MLP([hparams['decoder'][0] + 3, hparams['decoder'][0], hparams['decoder'][0]]),
            mlp_g=MLP([hparams['decoder'][0], hparams['decoder'][0], hparams['decoder'][0]])
        )

        # Optional batch normalization layers
        if self.bn_bool:
            self.bn = nn.ModuleList([
                nn.BatchNorm1d(self.size_hidden_layers, track_running_stats=False)
                for _ in range(self.nb_hidden_layers)
            ])

    def forward(self, data):
        # Raw input features and positional data
        z, pos, edge_index = data.x, data.pos, data.edge_index

        # Debugging: Input features
        # print(f"Input features shape (z): {z.shape}")

        # Encode the raw input features
        z = self.encoder(z)

        # Debugging: After encoder
        # print(f"After encoder shape: {z.shape}")

        # Transform encoded features to match the hidden layer size
        z = self.input_transform(z)

        # Debugging: After input transform
        # print(f"After input transform shape: {z.shape}")

        # Process the input layer
        z = self.in_layer(z, pos, edge_index)
        if self.bn_bool:
            z = self.bn[0](z)
        z = self.activation(z)

        # Debugging: After in_layer
        # print(f"After in_layer shape: {z.shape}")

        # Pass through the hidden layers
        for n, layer in enumerate(self.hidden_layers):
            z = layer(z, pos, edge_index)
            if self.bn_bool:
                z = self.bn[n + 1](z)
            z = self.activation(z)

            # Debugging: After each hidden layer
            # print(f"After hidden layer {n + 1} shape: {z.shape}")

        # Project features to match the output layer input size
        z = self.out_projection(z)

        # Debugging: After out_projection
        # print(f"After out_projection shape: {z.shape}")

        # Process the output layer
        z = self.out_layer(z, pos, edge_index)

        # Debugging: After out_layer
        # print(f"After out_layer shape: {z.shape}")

        # Decode the final output
        z = self.decoder(z)

        # Debugging: After decoder
        # print(f"After decoder shape: {z.shape}")

        return z
