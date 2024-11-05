from torch import nn


class EmbeddingModel(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.2):
        super(EmbeddingModel, self).__init__()
        self._encoder = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_size // 2, output_size),
        )

    def _forward(self, x):
        return self._encoder(x)

    def forward(self, x1, x2):
        return self._forward(x1), self._forward(x2)
