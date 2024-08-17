from torch import nn


class classifier(nn.Module):
    def __init__(
        self,
        input_size,
        num_class,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, num_class),
            nn.Sigmoid()
        )

    def forward(self, x):
        #batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        #x = x.view(batch_size, -1)
        return self.model(x)


if __name__ == "__main__":
    _ = classifier()