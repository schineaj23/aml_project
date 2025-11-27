import os
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
from complex_batchnorm import BatchNorm1d as ComplexBatchNorm1d
import argparse


class IQDataset(Dataset):
    def __init__(self, root_dir, max_per_class=50, transform=None, use_complex=False):
        """
        root_dir: path containing subfolders bpsk/, qpsk/, qam16/, qam64/
        max_per_class: limit loaded files per modulation class
        transform: transformation function if we so choose
        use_complex: use the complex128 signal directly instead of using [real, imag] shape (2, N)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.use_complex = use_complex

        self.class_map = {"bpsk": 0, "qpsk": 1, "qam16": 2, "qam64": 3}
        self.samples = []  # list of (filepath, label)

        signals_dir = os.path.join(root_dir, "signals")
        if not os.path.isdir(signals_dir):
            print(f"Signals directory not found: {signals_dir}")
            return

        # Keep counts per label so we can respect max_per_class if desired.
        counts = defaultdict(int)

        files = sorted([f for f in os.listdir(signals_dir) if f.endswith(".npy")])
        for f in files:
            base = f[:-4]  # strip .npy
            parts = base.split("_")
            if len(parts) < 3:
                # not matching expected pattern
                continue
            mod = parts[-1]
            label = self.class_map.get(mod)
            if label is None:
                # unknown modulation suffix; skip
                print("unknown modulation for file:", f)
                continue

            # Enforce per-class limit
            if max_per_class is not None and counts[label] >= max_per_class:
                continue

            self.samples.append((os.path.join(signals_dir, f), label))
            counts[label] += 1

        print(f"Loaded {len(self.samples)} total samples (counts per label: {dict(counts)})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # Load complex128 npy
        x = np.load(path)  # shape: (N,) complex128
        

        # Just use the torch.complex128 directly for the CVNN
        # shape (N,) complex128
        x_prime = x
        if not self.use_complex:
            # Convert complex -> 2-channel float32 [real, imag]
            x_prime = np.stack([x.real, x.imag], axis=0).astype(np.float32)
            # shape: (2, N)

        if self.transform:
            x_prime = self.transform(x_prime)
        label_arr = torch.zeros(4, dtype=torch.float32)
        label_arr[label] = 1.0
        
        return torch.tensor(x_prime), torch.tensor(label_arr, dtype=torch.float32)


# instead of their strategies which go from essentially 64 -> 1024
# since we only have 4 classes go from 1024 (sample len) -> 32 then FC to 4
# (just keep downsampling)


# VGG architecture adapted for 1D signal data expects IQ (2, N) shape
class ConventionalVGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            self.vgg_block_small(2, 1024),
            self.vgg_block_small(1024, 512),
            self.vgg_block_big(512, 256),
            self.vgg_block_big(256, 128),
            self.vgg_block_big(128, 128),
        )

        # further downsample with linear layers
        # maybe this is unnecessary just 128 -> 4
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)

    # copies the N_in -> N_out structure of their first few blocks
    def vgg_block_small(self, N_in, N_out):
        return nn.Sequential(
            nn.Conv1d(N_in, N_out, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(N_out, N_out, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

    # copies the N_in -> N_out -> N_out conv structure of rest of blocks
    def vgg_block_big(self, N_in, N_out):
        return nn.Sequential(
            nn.Conv1d(N_in, N_out, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(N_out, N_out, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(N_out, N_out, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(N_out, N_out, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(N_out, N_out, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

    def forward(self, x: torch.tensor):
        # x: (batch, 2, N)
        x = self.conv_blocks(x)

        # x: (batch, channels=512, length)
        # global avg pooling hack (in reality didnt want to do weird flattening stuff)
        x = torch.mean(x, dim=2)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_complex=False):
        super().__init__()

        if use_complex:
            self.norm = ComplexBatchNorm1d
            # self.activation = complextorch.nn.CReLU
            self.activation = nn.Sigmoid
        else:
            self.norm = nn.BatchNorm1d
            self.activation = nn.ReLU

        self.inner_block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=1,
            ),
            self.norm(out_channels),
            self.activation(),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=1,
            ),
            self.norm(out_channels),
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()

        self.outer_activation = self.activation()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.inner_block(x)
        return self.outer_activation(torch.add(out, identity))


# pretty shallow version just to get something running
class ResNet(nn.Module):
    def __init__(self, use_complex=False):
        super().__init__()

        if use_complex:
            # self.activation = complextorch.nn.CReLU()
            self.activation = nn.Sigmoid()
            input_features = 1
        else:
            self.activation = nn.ReLU()
            input_features = 2

        self.residual = nn.Sequential(
            ResidualBlock(input_features, 64, use_complex=use_complex),
            ResidualBlock(64, 128, use_complex=use_complex),
            ResidualBlock(128, 256, use_complex=use_complex),
            ResidualBlock(256, 256, use_complex=use_complex),
        )

        self.fc = nn.Linear(256, 4)

    def forward(self, x: torch.tensor):
        # x: (batch, 2, N)
        x = self.residual(x)
        # global average pooling
        x = torch.mean(x, dim=2)
        return self.activation(self.fc(x))

# inspired by https://arxiv.org/abs/1909.13299
# implemented in https://arxiv.org/pdf/2302.08286 (for tf/keras)
# just average together the real and imag parts for loss lol
class AverageCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn_real = nn.CrossEntropyLoss()
        self.loss_fn_imag = nn.CrossEntropyLoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if input.is_complex():
            real_loss = self.loss_fn_real(torch.real(input), target)
            imag_loss = self.loss_fn_imag(torch.imag(input), target)
            return (real_loss + imag_loss) / 2


        return self.loss_fn_real(input, target)

def train(dataloader, model, loss_fn, optimizer, epochs):
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    for epoch in range(epochs):
        model.train()
        with torch.autograd.set_detect_anomaly(True):
            for batch, (X, y) in enumerate(dataloader):
                X = X.to("cuda:0")
                y = y.to("cuda:0")
                # Compute prediction and loss
                X = X.reshape(batch_size, 2, -1)
                pred = model(X)

                loss = loss_fn(pred, y)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss, current = loss.item(), batch * batch_size + len(X)
                print(f"Epoch [{epoch + 1}/{epochs}], loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def main():
    parser = argparse.ArgumentParser(description="Train an IQ model")
    parser.add_argument("--dataset", "-d", type=str, default="./data",
                        help="path to dataset root directory (contains `signals/`)")
    parser.add_argument("--model", "-m", type=str, choices=["resnet", "vgg"], default="resnet",
                        help="model architecture to use")
    parser.add_argument("--use-complex", action="store_true", default=False,
                        help="treat inputs as complex128 (default: False)")
    parser.add_argument("--batch", type=int, default=5, help="training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="number of training epochs")

    print(torch.__version__)
    print(torch.cuda.is_available())

    args = parser.parse_args()

    dataset_root = args.dataset
    selected_model = args.model
    use_complex_inputs = args.use_complex
    batch_size = args.batch

    print(f"Using dataset={dataset_root}, model={selected_model}, use_complex={use_complex_inputs}, "
          f"batch_size={batch_size}")
    dataset = IQDataset(dataset_root, max_per_class=1000, use_complex=use_complex_inputs)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    for batch_x, batch_y in loader:
        print(batch_x.shape, batch_y)
        break

    print(batch_x[0])

    model = ResNet(use_complex=use_complex_inputs)
    #model = model.to(torch.complex128)
    model = model.to("cuda:0")
    optim = torch.optim.SGD(model.parameters())

    # print(model)
    # exit(0)
    torch.autograd.set_detect_anomaly(True)
    train(loader, model, AverageCrossEntropyLoss(), optim, args.epochs)


if __name__ == "__main__":
    main()

