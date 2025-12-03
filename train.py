import os
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
from complex_batchnorm import BatchNorm1d as ComplexBatchNorm1d
from complextorch import nn as cvnn
import argparse
from tqdm import tqdm
from torch.profiler import profile, ProfilerActivity, record_function


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

        return x_prime, label_arr


# instead of their strategies which go from essentially 64 -> 1024
# since we only have 4 classes go from 1024 (sample len) -> 32 then FC to 4
# (just keep downsampling)


# VGG architecture adapted for 1D signal data expects IQ (2, N) shape
class ConventionalVGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            self.vgg_block_small(2, 64),
            self.vgg_block_small(64, 128),
            self.vgg_block_small(128, 256),
            # self.vgg_block_big(128, 128),
            # self.vgg_block_big(256, 512),
            # self.vgg_block_big(256, 512),
            # self.vgg_block_big(256, 512),
            # self.vgg_block_big(512, 512),
        )

        # further downsample with linear layers
        # maybe this is unnecessary just 128 -> 4
        self.fc1 = nn.Linear(256, 4)
        # self.fc2 = nn.Linear(64, 32)
        # self.fc3 = nn.Linear(32, 4)

    # copies the N_in -> N_out structure of their first few blocks
    def vgg_block_small(self, N_in, N_out):
        return nn.Sequential(
            nn.Conv1d(N_in, N_out, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(N_out, N_out, kernel_size=3, padding=1),
            nn.MaxPool1d(2, stride=2),
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
            nn.MaxPool1d(2, stride=2),
        )

    def forward(self, x: torch.tensor):
        # x: (batch, 2, N)
        x = self.conv_blocks(x)

        # x: (batch, channels=512, length)
        # global avg pooling hack (in reality didnt want to do weird flattening stuff)
        x = torch.mean(x, dim=2)

        x = self.fc1(x)
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.inner_block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=1,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=1,
            ),
            nn.BatchNorm1d(out_channels),
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.inner_block(x)
        return F.relu(torch.add(out, identity))


# pretty shallow version just to get something running
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.residual = nn.Sequential(
            ResidualBlock(2, 64),
            ResidualBlock(64, 128),
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
        )
        self.fc = nn.Linear(256, 4)

    def forward(self, x: torch.tensor):
        # x: (batch, 2, N)
        x = self.residual(x)
        # global average pooling
        x = torch.mean(x, dim=2)
        return self.fc(x)


class ComplexResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()

        self.norm = ComplexBatchNorm1d
        self.activation = cvnn.zReLU

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


class ComplexResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.residual = nn.Sequential(
            ComplexResidualBlock(1, 64),
            ComplexResidualBlock(64, 128),
            ComplexResidualBlock(128, 256),
            ComplexResidualBlock(256, 256),
        )

        self.fc = nn.Linear(256, 4)

    def forward(self, x: torch.tensor):
        # x: (batch, 2, N)
        x = self.residual(x)
        # global average pooling
        x = torch.mean(x, dim=2)
        return self.fc(x)


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
            out = (real_loss + imag_loss) / 2
            return out

        return self.loss_fn_real(input, target)


def train(dataloader, model, is_complex, loss_fn, optimizer, epochs):
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    for epoch in range(epochs):
        model.train()
        with torch.autograd.set_detect_anomaly(True):
            for batch, (X, y) in enumerate(dataloader):
                if is_complex and not X.is_contiguous():
                    X = X.contiguous()
                if torch.cuda.is_available():
                    X = X.to("cuda:0", non_blocking=True)
                    y = y.to("cuda:0", non_blocking=True)
                # Compute prediction and loss
                if is_complex:
                    x = X.view(batch_size, 1, -1)
                else:
                    x = X.reshape(batch_size, 2, -1)
                pred = model(x)
                loss = loss_fn(pred, y)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss, current = loss.item(), batch * batch_size + len(X)
                print(f"Epoch [{epoch + 1}/{epochs}], loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def evaluate(dataloader, model, loss_fn, is_complex):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    batch_size = dataloader.batch_size
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for X, y in tqdm(dataloader):
            if is_complex and not X.is_contiguous():
                X = X.contiguous()
            if torch.cuda.is_available():
                X = X.to("cuda:0", non_blocking=True)
                y = y.to("cuda:0", non_blocking=True)
            if is_complex:
                x = X.view(batch_size, 1, -1)
            else:
                x = X.reshape(batch_size, 2, -1)
            pred = model(x)
            test_loss += loss_fn(pred, y)
            if is_complex:
                pred = torch.real(pred)
            correct += (pred.argmax(1) == y.argmax(1)).sum()
    test_loss = test_loss.item()
    test_loss /= num_batches
    correct = correct.item()
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    parser = argparse.ArgumentParser(description="Train an IQ model")
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="./data",
        help="path to dataset root directory (contains `signals/`)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        choices=["resnet", "vgg", "complex"],
        default="resnet",
        required=True,
        help="model architecture to use",
    )
    parser.add_argument("--batch", type=int, default=5, help="training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="number of training epochs")
    parser.add_argument(
        "--save-path",
        type=str,
        default="./model",
        help="path to save model checkpoint after training",
    )
    parser.add_argument("--save", "-s", action="store_true")
    parser.add_argument(
        "--eval-only", action="store_true", help="only run evaluation on the test set"
    )
    parser.add_argument(
        "--resume-checkpoint", "-c", action="store_true", help="resumes training from a checkpoint"
    )
    parser.add_argument("--load-path", type=str, help="file path of model to load")
    print(torch.__version__)
    print(torch.cuda.is_available())

    args = parser.parse_args()

    dataset_root = args.dataset
    selected_model = args.model
    use_complex_inputs = selected_model == "complex"
    batch_size = args.batch
    save_path = args.save_path
    save_model = args.save
    eval_only = args.eval_only
    load_model_path = args.load_path
    load_checkpoint = args.resume_checkpoint

    print(
        f"Using dataset={dataset_root}, model={selected_model}, use_complex={use_complex_inputs}, "
        f"batch_size={batch_size}"
    )
    train_dataset = IQDataset(dataset_root, max_per_class=1000, use_complex=use_complex_inputs)

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    if selected_model == "resnet":
        model = ResNet()
    elif selected_model == "vgg":
        model = ConventionalVGG()
    else:
        model = ComplexResNet()
        model = model.to(torch.complex128)
    if torch.cuda.is_available():
        model = model.to("cuda:0")

    optim = torch.optim.Adam(model.parameters())

    if eval_only:
        if load_model_path is None:
            print("Must provide --load-path when using --eval-only")
            return
        model.load_state_dict(torch.load(load_model_path, weights_only=True))
        test_dataset = IQDataset(dataset_root, max_per_class=20, use_complex=use_complex_inputs)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True, record_shapes=True) as prof:
        #     with record_function("model_inference"):
        evaluate(test_loader, model, AverageCrossEntropyLoss(), use_complex_inputs)
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        # prof.export_chrome_trace(f"trace{"_complex" if use_complex_inputs else ""}.json")
        return

    if load_checkpoint:
        if load_model_path is None:
            print("Must provide --load-path when using --resume-checkpoint")
            return
        model.load_state_dict(torch.load(load_model_path, weights_only=True))
        print(f"Resuming training from checkpoint {load_model_path}")

    train(loader, model, use_complex_inputs, AverageCrossEntropyLoss(), optim, args.epochs)

    if save_model:
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, f"{selected_model}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    test_dataset = IQDataset(dataset_root, max_per_class=1000, use_complex=use_complex_inputs)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    evaluate(test_loader, model, AverageCrossEntropyLoss(), use_complex_inputs)


if __name__ == "__main__":
    main()
