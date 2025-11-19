import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple

from .model import CompressedMLP


def build_mnist_loaders(batch_size_train: int = 64, batch_size_test: int = 1000) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)
    return train_loader, test_loader


def train_one_epoch_interleaved(
    model: CompressedMLP,
    device: torch.device,
    train_loader: DataLoader,
    aux_weight: float = 0.1,
    reg_weight: float = 1e-6,
) -> float:
    """Train interleaved parameter vector with expanded forward + aux band-1 + band-3 L2."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    interleaved_params = nn.Parameter(model.interleave_params().to(device))
    optimizer_inter = optim.SGD([interleaved_params], lr=0.01)
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer_inter.zero_grad()
        logits_full = model.forward_expanded(data, interleaved_params)
        loss_full = criterion(logits_full, target)
        logits_band1 = model.forward_banded(data, interleaved_params, bands=1)
        loss_aux = criterion(logits_band1, target)
        reg = model.compute_band3_l2(interleaved_params) if hasattr(model, "compute_band3_l2") else 0.0
        loss = loss_full + aux_weight * loss_aux + reg_weight * (reg if isinstance(reg, torch.Tensor) else 0.0)
        loss.backward()
        optimizer_inter.step()
        model.deinterleave_params(interleaved_params)
        running_loss += float(loss.item())
    return running_loss / max(1, len(train_loader))


def evaluate_merged(model: CompressedMLP, device: torch.device, test_loader: DataLoader) -> Tuple[float, float]:
    """Evaluate baseline merged path."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)
    test_loss /= max(1, len(test_loader))
    acc = 100.0 * correct / max(1, total)
    return test_loss, acc


