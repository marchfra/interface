import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

# Check if GPU is available and set device accordingly
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    verbose: bool = False,
) -> float:
    """Train the model for one epoch."""
    size = len(dataloader.dataset)  # pyright: ignore[reportArgumentType]
    model.train()

    train_loss: float = 0

    for batch, (x, y) in enumerate(dataloader):
        x[:], y[:] = x.to(device), y.to(device)
        optimizer.zero_grad()

        # Compute prediction error
        y_pred = model(x)
        loss: Tensor = loss_fn(y_pred, y)
        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Print progress (11 updates total)
        if verbose and batch % (0.1 * size // x.shape[0]) == 0:
            batch_loss, current = loss.item(), batch * len(x)
            print(f"loss: {batch_loss:>7f} [{current:>5d}/{size:>5d}]")

    return train_loss / len(dataloader)


def test_classification(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    *,
    verbose: bool = False,  # noqa: PT028
) -> tuple[float, float]:
    """Test the trained model on a classification task."""
    size = len(dataloader.dataset)  # pyright: ignore[reportArgumentType]
    num_batches = len(dataloader)

    model.eval()
    test_loss: float = 0
    accuracy: float = 0
    with torch.no_grad():
        for x, y in dataloader:
            x[:], y[:] = x.to(device), y.to(device)
            y_pred = model(x)
            test_loss += loss_fn(y_pred, y).item()
            accuracy += (y_pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    accuracy /= size
    if verbose:
        print(
            f"Test Error:\n    "
            f"Accuracy: {(100 * accuracy):>0.1f}%, Avg loss: {test_loss:>8f}\n",
        )

    return test_loss, accuracy


def test_regression(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    *,
    verbose: bool = False,  # noqa: PT028
) -> float:
    """Test the trained model on a regression task."""
    num_batches = len(dataloader)

    model.eval()
    test_loss: float = 0
    with torch.no_grad():
        for x, y in dataloader:
            x[:], y[:] = x.to(device), y.to(device)
            y_pred = model(x)
            test_loss += loss_fn(y_pred, y).item()
    test_loss /= num_batches
    if verbose:
        print(f"Test Error:\n    Avg loss: {test_loss:>8f}\n")

    return test_loss


def train_classification(  # noqa: PLR0913
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    learning_rate: float,
    epochs: int = 50,
    *,
    verbose: bool = False,
) -> tuple[list[float], list[float], list[float]]:
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_losses: list[float] = []
    test_losses: list[float] = []
    accuracies: list[float] = []

    for epoch in range(epochs):
        # Halve learning rate after half epochs
        if epoch == int(epochs / 2):
            optimizer.param_groups[0]["lr"] = learning_rate / 2

        epoch_loss = train(train_dataloader, model, loss_fn, optimizer, verbose=verbose)
        test_loss, accuracy = test_classification(
            test_dataloader,
            model,
            loss_fn,
            verbose=verbose,
        )

        if verbose:
            print(
                f"Epoch {epoch + 1:2}/{epochs} - "
                f"Accuracy: {(100 * accuracy):>0.1f}%, Avg loss: {test_loss:>8f}",
            )

        train_losses.append(epoch_loss)
        test_losses.append(test_loss)
        accuracies.append(accuracy)

    if verbose:
        print("Done!")

    return train_losses, test_losses, accuracies


def train_regression(  # noqa: PLR0913
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    learning_rate: float,
    epochs: int = 50,
    test_interval: int = 1,
    *,
    verbose: bool = False,
) -> tuple[list[float], list[float]]:
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_losses: list[float] = []
    test_losses: list[float] = []

    for epoch in range(epochs):
        # Halve learning rate after half epochs
        if epoch == int(epochs / 2):
            optimizer.param_groups[0]["lr"] = learning_rate / 2

        epoch_loss = train(train_dataloader, model, loss_fn, optimizer, verbose=verbose)
        train_losses.append(epoch_loss)
        if epoch % test_interval == 0:
            test_loss = test_regression(
                test_dataloader,
                model,
                loss_fn,
                verbose=verbose,
            )
            test_losses.append(test_loss)

        # print(f"Epoch {epoch + 1:2}/{epochs} - Avg loss: {test_loss:>8f}")

    if (epochs - 1) % test_interval != 0:
        test_loss = test_regression(
            test_dataloader,
            model,
            loss_fn,
            verbose=verbose,
        )
        test_losses.append(test_loss)
    # print("Done!")

    return train_losses, test_losses
