import torch.nn as nn
from torch import optim
from tqdm import tqdm, trange
from torch.optim.lr_scheduler import ExponentialLR
from src.error_measures import get_accuracy, get_error_measures


def prepare_optimizer(model, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    return optimizer, scheduler


# Lighter loop without additional conditions
def training_loop(model, train_loader, device, num_epochs=50, lr=1e-3):
    model.to(device)  # move model to GPU
    optimizer, scheduler = prepare_optimizer(model, lr)
    loss_fun = nn.MSELoss()

    for epoch in trange(num_epochs):
        model.train()
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)  # move data to GPU

            preds = model(data).squeeze(dim=1)
            loss = loss_fun(preds, targets)
            optimizer.zero_grad()
            loss.backward()  # Gradient calculation
            optimizer.step()  # Weight update
        scheduler.step()


def training_loop_stats(model, train_loader, test_loader, device, num_epochs=50, lr=1e-3,
                        collect_stats=False, collect_time=False):
    model.to(device)  # move model to GPU
    optimizer, scheduler = prepare_optimizer(model, lr)
    loss_fun = nn.MSELoss()
    stats = [[], [], [], [], []]
    acc = []
    times = []

    with tqdm(range(num_epochs)) as t:
        for epoch in t:
            model.train()
            for data, targets in train_loader:
                data, targets = data.to(device), targets.to(device)  # move data to GPU

                preds = model(data).squeeze(dim=1)
                loss = loss_fun(preds, targets)
                optimizer.zero_grad()
                loss.backward()  # Gradient calculation
                optimizer.step()  # Weight update
            scheduler.step()

            if collect_stats:
                errors = get_error_measures(model, test_loader, device=device, print_e=False)
                for i, err in enumerate(errors):
                    stats[i].append(err)
                acc.append(get_accuracy(model, test_loader, device))

            if collect_time:
                elapsed = t.format_dict["elapsed"]
                times.append(elapsed)
    stats.append(acc)
    return stats, times
