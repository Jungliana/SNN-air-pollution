import torch.nn as nn
from torch import optim
from tqdm import trange
from src.error_measures import get_accuracy


def prepare_optimizer(model):
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    return optimizer


def training_loop(model, train_loader, valid_loader, device, num_epochs=50, validation=False):
    optimizer = prepare_optimizer(model)
    loss_fun = nn.MSELoss()
    model.to(device)  # move model to GPU

    for epoch in trange(num_epochs):
        model.train()
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)  # move data to GPU

            preds = model(data).squeeze(dim=1)
            loss = loss_fun(preds, targets)

            optimizer.zero_grad()
            loss.backward()  # Gradient calculation
            optimizer.step()  # Weight update

        if validation and epoch % 5 == 0:
            print(f'Accuracy: {get_accuracy(model, valid_loader, device=device, pct_close=0.2)*100}%')

    print(f'Accuracy on validation dataset: {get_accuracy(model, valid_loader, device=device, pct_close=0.2)*100}%')
