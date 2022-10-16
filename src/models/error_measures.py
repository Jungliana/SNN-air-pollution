import torch


def get_accuracy(model, loader, device, pct_close):
    correct = 0
    total = 0
    model.eval()
    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        output = model(data)
        pred = torch.round(output, decimals=2)
        correct += (torch.abs(pred-targets.view_as(pred)) <= torch.abs(pct_close * targets.view_as(pred))).sum().item()
        total += data.shape[0]
    return correct / total
