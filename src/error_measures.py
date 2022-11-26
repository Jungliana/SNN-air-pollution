import numpy as np
import torch


def get_accuracy(model, loader, device, pct_close=0.25):
    correct = 0
    total = 0
    model.eval()
    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        output = model(data)
        pred = torch.round(output, decimals=2)
        correct += (torch.abs(pred-targets.view_as(pred)) <= torch.abs(pct_close * targets.view_as(pred))).sum().item()
        total += data.shape[0]
    return round(correct / total, 4)


def get_error_measures(model, loader, device, print_e=False):
    model.to(device)
    model.eval()
    targets, preds = gather_predictions(model, loader, device)
    index = calculate_index_of_agreement(targets, preds)
    mse, rmse = calculate_MSE_RMSE(targets, preds)
    mae = calculate_MAE(targets, preds)
    mape = calculate_MAPE(targets, preds)
    if print_e:
        print_measures(mae, mse, rmse, index, mape)
    return mae, mse, rmse, index, mape


def gather_predictions(model, loader, device):
    targets = np.array([])
    preds = np.array([])

    for data, target in loader:
        targets = np.concatenate([targets, target.numpy()])
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = torch.round(output, decimals=2)
        preds = np.concatenate([preds, pred.squeeze().cpu().detach().numpy()])

    return targets, preds


def calculate_MSE_RMSE(targets, preds):
    diff2 = (targets - preds)**2
    mse = diff2.mean()
    rmse = np.sqrt(mse)
    return mse, rmse


def calculate_index_of_agreement(targets, preds):
    obs_mean = targets.mean()
    numerator = (targets - preds)**2
    denominator = (np.abs(preds - obs_mean) + np.abs(targets - obs_mean))**2
    return 1 - (numerator.sum()/denominator.sum())


def calculate_MAE(targets, preds):
    return np.abs(targets-preds).mean()


def calculate_MAPE(targets, preds):
    return np.abs((targets-preds)/targets).mean() * 100


def print_measures(mae, mse, rmse, index, mape):
    print(f'MAE: {"{:.4f}".format(mae)}, MSE:{"{:.4f}".format(mse)},'
          f'\nRMSE: {"{:.4f}".format(rmse)}, IA: {"{:.4f}".format(index)}, MAPE: {"{:.4f}".format(mape)}%')


def collect_stats(model, stat_list, loader, device):
    results = get_error_measures(model, loader, device, print_e=False)
    for i in range(len(results)):
        stat_list[i] += results[i]
    stat_list[-1] += get_accuracy(model, loader, device, pct_close=0.25)


def print_average_stats(stat_list, trials):
    print(f'MAE: {stat_list[0]/trials}, MSE:{stat_list[1]/trials},'
          f'\nRMSE: {stat_list[2]/trials}, IA: {stat_list[3]/trials},'
          f'\nMAPE: {stat_list[4]/trials} %, acc: {(stat_list[5]/trials)*100} %\n')
