import numpy as np
import torch
from barbar import Bar
from MetricsFile import Metrics
import tensorflow as tf
import matplotlib.pyplot as plt


def train(model, dataloader, optimizer, criterion, device):
    model.to(device)
    model.train()
    metrics = Metrics()

    for (id_imgs, inputs, labels) in Bar(dataloader):
        optimizer.zero_grad()

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        metrics.batch(labels=labels, preds=predicted, loss=loss.item())
    metrics.print_one_liner(phase='Train')


def validate(model, dataloader, criterion, device):
    model.to(device)
    model.eval()
    metrics = Metrics()

    with torch.no_grad():
        for (id_imgs, inputs, labels) in Bar(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            metrics.batch(labels=labels, preds=predicted, loss=loss.item())
    return metrics.print_one_liner(phase='Val')


def train_validate(model, train_loader, val_loader, optimizer, criterion, device, epochs, save_criteria, weights_path):
    best_criteria = 0
    min_val_loss = np.Inf
    epoch_no_improve = 0

    for epoch in range(1, epochs+1):
        print("Epoch #{}".format(epoch))
        train(model, train_loader, optimizer, criterion, device)
        metrics = validate(model, val_loader, criterion, device)

        if save_criteria == 'Loss': metrics['Model Loss'] *= -1
        if epoch == 1 or metrics['Model '+save_criteria] > best_criteria:
            best_criteria = metrics['Model '+save_criteria]
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': metrics['Model Accuracy'],
                'loss': metrics['Model Loss'],
                'sensitivity': metrics['Model Sensitivity'], 
                'specificity': metrics['Model Specificity']
            }, '{}weights_epoch_{}_{}_{}.pth'.format(weights_path, epoch, save_criteria, str(best_criteria).replace('.', '_')))

        val_loss = metrics['Model Loss']
        min_val_loss, epoch_no_improve, condition = EarlyStopping(val_loss, min_val_loss, epoch, epoch_no_improve)
        if condition == True:
            break
        else:
            continue



def EarlyStopping(val_loss, min_val_loss, epoch_num, epoch_no_improve):
    n_epochs_stop = 5
    if val_loss <= min_val_loss:
        epoch_no_improve = 0
        min_val_loss = val_loss
    else:
        epoch_no_improve += 1
    if epoch_num > 5 and epoch_no_improve == n_epochs_stop:
        print("Early Stopping")
        condition = True
        return min_val_loss, epoch_no_improve, condition
    else:
        condition = False
        return min_val_loss, epoch_no_improve, condition









