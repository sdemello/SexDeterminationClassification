import numpy as np
import torch
import torch.nn as nn
from barbar import Bar
from MetricsFile import Metrics
import tensorflow as tf
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


def train_validate_model(model, dataloader, batch_size, optimizer, criterion, device, epoch_number, cross_val_count):
    model.to(device)
    metrics_train = Metrics()
    metrics_valid = Metrics()
    temp_train_dataset = []
    temp_valid_dataset = []
    if cross_val_count is None:
        cross_val_count = 0
    k_fold_number = 4

    fold_train_indices, fold_val_indices = k_fold_indices(dataloader.dataset, k_fold_number)

    if cross_val_count >= k_fold_number:
        cross_val_count = 0

    for i in range(len(dataloader.dataset)):
        if i in fold_train_indices[cross_val_count]:
            temp_train_dataset.append(dataloader.dataset[i])

    for i in range(len(dataloader.dataset)):
        if i in fold_val_indices[cross_val_count]:
            temp_valid_dataset.append(dataloader.dataset[i])

    cross_val_count += 1

    train_dataloader = DataLoader(dataset=temp_train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset=temp_valid_dataset, batch_size=batch_size, shuffle=False)

    # Training
    model.train()

    for (id_imgs, inputs, labels) in Bar(train_dataloader):
        optimizer.zero_grad()

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        metrics_train.batch(labels=labels, preds=predicted, loss=loss.item())
    metrics_train.print_one_liner(phase='Train', condition=True)

    # Validation
    model.eval()

    with torch.no_grad():
        for (id_imgs, inputs, labels) in Bar(valid_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            metrics_valid.batch(labels=labels, preds=predicted, loss=loss.item())
    metrics_valid.print_one_liner(phase='Val', condition=True)

    return metrics_train.print_one_liner(phase='Train', condition=False), metrics_valid.print_one_liner(phase='Val', condition=False), cross_val_count


def train_validate(model, train_loader, batch_size, optimizer, criterion, device, epochs, save_criteria, weights_path, classification_system, plot_directory):
    best_criteria = 0
    min_val_loss = np.Inf
    epoch_no_improve = 0
    train_accuracy_array = []
    validation_accuracy_array = []
    train_loss_array = []
    validation_loss_array = []
    cross_val_count = 0
    get_last_epoch = epochs

    for epoch in range(epochs):
        print("Epoch #{}".format(epoch+1))
        metrics_train, metrics_val, cross_val_count = train_validate_model(model, train_loader, batch_size, optimizer, criterion, device, epoch, cross_val_count)

        temp_train_accuracy = metrics_train['Model Accuracy']
        temp_validation_accuracy = metrics_val['Model Accuracy']
        train_accuracy_array.append(metrics_train['Model Accuracy'])
        validation_accuracy_array.append(temp_validation_accuracy)

        temp_train_loss = metrics_train['Model Loss']
        temp_validation_loss = metrics_val['Model Loss']
        train_loss_array.append(temp_train_loss)
        validation_loss_array.append(temp_validation_loss)

        if save_criteria == 'Loss':
            metrics_val['Model Loss'] *= -1
        if epoch == 1 or metrics_val['Model '+save_criteria] > best_criteria:
            best_criteria = metrics_val['Model '+save_criteria]
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': metrics_val['Model Accuracy'],
                'loss': metrics_val['Model Loss'],
                'sensitivity': metrics_val['Model Sensitivity'],
                'specificity': metrics_val['Model Specificity']
            }, '{}weights_epoch_{}_{}_{}.pth'.format(weights_path, epoch, save_criteria, str(best_criteria).replace('.', '_')))
        val_loss = metrics_val['Model Loss']
        min_val_loss, epoch_no_improve, condition = EarlyStopping(val_loss, min_val_loss, epoch, epoch_no_improve)
        if condition is True:
	    get_last_epoch = epoch
	    PlotAccuracy(epochs, train_accuracy_array, validation_accuracy_array, classification_system, plot_directory)
    	    PlotLoss(epochs, train_loss_array, validation_loss_array, classification_system, plot_directory)
            break
        else:
            continue

    if get_last_epoch == epochs: 
        PlotAccuracy(epochs, train_accuracy_array, validation_accuracy_array, classification_system, plot_directory)
        PlotLoss(epochs, train_loss_array, validation_loss_array, classification_system, plot_directory)


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


def PlotAccuracy(epochs, train_array, validation_array, classification_system, plot_accuracy_directory):
    epoch_count = range(1, epochs+1)

    plt.figure(figsize=(10, 10))
    plt.plot(epoch_count, train_array, 'r--')
    plt.plot(epoch_count, validation_array, 'b-')
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xlim([1, epochs])
    plt.ylim([0.0, 1.0])
    plt.title('Epochs vs Training and Validation Accuracy Chart')

    if classification_system == 'SexClassification':
        plot_accuracy_directory1 = plot_accuracy_directory + 'AccuracyChart/AccuracyChartSexClass.png'
    elif classification_system == 'RecessDistension':
        plot_accuracy_directory1 = plot_accuracy_directory + 'AccuracyChart/AccuracyChartRecessDist.png'
    plt.savefig(plot_accuracy_directory1)


def PlotLoss(epochs, train_loss_array, validation_loss_array, classification_system, plot_loss_directory):
    epoch_count = range(1, epochs+1)

    plt.figure(figsize=(10, 10))
    plt.plot(epoch_count, train_loss_array, 'r--')
    plt.plot(epoch_count, validation_loss_array, 'b-')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim([1, epochs])
    plt.title('Epochs vs Training and Validation Loss Chart')

    if classification_system == 'SexClassification':
        plot_loss_directory1 = plot_loss_directory + 'LossChart/LossChartSexClass.png'
    elif classification_system == 'RecessDistension':
        plot_loss_directory1 = plot_loss_directory + 'LossChart/LossChartRecessDist.png'
    plt.savefig(plot_loss_directory1)


def k_fold_indices(data, k_folds):
    fold_size = len(data) // k_folds
    indices = np.arange(len(data))
    folds_train_indices = []
    folds_valid_indices = []
    for i in range(k_folds):
        valid_indices = indices[i*fold_size:(i+1)*fold_size]
        train_indices = np.concatenate([indices[:i*fold_size], indices[(i+1)*fold_size:]])
        folds_train_indices.append(train_indices)
        folds_valid_indices.append(valid_indices)
    return folds_train_indices, folds_valid_indices













