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
    return metrics.print_one_liner(phase='Train')


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


def train_validate(model, train_loader, val_loader, optimizer, criterion, device, epochs, save_criteria, weights_path, classification_system, plot_directory):
    best_criteria = 0
    min_val_loss = np.Inf
    epoch_no_improve = 0
    train_accuracy_array = []
    validation_accuracy_array = []
    train_loss_array = []
    validation_loss_array = []

    for epoch in range(epochs):
        print("Epoch #{}".format(epoch+1))
        metrics_train = train(model, train_loader, optimizer, criterion, device)
        metrics_val = validate(model, val_loader, criterion, device)

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
        '''
        val_loss = metrics_val['Model Loss']
        min_val_loss, epoch_no_improve, condition = EarlyStopping(val_loss, min_val_loss, epoch, epoch_no_improve)
        if condition is True:
            break
        else:
            continue
        '''

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

    if classification_system == 'GenderClassification':
        plot_accuracy_directory1 = plot_accuracy_directory + 'AccuracyChart/AccuracyChartGenderClass.png'
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
    # plt.ylim([0.0, 1.0])
    plt.title('Epochs vs Training and Validation Loss Chart')

    if classification_system == 'GenderClassification':
        plot_loss_directory1 = plot_loss_directory + 'LossChart/LossChartGenderClass.png'
    elif classification_system == 'RecessDistension':
        plot_loss_directory1 = plot_loss_directory + 'LossChart/LossChartRecessDist.png'
    plt.savefig(plot_loss_directory1)








