import torch
from torch.functional import F
import pandas as pd
from barbar import Bar
import tensorflow as tf
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import time
import os

from MetricsFile import Metrics

def test(model, dataloader, criterion, device):
    model.to(device)
    model.eval()

    metrics = Metrics()
    ids = []
    labels1 = []
    preds = []

    with torch.no_grad():
        for id_imgs, inputs, labels in Bar(dataloader):
            print(id_imgs[0])
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            #probs = F.softmax(outputs, dim=1).data.cpu().numpy()[0]
            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)

            metrics.batch(labels=labels, preds=predicted, loss=loss.item())

            ids.append(id_imgs[0])
            labels1.append(labels.tolist())
            preds.append(predicted.tolist())

            GetSaliencyImage(inputs[len(inputs)-1], model)
    metrics.print_summary()

    return ids, labels1, preds, metrics.summary()


def InverseTransform(img):
    inv_normalize = transforms.Normalize(
        mean = [-0.1565/0.1428, -0.1488/0.1412, -0.1424/0.1361],
        std = [1/0.1438, 1/0.1412, 1/0.1361]
    )
    inv_normalized_img = inv_normalize(img)
    return inv_normalized_img


def PlotMaps(input, slc):
    saliency_img_directory = 'C:/Users/User/PycharmProjects/MachineLearning/SaliencyMaps/'
    #saliency_img_directory = '/home/shawn/Documents/BinaryClassifier/SaliencyMaps/'
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(np.transpose(input.cpu().detach().numpy(), (1, 2, 0)))
    plt.subplot(1, 2, 2)
    plt.imshow(slc.cpu().detach().numpy(), cmap=plt.cm.hot)
    pathlib.Path(saliency_img_directory).mkdir(parents=True, exist_ok=True)
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M%S', t)
    saliency_file_name = ('saliency' + timestamp + '.png')
    plt.savefig(saliency_img_directory + saliency_file_name)


def GetSaliencyImage(input, model):
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    input = input.unsqueeze(0)
    input.requires_grad_(True)
    torch.set_grad_enabled(True)

    preds2 = model(input)
    score, indices = torch.max(preds2, 1)
    score.backward()
    slc, _ = torch.max(torch.abs(input.grad[0]), dim=0)
    slc = (slc - slc.min())/(slc.max() - slc.min())

    with torch.no_grad():
        input_img_inv = InverseTransform(input[0])

    PlotMaps(input_img_inv, slc)


def test_report(model, dataloader, criterion, device, report_path):
    ids, labels, preds, metrics = test(model, dataloader, criterion, device)
    test_df = pd.DataFrame(columns=['IDs', 'Labels', 'Pred'])
    for i in range(len(ids)):
        test_df.loc[i] = [ids[i], labels[i], preds[i]]
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    report = pd.concat([test_df, metrics_df], axis=1, sort=False)
    report_file_name = 'Reports.txt'
    report.to_csv(report_path + report_file_name)




