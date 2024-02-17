import torch
from torch.functional import F
import torch.nn as nn
import pandas as pd
from barbar import Bar
import tensorflow as tf
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import time
import os
from sklearn.metrics import roc_curve
from itertools import chain
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import cv2
from matplotlib import colormaps

from MetricsFile import Metrics

last_layer_name = "features.8"


def test(model, test_dataloader, criterion, device, batch_size, gradcam_img_directory):
    model.to(device)
    model.eval()

    metrics_test = Metrics()
    ids = []
    labels1 = []
    preds = []
    preds_values = []
    outputs_arr = []

    # find the last layer for the grad cam style map
    # train_nodes, eval_nodes = get_graph_node_names(model)

    with torch.no_grad():
        for id_imgs, inputs, labels, data_mean_img, data_std_img in Bar(test_dataloader):
            print(id_imgs[0])
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs1 = model(inputs)
            # probs = F.softmax(outputs, dim=1).data.cpu().numpy()[0]
            m = nn.Softmax(dim=1)
            outputs = m(outputs1)
            predicted_values, predicted_indices = torch.max(outputs.data, 1)

            loss = criterion(outputs1, labels)

            metrics_test.batch(labels=labels, preds=predicted_indices, loss=loss.item())

            ids.append(list(id_imgs))
            labels1.append(labels.tolist())
            preds.append(predicted_indices.tolist())
	    preds_values.append(predicted_values.tolist())
            outputs_arr.append(outputs.data.tolist())

            if len(inputs) >= batch_size:
                for i in range(batch_size):
                    if i == (batch_size - 1):
                        # GetSaliencyMap(inputs[batch_size - 1], data_mean_img[batch_size - 1], data_std_img[batch_size - 1], model, saliency_img_directory)
            		heatmap, pred_cls = make_gradcam_heatmap(inputs[batch_size - 1], model, last_layer_name)
			display_heatmap(inputs[batch_size - 1], data_mean_img[batch_size - 1], data_std_img[batch_size - 1], heatmap, id_imgs[batch_size - 1], labels[batch_size - 1], predicted_indices[batch_size - 1], predicted_values[batch_size - 1], gradcam_img_directory)
	    elif batch_size > len(inputs):
                for i in range(len(inputs)):
                    if i == (len(inputs)-1):
                        # GetSaliencyMap(inputs[len(inputs) - 1], data_mean_img[len(inputs) - 1], data_std_img[len(inputs) - 1], model, saliency_img_directory)
			heatmap, pred_cls = make_gradcam_heatmap(inputs[len(inputs) - 1], model, last_layer_name)
                        display_heatmap(inputs[len(inputs) - 1], data_mean_img[len(inputs) - 1], data_std_img[len(inputs) - 1], heatmap, id_imgs[len(inputs) - 1], labels[len(inputs) - 1], predicted_indices[len(inputs) - 1], predicted_values[len(inputs) - 1], gradcam_img_directory)

    ids = list(chain.from_iterable(ids))
    labels1_temp = list(chain.from_iterable(labels1))
    preds_temp = list(chain.from_iterable(preds))
    preds_values_temp = list(chain.from_iterable(preds_values))
    outputs_arr_temp = list(chain.from_iterable(outputs_arr))
    labels1_temp = np.asarray(labels1_temp, dtype=np.int32)
    labels1_temp = np.squeeze(labels1_temp)
    preds_temp = np.asarray(preds_temp, dtype=np.int32)
    preds_temp = np.squeeze(preds_temp)
    preds_values_temp = np.asarray(preds_values_temp, dtype=np.float32)
    preds_values_temp = np.squeeze(preds_values_temp)
    outputs_arr_temp = np.asarray(outputs_arr_temp, dtype=np.float32)
    outputs_arr_temp = np.squeeze(outputs_arr_temp)

    sensitivity_specificity_cutoff(labels1_temp, preds_values_temp)

    metrics_test.print_summary()

    return ids, labels1_temp, preds_temp, preds_values_temp, metrics_test.summary()


def InverseTransform(img, data_mean_img, data_std_img):
    data_mean_img = data_mean_img.tolist()
    data_std_img = data_std_img.tolist()

    altered_data_mean_img, inv_data_std_img = find_inv_mean_and_std(img, data_mean_img, data_std_img)
    inv_normalize = transforms.Normalize(altered_data_mean_img, inv_data_std_img)
    inv_normalized_img = inv_normalize(img)

    return inv_normalized_img


def find_inv_mean_and_std(img, data_mean_img, data_std_img):
    altered_data_mean_img = []
    inv_data_std_img = []

    for number1 in range(len(data_mean_img)):
        altered_data_mean_img.append(((-1 * (data_mean_img[number1])) / data_std_img[number1]))

    for number1 in range(len(data_std_img)):
        inv_data_std_img.append((1 / data_std_img[number1]))

    altered_data_mean_img = [float(number2) for number2 in altered_data_mean_img]
    inv_data_std_img = [float(number2) for number2 in inv_data_std_img]

    return altered_data_mean_img, inv_data_std_img


def PlotMaps(input1, slc, saliency_img_directory):
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(np.transpose(input1.cpu().detach().numpy(), (1, 2, 0)))
    plt.subplot(1, 2, 2)
    plt.imshow(slc.cpu().detach().numpy(), cmap=plt.cm.hot)
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M%S', t)
    saliency_file_name = ('saliency' + timestamp + '.png')
    plt.savefig(saliency_img_directory + saliency_file_name)


def GetSaliencyMap(input1, data_mean_img, data_std_img, model, saliency_img_directory):
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    input1 = input1.unsqueeze(0)
    input1.requires_grad_(True)
    torch.set_grad_enabled(True)

    preds2 = model(input1)
    score, indices = torch.max(preds2, 1)
    score.backward()
    slc, _ = torch.max(torch.abs(input1.grad[0]), dim=0)
    slc = (slc - slc.min())/(slc.max() - slc.min())

    with torch.no_grad():
        input_img_inv = InverseTransform(input1[0], data_mean_img, data_std_img)

    PlotMaps(input_img_inv, slc, saliency_img_directory)


def test_report(model, test_dataloader, criterion, device, report_path, batch_size, classification_system, gradcam_img_directory):
    ids, labels, preds, preds_values, metrics = test(model, test_dataloader, criterion, device, batch_size, gradcam_img_directory)
    test_df = pd.DataFrame(columns=['IDs', 'Labels', 'Predictions', 'PredictionValues'])
    for i in range(len(ids)):
        test_df.loc[i] = [ids[i], labels[i], preds[i], preds_values[i]]
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    report = pd.concat([test_df, metrics_df], axis=1, sort=False)
    report = report.convert_dtypes()
    report_file_name = ''
    if classification_system == 'SexClassification':
        report_file_name = 'ReportsSexClass.txt'
    elif classification_system == 'RecessDistension':
        report_file_name = 'ReportsRecessDist.txt'
    report.to_csv(report_path + report_file_name)
    report.to_string(buf=(report_path + report_file_name))


# Use Youden Index to determine cut-off for classification
def sensitivity_specificity_cutoff(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.argmax(tpr - fpr)
    print("Youden Index: ")
    print(thresholds[idx])



def get_model_outputs(model, img_tensor, layer_name):
    model.eval()

    return_nodes = {
        layer_name: "last_conv_op",
        "classifier": "out_classifier"
    }

    effnet_body = create_feature_extractor(model, return_nodes=return_nodes)

    output_nodes_dict = effnet_body(img_tensor)

    last_conv_layer_output = output_nodes_dict["last_conv_op"]
    preds_classifier = output_nodes_dict["out_classifier"]

    return last_conv_layer_output, preds_classifier


def make_gradcam_heatmap(img_input, model, last_layer_name):
    img_input = img_input.unsqueeze(0)
    img_input.requires_grad_(True)
    torch.set_grad_enabled(True)
    for param in model.parameters():
        param.requires_grad = True

    last_conv_layer_output, preds = get_model_outputs(model, img_input, last_layer_name)

    layer_grads = []
    hook = last_conv_layer_output.register_hook(lambda grad: layer_grads.append(grad.detach()))

    pred_index = torch.argmax(preds.squeeze(0))

    class_channel = preds[:, pred_index]
    class_channel.sum().backward()

    pooled_grads = torch.mean(layer_grads[0], (0, 2, 3))

    conv_op_reshape = last_conv_layer_output[0].permute(1, 2, 0).detach()

    heatmap = conv_op_reshape @ pooled_grads
    heatmap = heatmap.clamp(min=0.) / heatmap.max()

    hook.remove()

    return heatmap.cpu().detach().numpy(), int(pred_index.cpu().detach())


def display_heatmap(image, data_mean_img, data_std_img, heatmap, id_img, label, preds, preds_values, gradcam_img_directory):
    label = int(label)
    preds = int(preds)
    image = InverseTransform(image, data_mean_img, data_std_img)
    h, w, c = image.shape
    image1 = image.permute(1, 2, 0)
    image1 = image1.cpu().detach().numpy()

    heatmap = cv2.resize(heatmap, (c, w))

    heatmap = np.uint8(255 * heatmap)
    cmap = colormaps["jet"]
    colors = cmap(np.arange(256))[:, :3]
    rgb_heatmap = colors[heatmap]

    images = [image1, rgb_heatmap]

    plt.figure(figsize=(15, 9))
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.imshow(images[i])
        plt.tight_layout()
    txt = r"$\bf{{{a1}}}$ {a1_1}, $\bf{{{a2}}}$ {a2_2},  $\bf{{{a3}}}$ {a3_3}, $\bf{{{a4}}}$ {a4_4:.5f}".format(a1='IMG: ', a1_1=id_img, a2='LABEL: ', a2_2=edited_label, a3='PREDICTION: ', a3_3=edited_preds, a4='PREDICTION PROBABILITY: ', a4_4=preds_values)
    plt.figtext(0.2, 0.05, txt)
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M%S', t)
    gradcam_file_name = ('gradcam' + timestamp + '.png')
    plt.savefig(gradcam_img_directory + gradcam_file_name)











