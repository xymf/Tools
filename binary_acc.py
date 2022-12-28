import numpy as np
import os
import gdalTools


def binary_accuracy(pred, label):
    w, h = pred.shape
    result = np.zeros((w, h, 3))
    pred = (pred >= 0.5)
    label = (label >= 0.5)

    TP = pred * label
    FP = pred * (1 - label)
    FN = (1 - pred) * label
    TN = (1 - pred) * (1 - label)

    # TP
    result[:, :, 0] = np.where(TP == 1, 255, result[:, :, 0])
    result[:, :, 1] = np.where(TP == 1, 255, result[:, :, 1])
    result[:, :, 2] = np.where(TP == 1, 255, result[:, :, 2])

    # FP
    result[:, :, 0] = np.where(FP == 1, 255, result[:, :, 0])
    result[:, :, 1] = np.where(FP == 1, 0, result[:, :, 1])
    result[:, :, 2] = np.where(FP == 1, 0, result[:, :, 2])

    # FN
    result[:, :, 0] = np.where(FN == 1, 0, result[:, :, 0])
    result[:, :, 1] = np.where(FN == 1, 0, result[:, :, 1])
    result[:, :, 2] = np.where(FN == 1, 255, result[:, :, 2])

    # TN
    result[:, :, 0] = np.where(TN == 1, 0, result[:, :, 0])
    result[:, :, 1] = np.where(TN == 1, 0, result[:, :, 1])
    result[:, :, 2] = np.where(TN == 1, 0, result[:, :, 2])

    return result


if __name__ == '__main__':
    import glob
    import tqdm

    gtPath = r'D:\MyWorkSpace\paper\fishpond\data_evaluation\test2\poly.tif'
    predList = glob.glob("./*/*/poly.tif")
    names = []
    accs = []
    ious = []
    f1s = []
    precisions = []
    recalls = []
    for predictPath in tqdm.tqdm(predList):
        outName = predictPath.replace(".tif", "_vis.tif")
        im_proj, im_geotrans, im_width, im_height, pred = gdalTools.read_img(predictPath)
        im_proj, im_geotrans, im_width, im_height, gt = gdalTools.read_img(gtPath)
        gt = np.where(gt > 0, 1, 0).astype(np.uint8)
        pred = np.where(pred > 0, 1, 0).astype(np.uint8)
        result = binary_accuracy(pred, gt)

        gdalTools.write_img(outName, im_proj, im_geotrans, result.transpose((2, 0, 1)).astype(np.uint8))

