from sklearn.metrics import confusion_matrix

# https://www.v7labs.com/blog/intersection-over-union-guide
def intersection_over_union(pred, gt):
    tn, fp, fn, tp = confusion_matrix(gt, pred, labels=[0, 1]).ravel()


    return tp / (tp + fp + fn)
