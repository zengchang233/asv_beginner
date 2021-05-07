from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

def compute_eer(y_true, y_pred):
    fpr, tpr, threshold = roc_curve(y_true, y_pred, pos_label = 1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    threshold = interp1d(fpr, threshold)(eer)
    return eer, threshold