import numpy as np
from scipy import stats

from scipy.stats import hmean, kendalltau


np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(precision=2)



class MetricsDict:
    """A class to store all the data for computing metrics.
    """
    def __init__(self):
        self.metric_dict = {
            "pred_scores": [],
            "gt_scores": [],
            "gt_actions": [],
            "all_judge_scores": [],
            "difficulty": [],
        }

    def update(self, key, value):
        if key not in self.metric_dict.keys():
            self.metric_dict[key] = []
        
        self.metric_dict[key].append(value)

    def get_metric_dict(self):
        return self.metric_dict


def evaluate(metric_dict,
             is_train,
             dataset_name,
             curr_epoch =-1
             ):
        metric_dict = metric_dict.get_metric_dict()

        if not is_train:
            #if var is available use that for uncertainty else use logits
            plot_rejection_curve(metric_dict)

        pred_scores = metric_dict["pred_scores"]
        true_scores = metric_dict["gt_scores"]
      
        pred_scores = np.concatenate([np.atleast_1d(x) for x in pred_scores])
        true_scores = np.concatenate([np.atleast_1d(x) for x in true_scores])

        min_true_score = true_scores.min()
        max_true_score = true_scores.max()

        rho, p = stats.spearmanr(pred_scores, true_scores)
        L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
        RL2 = np.power((pred_scores - true_scores) / (max_true_score - min_true_score), 2).sum() / true_scores.shape[0]

        if dataset_name == "cat101":
            accuracy = (np.round(pred_scores) == true_scores).mean()
            print(f"Epoch {curr_epoch} Accuracy: {accuracy}")

        return rho*100, L2, RL2*100

def plot_rejection_curve(metric_dict):
    if "global_sqrt_var_emb" in metric_dict.keys():
        global_sqrt_var_emb = metric_dict["global_sqrt_var_emb"]
        global_sqrt_var_emb = np.concatenate(global_sqrt_var_emb)
        mean_sqrt_var = hmean(global_sqrt_var_emb, axis = 1)
        uncertainties = mean_sqrt_var
    else:
        return
    
    #matplotlib.use('module://drawilleplot')
    pred_scores = metric_dict["pred_scores"]
    true_scores = metric_dict["gt_scores"]
    
    pred_scores = np.concatenate([np.atleast_1d(x) for x in pred_scores])
    true_scores = np.concatenate([np.atleast_1d(x) for x in true_scores])

    y_mae, bins = rejection_plot(pred_scores, true_scores, uncertainties)

    print("Calibration Kendall Tau (MAE): {:0.4f}".format(kendalltau(y_mae, bins)[0]))



def rejection_plot(preds, gts, uncertainty, num_bins=11):
    all_mae = []
    bins = np.linspace(0, 100, num_bins)
    #[  0.    11.11  22.22  33.33  44.44  55.56  66.67  77.78  88.89 100.  ]
    
    conf_sort_idx = np.argsort(uncertainty)
    uncertainty = uncertainty[conf_sort_idx]

    preds = preds[conf_sort_idx]
    gts = gts[conf_sort_idx]
    
    for bin_num in range(bins.shape[0]-1):
        bin_low = np.percentile(uncertainty, bins[bin_num])
        
        if bin_num != bins.shape[0]-2:
            bin_high = np.percentile(uncertainty, bins[bin_num+1])
        else:
            bin_high = 10000000000000000
        bin_idxs = np.where((uncertainty>=bin_low) & (uncertainty<bin_high))[0]

        bin_preds = preds[bin_idxs]

        bin_gts = gts[bin_idxs]

        bin_mae = np.absolute(bin_preds - bin_gts).mean()

        all_mae.append(bin_mae)

    return all_mae, bins[:-1]