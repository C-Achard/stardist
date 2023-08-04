from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

def plot_performance(taus, stats, name, metric="IoU"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(name)
    for m in ('precision', 'recall', 'accuracy', 'f1', 'mean_true_score', 'mean_matched_score', 'panoptic_quality'):
        ax1.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
    ax1.set_xlabel(f'{metric}'+ r' threshold $\tau$')
    ax1.set_ylabel('Metric value')
    ax1.grid()
    ax1.legend()

    for m in ('fp', 'tp', 'fn'):
        ax2.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
    ax2.set_xlabel(f'{metric}'+r' threshold $\tau$')
    ax2.set_ylabel('Number #')
    ax2.grid()
    ax2.legend()

    return fig

def plot_stat_comparison(taus, stats_list, model_names, stat='f1', metric="IoU"):
    """Compare one stat for several models on a single plot"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for i, stats in enumerate(stats_list):
        ax.plot(taus, [s._asdict()[stat] for s in stats], '.-', lw=2, label=model_names[i])
    ax.set_xlabel(f'{metric}' + r' threshold $\tau$')
    ax.set_ylabel(stat)
    ax.grid()
    ax.legend()

if __name__=="__main__":
    path_images = Path.home() / "Desktop/Code/CELLSEG_BENCHMARK"
    gt_path = path_images / "TPH2_mesospim/test_data/labels/isotropic_visual.tif"
    results = path_images / "RESULTS/full data/instance"
    predictions = [imread(str(p)) for p in sorted(results.glob("*.tif"))]
    pred_names = [p.name for p in sorted(results.glob("*.tif"))]

    taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    Y_val = imread(str(gt_path))
    for i, p in enumerate(predictions):
        Y_val_pred = p
        print(f"Validating on {pred_names[i]}")
        stats = [matching_dataset(Y_val, Y_val_pred, thresh=t, show_progress=False) for t in tqdm(taus)]
        plot_performance(taus, stats, name=pred_names[i])
        plt.show()
        print("*"*20)

