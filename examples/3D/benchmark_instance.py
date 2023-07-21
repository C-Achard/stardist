from pathlib import path
from stardist.matching import matching_dataset

def plot_performance(taus, stats):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    for m in ('precision', 'recall', 'accuracy', 'f1', 'mean_true_score', 'mean_matched_score', 'panoptic_quality'):
        ax1.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
    ax1.set_xlabel(r'IoU threshold $\tau$')
    ax1.set_ylabel('Metric value')
    ax1.grid()
    ax1.legend()

    for m in ('fp', 'tp', 'fn'):
        ax2.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
    ax2.set_xlabel(r'IoU threshold $\tau$')
    ax2.set_ylabel('Number #')
    ax2.grid()
    ax2.legend()

    return fig

if __name__=="__main__":
    path_images = Path.home() / "Desktop/Code/CELLSEG_BENCHMARK/TPH2_DATA/visual_iso"
    gt_path = path_images / "labels/visual_labels.tif"
    

