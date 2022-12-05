import matplotlib
import matplotlib as mpl
import numpy as np
from utils import read_param2per
import matplotlib.pyplot as plt

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, format='%.2f', **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), col_labels)
    ax.set_yticks(np.arange(data.shape[0]), row_labels)
    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
    #          rotation_mode="anchor")

    # Turn spines off and create white grid.
    # ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+0.5)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+0.5)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, round(data[i, j],2), None, **kw)
            texts.append(text)

    return texts

def plot_fusion_heatmap():
    
    pass

if __name__ == "__main__":
    param2per1 = read_param2per("./result/param2performance_weibo_min_gcngat.txt")
    param2per2 = read_param2per("./result/param2performance_weibo_syn_gcngat.txt")
    param2per3 = read_param2per("./result/param2performance_amazon_photo_min_gcngat.txt")
    param2per4 = read_param2per("./result/param2performance_amazon_photo_syn_gcngat.txt")
    data1 = param2per1[:,2].reshape(11,10)
    data2 = param2per2[:,2].reshape(11,10)
    data3 = param2per3[:,2].reshape(11,10)
    data4 = param2per4[:,2].reshape(11,10)
    # np.random.seed(19680801)
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(7, 6))

    # Replicate the above example with a different font size and colormap.

    # im, _ = heatmap(harvest, vegetables, farmers, ax=ax,
    #                 cmap="Wistia", cbarlabel="harvest [t/year]")
    # annotate_heatmap(im, valfmt="{x:.1f}", size=7)

    # Create some new data, give further arguments to imshow (vmin),
    # use an integer format on the annotations and provide some colors.

    # data = np.random.randint(2, 100, size=(7, 7))
    y = ["{}".format(i) for i in range(11)]
    x = ["{}".format(i) for i in range(1,11)]
    im, _ = heatmap(data1, y, x, ax=ax1, vmin=0.93,vmax=0.98,
                    cmap="inferno_r", cbarlabel="")
    annotate_heatmap(im, valfmt="{x:d}", size=5, threshold=0.95, textcolors=("green","red"))

    im, _ = heatmap(data2, y, x, ax=ax2, vmin=0.65,vmax=0.69,
                    cmap="inferno_r", cbarlabel="")
    annotate_heatmap(im, valfmt="{x:d}", size=5, threshold=0.65, textcolors=("green","red"))
    
    im, _ = heatmap(data3, y, x, ax=ax3, vmin=0.92,vmax=0.95,
                    cmap="inferno_r", cbarlabel="")
    annotate_heatmap(im, valfmt="{x:d}", size=5, threshold=0.92, textcolors=("green","red"))

    im, _ = heatmap(data4, y, x, ax=ax4, vmin=0.61,vmax=0.63,
                    cmap="inferno_r", cbarlabel="")
    annotate_heatmap(im, valfmt="{x:d}", size=5, threshold=0.61, textcolors=("green","red"))
    plt.suptitle("Detection rate")
    plt.savefig(str('./result/fusion.eps'), bbox_inches='tight', format='eps')
    plt.show()

