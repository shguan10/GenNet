import pickle as pk
import scipy as sp
import numpy as np
import pdb

import matplotlib
import matplotlib.pyplot as plt

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
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

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
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
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
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
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def read():
  with open("data/linearcorr_allcorr_partial.pk","rb") as f:
    data = pk.load(f)
  c1sweep = [20,40,60,80,100]
  c2sweep = [0,1,2,4,8,16,32,64,128]

  frobNormSweep = np.linspace(0,2,num=10,endpoint=False)
  numits=1000
  # pdb.set_trace()
  
  names,data = zip(*data)
  names = np.array(names).reshape((len(c1sweep),len(c2sweep),len(frobNormSweep)))
  data = np.array(data).reshape((len(c1sweep),len(c2sweep),len(frobNormSweep),numits))
  mus = data.mean(axis=3)
  stds = data.std(axis=3)

  vmin = mus.min()
  vmax = mus.max()

  # graph the c1 c2 grid
  c1s = ["r1_"+str(s) for s in c1sweep]
  c2s = ["r2_"+str(s) for s in c2sweep]

  for ind,frobNorm in enumerate(frobNormSweep):
    # plot the mus
    tograph = mus[:,:,ind]

    fig, ax = plt.subplots()
    im, _ = heatmap(tograph, c1s, c2s,
                    cmap="PuOr", vmin=vmin, vmax=vmax,
                    cbarlabel="Mean of Benefit")

    def func(x, pos):
        return "{:.2f}".format(x).replace("0.", ".").replace("1.00", "")

    annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=7)

    ax.set_title("Mean of Benefit out of 1000 trials, frobNorm "+str(frobNorm))
    fig.tight_layout()
    # plt.show()
    plt.savefig("figs/partial_linear_mean_norm"+str(frobNorm)+".jpg")
    plt.clf()    

    # plot the stds
    tograph = stds[:,:,ind]

    fig, ax = plt.subplots()
    im, _ = heatmap(tograph, c1s, c2s,
                    cmap="PuOr", vmin=vmin, vmax=vmax,
                    cbarlabel="Std of Benefit")

    def func(x, pos):
        return "{:.2f}".format(x).replace("0.", ".").replace("1.00", "")

    annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=7)

    ax.set_title("Std of Benefit out of 1000 trials, frobNorm "+str(frobNorm))
    fig.tight_layout()
    # plt.show()
    plt.savefig("figs/partial_linear_std_norm"+str(frobNorm)+".jpg")
    plt.clf()

def go():
  frobNorm = "2.0"
  translate = True
  fname ="data/frobInc_100_10_100_10_4_"+frobNorm+"_"+str(translate)+".pk"
  with open(fname,"rb") as f:
    data = pk.load(f)
    data = np.array(data)
  # pdb.set_trace()
  # plt.scatter(0,1,alpha=0.3)
  mu = data[:,0].mean()
  sigma = data[:,0].std()
  plt.scatter(data[:,0],np.zeros(data[:,0].shape),alpha=0.3,label="the benefit measured from one trial\nmu="+str(mu)+", s="+str(sigma))
  plt.xlabel("value of benefit")
  plt.legend()
  plt.title("scatterplot of frobNorm "+frobNorm+(", no trans" if not translate else ""))

  plt.show()

def go2():
  mus = np.zeros((8,3))
  sigmas = np.zeros((8,3))

  for r2 in range(8):
    for frobInc in range(3):
      frobNorm = [0,0.4,0.8][frobInc]
      lenM1=100
      lenM2=10
      lenH = 100
      actualH = 10
      numHlayers = 1
      translate=True
      partialTranslate=True
  

      corrN=[lenM1,r2]
      name="frobInc_partial_"+str(lenM1)+"_"+str(lenM2)+"_"+str(lenH)+"_"+str(actualH)+"_"+str(numHlayers)+"_"+str(frobNorm)+"_"+str(translate)+"_"+str(corrN)

      print(name)
      with open("data/"+name+".pk","rb") as f:
        d = pk.load(f)
        d = np.array(d)
      numtrials = d.shape[0]
      r = d[:,0]/d[:,1]

      mu = r.mean()
      sigma = r.std()
      mus[r2,frobInc]=mu
      sigmas[r2,frobInc]=sigma

      plt.scatter(r,np.zeros(r.shape),alpha=0.3,label="the benefit measured from one trial\nmu="+str(mu)+", s="+str(sigma))
      plt.xlabel("value of benefit")
      plt.legend()
      plt.title("scatterplot of "+name)

      # plt.show()
  plt.clf()

  # plot the mus
  tograph = mus

  r2sweep = list(range(8))
  frobSweep = [0,0.4,0.8]

  vmin = mus.min()
  vmax = mus.max()

  r2s = ["r2 "+str(r) for r in r2sweep]
  frobs = ["frobNorm "+str(r) for r in frobSweep]

  fig, ax = plt.subplots()
  im, _ = heatmap(tograph, r2s, frobs,
                  cmap="PuOr", vmin=vmin, vmax=vmax,
                  cbarlabel="Mean of (benefit/control)")

  def func(x, pos):
      return "{:.2f}".format(x).replace("0.", ".").replace("1.00", "")

  annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=7)

  ax.set_title("Mean of (benefit/control) out of "+str(numtrials)+" trials")
  fig.tight_layout()
  plt.show()
  plt.savefig("figs/partial_deep_mean_ratio"+".jpg")
  plt.clf()    

  # plot the stds
  tograph = sigmas

  fig, ax = plt.subplots()
  im, _ = heatmap(tograph, r2s, frobs,
                  cmap="PuOr", vmin=vmin, vmax=vmax,
                  cbarlabel="Std of (benefit/ratio)")

  def func(x, pos):
      return "{:.2f}".format(x).replace("0.", ".").replace("1.00", "")

  annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=7)

  ax.set_title("Std of (benefit/control) out of "+str(numtrials)+" trials")
  fig.tight_layout()
  plt.show()
  plt.savefig("figs/partial_deep_std_ratio"+".jpg")
  plt.clf()

if __name__ == '__main__':
  # read()
  # go()
  go2()