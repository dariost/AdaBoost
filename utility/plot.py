#!/usr/bin/env python3
import json
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg


if __name__ == "__main__":
    assert len(sys.argv) == 2
    filename = sys.argv[1]
    d = json.loads(open(filename, "r").read())

    adatrain = list()
    bagtrain = list()
    adatest = list()
    bagtest = list()

    for i in range(len(d)):
        adatrain.append(d[i]['AdaBoost']['TrainingError'])
        adatest.append(d[i]['AdaBoost']['TestError'])
        bagtrain.append(d[i]['Bagging']['TrainingError'])
        bagtest.append(d[i]['Bagging']['TestError'])

    fig, ax = plt.subplots(figsize=(15, 20))

    ax.set_title("AdaBoost and Bagging training and test errors")
    bg_te = mpatches.Patch(color='#63f7a8', label='Bagging test error')
    bg_tr = mpatches.Patch(color='#00c91e', label='Bagging training error')
    ab_te = mpatches.Patch(color='#29c9ff', label='AdaBoost test error')
    ab_tr = mpatches.Patch(color='#196dfc', label='AdaBoost training error')
    ax.legend(handles=[bg_te, bg_tr, ab_te, ab_tr])
    ax.set_xlabel("Epoches")
    ax.set_ylabel("Errors")
    ax.grid(ls=":")


    # plotting test error -> training error of bagging
    ax.plot([i for i in range(1, len(d) + 1)], bagtest, c='#63f7a8')
    ax.plot([i for i in range(1, len(d) + 1)], bagtrain, c='#00c91e')
    # plotting test error -> training error of adaboost
    ax.plot([i for i in range(1, len(d) + 1)], adatest, c='#29c9ff')
    ax.plot([i for i in range(1, len(d) + 1)], adatrain, c='#196dfc')

    plt.show()
    outputname = "Report" + filename.split("_")[1].split(".")[0] + ".eps"
    #fig.savefig(outputname, format="eps")
    print("Saved figure {}".format(outputname))
