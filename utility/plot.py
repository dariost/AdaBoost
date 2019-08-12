import json
import matplotlib.pyplot as plt

if __name__ == "__main__"
    d = json.loads(open("result.json", "r").read())

    adatrain = list()
    bagtrain = list()
    adatest = list()
    bagtest = list()
    for i in range(len(d)):
        adatrain.append(d[i]['AdaBoost']['TrainingError'])
        bagtrain.append(d[i]['Bagging']['TrainingError'])
        adatest.append(d[i]['AdaBoost']['TestError'])
        bagtest.append(d[i]['Bagging']['TestError'])

    plt.plot([i for i in range(1, len(d) + 1)], bagtest, ls='-', c='lightblue')
    plt.plot([i for i in range(1, len(d) + 1)], bagtrain, ls='-', c='blue')
    plt.plot([i for i in range(1, len(d) + 1)], adatest, ls='-', c='pink')
    plt.plot([i for i in range(1, len(d) + 1)], adatrain, ls='-', c='red')

    plt.show()
