import csv
from math import e, pi, ceil, log
import matplotlib.pyplot as plt


def pdf(x, mean, sd):
    return (e ** (-((x - mean) ** 2) / (2 * (sd ** 2)))) \
           / (((2 * pi) ** 0.5) * sd)


def standard_deviation(data):
    squared_data = [x ** 2 for x in data]
    mean_squared_data = sum(squared_data) / \
                        len(squared_data)
    mean_data = sum(data) / len(data)
    variance = mean_squared_data - (mean_data ** 2)
    return variance ** 0.5


def avg(data):
    return sum(data) / len(data)


datasets = ['../datasets/Wimbledon-men-2013.csv',
            '../datasets/Wimbledon-women-2013.csv',
            '../datasets/AusOpen-men-2013.csv',
            '../datasets/AusOpen-women-2013.csv',
            '../datasets/FrenchOpen-men-2013.csv',
            '../datasets/FrenchOpen-women-2013.csv',
            '../datasets/USOpen-men-2013.csv',
            '../datasets/USOpen-women-2013.csv']

dataset_accuracy = []
for filename in datasets:
    parameter1 = 'BPW.1'
    parameter2 = 'BPW.2'
    par1, par2, result = [], [], []
    with open(filename, 'r') as data:

        reader = csv.reader(data)  # Create CSV Reader
        for row in reader: header = row; break  # Get headings of table.

        par1_index = header.index(parameter1)  # Get column number of Parameter 1
        par2_index = header.index(parameter2)  # Get column number of Parameter 2
        result_index = header.index('Result')  # Get column number of Result

        for row in reader:
            if row[par1_index] != '' and row[par2_index] != '':
                par1.append(row[par1_index])  # Add Parameter 1 data to x1 array.
                par2.append(row[par2_index])  # Add Parameter 2 data to x2 array.
                result.append(row[result_index])  # Add Result data to y array. Extracting sample data from data set.
    if len(par1) == 0 or len(par2) == 0:
        continue
    accuracy = []
    transformation = 0.01
    for x in range(150):
        split = ceil(len(par1) * 0.6) - 1
        x1 = [int(x) ** transformation for x in par1[0:split]]
        x2 = [int(x) ** transformation for x in par2[0:split]]
        y = [int(y) for y in result[0:split]]
        win_x1, win_x2 = [], []
        loss_x1, loss_x2 = [], []
        for index in range(len(x1)):  # Filter out wins and losses.
            if y[index] == 1:
                win_x1.append(x1[index])
                win_x2.append(x2[index])
            else:
                loss_x1.append(x1[index])
                loss_x2.append(x2[index])  # Filtering out wins and losses.

        win_x1_m = avg(win_x1)
        win_x2_m = avg(win_x2)
        loss_x1_m = avg(loss_x1)
        loss_x2_m = avg(loss_x2)

        win_x1_std = standard_deviation(win_x1)
        win_x2_std = standard_deviation(win_x2)
        loss_x1_std = standard_deviation(loss_x1)
        loss_x2_std = standard_deviation(loss_x2)
        p0 = len(loss_x1) / (len(win_x1) + len(loss_x1))
        p1 = len(win_x1) / (len(win_x1) + len(loss_x1))

        predictions = []
        x1_test = [int(x) ** transformation for x in par1[split + 1::]]
        x2_test = [int(x) ** transformation for x in par2[split + 1::]]
        y_test = [int(y) ** transformation for y in result[split + 1::]]

        for item1, item2 in zip(x1_test, x2_test):
            c0 = pdf(item1, loss_x1_m, loss_x1_std) \
                 * pdf(item2, loss_x2_m, loss_x2_std) \
                 * p0
            c1 = pdf(item1, win_x1_m, win_x1_std) \
                 * pdf(item2, win_x2_m, win_x2_std) \
                 * p1
            if c1 > c0:
                predictions.append(1)
            else:
                predictions.append(0)
        correct = 0
        for i, j in zip(predictions, y_test):
            if i == j:
                correct += 1
        accuracy.append(correct / len(predictions))

        transformation += 0.01

    dataset_accuracy.append([filename, max(accuracy)])
    fig, ax = plt.subplots()
    fig.suptitle("{} vs {} for {}".format(parameter1, parameter2, filename[12::]))
    ax.plot(win_x1, win_x2, 'ro')
    ax.plot(loss_x1, loss_x2, 'bo')
    plt.xlabel(parameter1)
    plt.ylabel(parameter2)

accuracy_average = []
for acc in dataset_accuracy:
    print("{} : {}".format(acc[0], acc[1]))
    accuracy_average.append(acc[1])

average = sum(accuracy_average) / len(accuracy_average)
print(average)
# plt.show()
