import csv
from math import log, ceil
import numpy as np
import matplotlib.pyplot as plt

g = 2  # Number of groups in y.

parameter1 = 'UFE.1'
parameter2 = 'UFE.2'
dataset_x = []
dataset_y = []
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
    with open(filename, 'r') as data:

        reader = csv.reader(data) # Create CSV Reader
        for row in reader: header = row; break # Get headings of table.

        par1_index = header.index(parameter1) # Get column number of Parameter 1
        par2_index = header.index(parameter2) # Get column number of Parameter 2
        result_index = header.index('Result') # Get column number of Result

        for row in reader:
            if row[par1_index] != '' and row[par2_index] != '':
                dataset_x.append([int(row[par1_index]), int(row[par2_index])]) #Add Parameter 1 data to x1 array.
                dataset_y.append(int(row[result_index])) #Add Result data to y array. Extracting sample data from data
                # set.

    if len(dataset_x) == 0 or len(dataset_y) == 0:
        continue

    accuracy = []
    transformation = 0.1
    for repeat in range(500):
        split = ceil(len(dataset_x) * 0.6) - 1
        training_x =[[i[0]**transformation, i[1]**transformation] for i in dataset_x[0:split]]
        training_y = dataset_y[0:split]
        x = np.array(training_x)
        y = np.array(training_y)
        x1 = []
        x2 = []
        for num, yi in enumerate(y):
            if yi == 0:
                x1.append(x[num]) # 1 = class 0 ; 2 =  class 1
            else:
                x2.append(x[num])

        u1 = np.mean(x1, axis=0)
        u2 = np.mean(x2, axis=0)
        u = np.mean(x, axis=0)

        x01 = x1 - u
        x02 = x2 - u
        c1 = np.dot(x01.transpose(), x01) / x01.shape[0]
        c2 = np.dot(x02.transpose(), x02) / x02.shape[0]

        # Finding discriminant lines
        cw = c1 + c2
        cw_inverse = np.linalg.inv(cw)
        cb = np.dot((u1-u2),(u1-u2).transpose())
        cw_cb = np.dot(cw_inverse, cb)
        w, v = np.linalg.eig(cw_cb)

        eigenvector = v[w.tolist().index((max(w)))]
        slope = eigenvector[1] / eigenvector[0]

        w1, w2 = len(x01) / len(x), len(x02) / len(x)
        C = [[i1[0] * w1
              + i2[0] * w2,
              i1[1] * w1
              + i2[1] * w2] for i1, i2 in zip(c1, c2)]
        C = np.array(C)
        C_inverse = np.linalg.inv(C)
        p = [w1, w2]

        predictions = []
        testing_x = np.array([[i[0]**transformation, i[1]**transformation] for i in dataset_x[split+1::]])
        testing_y = np.array(dataset_y[split+1::])
        for item in testing_x:
            v1_f1 = np.dot(u1, C_inverse)
            v1_f1 = np.dot(v1_f1, item.transpose())
            v2_f1 = 0.5 * np.dot(u1, C_inverse)
            v2_f1 = np.dot(v2_f1, u1.transpose())
            v3_f1 = log(p[0])
            f1 = v1_f1 - v2_f1 + v3_f1

            v1_f2 = np.dot(u2, C_inverse)
            v1_f2 = np.dot(v1_f2, item.transpose())
            v2_f2 = 0.5 * np.dot(u2, C_inverse)
            v2_f2 = np.dot(v2_f2, u2.transpose())
            v3_f2 = log(p[1])
            f2 = v1_f2 - v2_f2 + v3_f2
            if f1 > f2:
                predictions.append(0)
            else:
                predictions.append(1)

        correct = 0
        for item1, item2 in zip(predictions, testing_y):
            if item1 == item2:
                correct += 1

        accuracy.append(correct / len(predictions))
        transformation += 0.001
    dataset_accuracy.append([filename, max(accuracy)])
    fig, ax = plt.subplots()

    # Discriminant lines
    x_0 = -5
    x_1 = 100
    y_0 = 100
    y_1 = slope*(x_1-x_0)
    ax.scatter([x_0, x_1], [y_0, y_1], marker='^', s=20, c='r')
    ax.plot([x_0, x_1], [y_0, y_1], c='g')

    # Data
    ax.plot([z[0] for z in x1], [z[1] for z in x1], 'ro')
    ax.plot([z[0] for z in x2], [z[1] for z in x2], 'bo')
accuracy_average = []
for acc in dataset_accuracy:
    print("{} : {}".format(acc[0], acc[1]))
    accuracy_average.append(acc[1])

average = sum(accuracy_average) / len(accuracy_average)
print("\n Average accuracy: {}".format(average))