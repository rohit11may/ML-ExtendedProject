import csv
from math import e, ceil

datasets = ['../datasets/Wimbledon-men-2013.csv',
            '../datasets/Wimbledon-women-2013.csv',
            '../datasets/AusOpen-men-2013.csv',
            '../datasets/AusOpen-women-2013.csv',
            '../datasets/FrenchOpen-men-2013.csv',
            '../datasets/FrenchOpen-women-2013.csv',
            '../datasets/USOpen-men-2013.csv',
            '../datasets/USOpen-women-2013.csv']


def logistic_function(B0, B1, B2, x1, x2):
    probability = 1 / (1 + (e ** (-(B0 + (B1 * x1) + (B2 * x2)))))  # Calculate Prediction using Logistic Function
    return probability


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
    training_accuracy = []
    prediction_accuracy = []
    for x in range(1000):
        split = ceil(len(par1) * 0.6) - 1
        x1 = [int(x) for x in par1[0:split]]  # Apply transformation
        x2 = [int(x) for x in par2[0:split]]  # Apply transformation
        y = [int(y) for y in result[0:split]]

        B0, B1, B2 = 0, 0, 0
        alpha = 0.001
        bias = 1
        epochs = 10
        training_accuracy = []
        for repeat in range(epochs):
            total_correct = 0
            for index in range(len(x1)):
                prediction = logistic_function(B0, B1, B2, x1[index], x2[index])

                if prediction < 0.5:
                    sharp_prediction = 0
                else:
                    sharp_prediction = 1
                if sharp_prediction == y[index]:
                    total_correct += 1

                delta_coefficient = (alpha * (y[index] -  prediction) * prediction * (1 - prediction))

                # Update Coefficients.
                B0 += delta_coefficient * bias
                B1 += delta_coefficient * x1[index]
                B2 += delta_coefficient * x2[index]

            training_accuracy.append(((total_correct / len(x1)) * 100))

        predictions = []
        x1_test = [int(x) for x in par1[split + 1::]]
        x2_test = [int(x) for x in par2[split + 1::]]
        y_test = [int(y) for y in result[split + 1::]]
        for item1, item2 in zip(x1_test, x2_test):
            prediction = logistic_function(B0, B1, B2, item1, item2)
            # Calculate Accuracy
            if prediction < 0.5:
                predictions.append(0)
            else:
                predictions.append(1)
        correct = 0
        for i, j in zip(predictions, y_test):
            if i == j:
                correct += 1
        prediction_accuracy.append(correct / len(predictions))
        alpha += 0.0005
    dataset_accuracy.append([filename, max(prediction_accuracy)])

accuracy_average = []
for acc in dataset_accuracy:
    print("{} : {}".format(acc[0], acc[1]))
    accuracy_average.append(acc[1])

average = sum(accuracy_average) / len(accuracy_average)
print("\n Average accuracy: {}".format(average))
