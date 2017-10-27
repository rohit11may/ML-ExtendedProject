from math import log
import csv
import matplotlib.pyplot as plt

def standard_deviation(data):
    #Find variance.
    squared_data = [x**2 for x in data]
    mean_squared_data = sum(squared_data) / len(squared_data)
    mean_data = sum(data) / len(data)
    variance = mean_squared_data - (mean_data ** 2)
    return variance**0.5

def avg(data):
    return sum(data) / len(data)

def standardize(data, m, s):
    standardized_data = [(x - m)/s for x in data]
    return standardized_data

def sqr_diff(data):
    m = avg(data)
    squared_data_diff = [(x - m)**2 for x in data]
    return sum(squared_data_diff)

parameter1 = 'BPW.1'
parameter2 = 'BPW.2'
par1, par2, result = [], [], []

with open('../datasets/AusOpen-men-2013.csv', 'r') as data:

    reader = csv.reader(data) # Create CSV Reader
    for row in reader: header = row; break # Get headings of table.

    par1_index = header.index(parameter1) # Get column number of Parameter 1
    par2_index = header.index(parameter2) # Get column number of Parameter 2
    result_index = header.index('Result') # Get column number of Result

    for row in reader:
        par1.append(row[par1_index]) #Add Parameter 1 data to x1 array.
        par2.append(row[par2_index]) #Add Parameter 2 data to x2 array.
        result.append(row[result_index]) #Add Result data to y array. Extracting sample data from data set.


ratio = [int(par1[num])/max(1,int(par2[num])) for num, item in enumerate(par1)]
y = [int(y) for y in result]
win_ratio, loss_ratio = [], []

for num, item in enumerate(ratio): #Filter out wins and losses.
    if y[num] == 1:
        win_ratio.append(ratio[num])
    else:
        loss_ratio.append(ratio[num]) # Filtering out wins and losses.

win_ratio = win_ratio[0:len(win_ratio)- 2] #Sort and remove outliers
loss_ratio = loss_ratio[0:len(loss_ratio)- 2]
all_data = win_ratio[:] # Collect all data
for x in loss_ratio: all_data.append(x)

mean, std = avg(all_data), standard_deviation(all_data) # Calculate standard deviation
# win_ratio = standardize(win_ratio, mean, std) # Standardize win ratio
# loss_ratio = standardize(loss_ratio, mean, std) # Standardize loss ratio
# Try standardizing separately!


variance = (1/((len(win_ratio) + len(loss_ratio) - 2)) * (sqr_diff(win_ratio) + sqr_diff(loss_ratio)))

prob_win = len(win_ratio) / (len(win_ratio) + len(loss_ratio))
prob_loss = len(loss_ratio) / (len(win_ratio) + len(loss_ratio))
mean_win = sum(win_ratio) / len(win_ratio)
mean_loss = sum(loss_ratio) / len(loss_ratio)

discWin = [] # Win discriminant
discLoss = [] # Loss discriminant
for element in all_data:
    # Use discriminant function. x * (mean / variance) - (mean^2 / 2*variance) + ln(prob)
    discWin.append((element * (mean_win / variance)) - ((mean_win ** 2) / (variance*2)) + log(prob_win))
    discLoss.append((element * (mean_loss / variance)) - ((mean_loss ** 2) / (variance*2)) + log(prob_loss))

predictions = []
for win, loss in zip(discWin, discLoss):
    if win > loss:
        predictions.append(1)
    else:
        predictions.append(0)

correct = 0
for num, prediction in enumerate(predictions):
    if num <= (len(win_ratio)-1):
        if prediction == 1:
            correct += 1
    else:
        if prediction == 0:
            correct += 1

accuracy = correct / len(predictions)
print(accuracy)

# plt.plot([num for num, item in enumerate(win_ratio)], win_ratio, 'ro') #Plot wins
# plt.plot([num for num, item in enumerate(loss_ratio)], loss_ratio,'bo') #Plot losses

plt.plot( )
plt.xlabel('Series')
plt.ylabel('Ratio of {}:{}'.format(parameter1, parameter2))
plt.show() # Graphing Data to be used in Logistic Regression.