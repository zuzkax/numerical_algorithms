import numpy as np
import matplotlib.pyplot as plt

file_path = "139374.dat"
data = np.loadtxt(file_path)

x = data[:, 0]
y = data[:, 1]
f = data[:, 2]

def avg(data):
    count = 0
    sum = 0
    for i in data:
        sum += i
        count += 1

    return sum/count

def median(data):
    sorted_data = sorted(data)
    n = len(data)

    if n % 2 == 1:
        return sorted_data[n // 2]
    else:
        r = n // 2
        l = r - 1
        return (sorted_data[l] + sorted_data[r]) / 2

def standard_deviation(data):
    n =  len(data)

    if n <= 1:
        return 0

    mean = avg(data)
    summ = sum((x - mean) ** 2 for x in data)
    var = summ/(n-1)

    return np.sqrt(var)

def zad2(y,f):
    y_const = np.unique(y)

    avg_list = []
    median_list = []
    std_dev_list = []

    for i in y_const:
        listf = []
        index = 0
        while index < len(y):
            if y[index] == i:
                listf.append(f[index])
            index += 1
        print(f"average value for y = {i}: {avg(listf)}")
        print(f"median for y =  {i}: {median(listf)}")
        print(f"standard deviation for y = {i}: {standard_deviation(listf)}")

        avg_list.append(avg(listf))
        median_list.append(median(listf))
        std_dev_list.append(standard_deviation(listf))

    bar_width = 0.25
    index = np.arange(len(y_const))
    avgch = plt.bar(index, avg_list, bar_width, label='Średnia')
    medch = plt.bar(index + bar_width, median_list, bar_width, label='Mediana')
    stddevch = plt.bar(index + 2 * bar_width, std_dev_list, bar_width, label='Odchylenie standardowe')

    plt.xlabel('y')
    plt.ylabel('F(x,y)')
    plt.title('Mediana, średnia i odchylenie standardowe dla każdej wartości y')
    plt.legend()
    plt.show()

zad2(y,f)
