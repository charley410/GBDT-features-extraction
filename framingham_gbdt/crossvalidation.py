from sklearn.model_selection import train_test_split

def spilt_cv(x,y):
    # data1 , data_rest = train_test_split(data, test_size = 9/10)
    # data2 , data_rest = train_test_split(data_rest, test_size = 8/9)
    data_x = {}
    data_y = {}
    x_rest = x
    y_rest = y
    num = {}
    for i in range(9):
        num[i] = '%.2f'%((9-i)/(10-i))
        data_x[i+1], x_rest, data_y[i+1],y_rest = train_test_split(x_rest, y_rest, test_size = float(num[i]))
    data_x[10] = x_rest
    data_y[10] = y_rest
    return data_x,data_y

from sklearn.datasets import load_iris
iris  = load_iris()
x = iris['data']
y = iris['target']
#
data_x ,data_y = spilt_cv(x,y)
print(data_x)
print(data_y)

