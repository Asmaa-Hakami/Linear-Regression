import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

#Load the data from CSV file.
data = pd.read_csv("insurance.csv")

X = data.iloc[:,:-1] #Features columns
Y = data.iloc[:,6] #Lables column

#split the dataset into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state = 0)

#Check if there is any missing value.
print(data.isnull().sum()) # The output is zero, so there is no missing value.

# Take a copy of the data to be converted to numerical
data_copy_test = X_test.copy()
data_copy_train = X_train.copy()

# Convert categorical values to numerical
lb_make = LabelEncoder()
data_copy_test['sex'] = lb_make.fit_transform(data_copy_test['sex'])
data_copy_test['smoker'] = lb_make.fit_transform(data_copy_test['smoker'])
data_copy_test['region'] = lb_make.fit_transform(data_copy_test['region'])

data_copy_train['sex'] = lb_make.fit_transform(data_copy_train['sex'])
data_copy_train['smoker'] = lb_make.fit_transform(data_copy_train['smoker'])
data_copy_train['region'] = lb_make.fit_transform(data_copy_train['region'])

# Scale the data 
scaler = StandardScaler()
data_copy_train = scaler.fit_transform(data_copy_train)
data_copy_test = scaler.fit_transform(data_copy_test)

#----------------------------gradient_descent----------------------------
def gradient_descent(x, y, theta, iterations, alpha):
    all_costs = [] # to save the cost for each iteration
    past_thetas = theta
    m = len(y)
    
    for i in range(iterations):
        squared_error = []
        # This for loop to take single example
        for j in range (len(x)): 
            Y_prediction = past_thetas[0] + (np.dot(x[j],past_thetas[1:7]))
            if y.values[j]!= Y_prediction: # If the condition is correct! then we need to update all theta
                error = y.values[j] - Y_prediction
                squared_error.append(np.power(error,2))
                past_thetas[0] = past_thetas[0] - (alpha * 2/m * 1 * error) # Update theta 0
                #Update the rest of theta (1 to 6)
                for t in range (len(x[j])):
                    past_thetas[t+1] = past_thetas[t+1] - (alpha * (2/m) * x[j][t] * error)

        cost = 1/(m) * sum(squared_error)
        all_costs.append(cost)
        plt.title('Cost Function J')
        plt.xlabel('No. of iterations')
        plt.ylabel('Cost')
        plt.plot(all_costs)
        plt.show()
        
    return past_thetas, all_costs
#----------------------------End of the method----------------------------


all_theta= [0.5,0.3,-0.5,0.2,0.5,0.3,0.5] # Initialize theta
past_thetas, past_costs = gradient_descent(data_copy_train, y_train, all_theta, 10, 0.01) # Call the method to train the model

#----------------------------Test the model----------------------------
Y_pred = []
error = []
for i in range (len(data_copy_test)):
    Y_prediction = past_thetas[0] + (np.dot(data_copy_test[i],past_thetas[1:7]))
    e = y_test.values[i] - Y_prediction
    error.append(np.power(e, 2))
    Y_pred.append(Y_prediction)


MSE = 1/len(y_train) * sum(error)
print("MSE is",MSE)
most_sig = 0

# Find and report the most significant features in the regression model. 
fs = SelectKBest(score_func=f_regression, k='all')
fs.fit(data_copy_train, y_train)
for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))

