# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: 
RegisterNumber:  
*/
```
import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

X = np.array([[5.1, 1.4], [4.9, 1.4], [4.7, 1.3],[6.0, 4.5], [6.5, 4.6], [6.3, 4.7],[7.1, 5.9], [6.9, 5.7], [7.6, 6.6]])

y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

sgd = SGDClassifier(max_iter=1000, tol=1e-3, learning_rate='optimal')

sgd.fit(X_train, y_train)

y_pred = sgd.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Predicted Species:", y_pred)

plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', marker='x', label='Predicted')

plt.xlabel('Sepal Length')

plt.ylabel('Petal Length')

plt.title('Prediction of Iris Species using SGD Classifier')

plt.legend()

plt.show()

## Output:
<img src="ex7 output 1.png" alt="Output" width="500">

<img src="ex7 output 2.png" alt="Output" width="500">


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
