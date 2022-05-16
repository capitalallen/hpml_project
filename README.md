# HPML Project 
## Description of the project

### Background 

Most AutoML tools follow a common 3-stage pipline (Figure 1). There is no one AutoML tool that can clearly outperform every other tool. 
![3-stage pipeline](/images/2.png)

Most AutoML tools follow a common 3 stage pipline (Figure 1). There is no one AutoML tool that can clearly outperform every other tool. 
### Challenges 
Althrough each AutoML library is trying to solve same problem - Automate ML & DL training, they each have dramtically different APIs and ways to accomplish this goal.

There is a learning curve for mastering those AutoML libraries and each of those libraries has both pros and cons when solving different types of ML tasks. 

As the result, users have to spend lots of time learning and trying different AutoML tools, which jeopardizes the goal of saving time that those AutoML tools are trying to achieve.   

### Solution
After examing the performance of some AutoML tools on many datasets, we build this project that focuses on building a library aggregating all popular AutoML Tools and presents a uniform API layer that allows users to easily switch between different AutoML tools and save lots of time on learning each individual tool. 
![Design](/images/1.png)


## Description of the repository and code structure
### Description of the repository
This repo includes the implemention of 3 different AutoML libraries (AutoGoal, Auto-keras, H2o) to solve binary classification and image cliassifcation on certain datasets 

### Code structure
Source code is in code folder.

We applied Adapter design pattern when designing the structure of the project, which allows us to allows different AutoML tools with incompatible interfaces to collaborate. and add more in the future. 
client.py in src folder acts as the adapter wraps AutoML libraries to hide the complexity of conversion happening behind the scenes. 

auto_goal, auto_h2o and auto_keras folders include the use of AutoML tools to solve various types of ML problems. 

## Example commands to execute the code
### AutoGoal for Image classification on CIFIAR10 dataset
    ```
    from src.client import Client 
    library='auto_goal'
    data_path='./data'
    client = Client(library)
    client.execute(data_path)
    ```
### Auto-keras for Image classification on CIFIAR10 dataset
    ```
    from src.client import Client 
    library='auto_keras'
    type='imageClassifier'
    client = Client(library)
    client.execute(data_path,type)
    ```
### Auto-keras for binary classification
    ```
    from src.client import Client 
    library='auto_keras'
    type='binary_classifier'
    train_path='https://s3.amazonaws.com/erin-data/higgs/higgs_train_10k.csv'
    test_path='https://s3.amazonaws.com/erin-data/higgs/higgs_test_5k.csv'
    client = Client(library)
    client.execute(data_path,type,train_path,test_path)
    ```
### H2O for binary classification
    ```
    from src.client import Client 
    library='h2o'
    label_column='response'
    train_path='https://s3.amazonaws.com/erin-data/higgs/higgs_train_10k.csv'
    test_path='https://s3.amazonaws.com/erin-data/higgs/higgs_test_5k.csv'
    client = Client(library)
    client.execute(label_column,train_path,test_path)
    ```

## Results and observations  
Evaluate AutoGoal and Auto-Keras on Cifiar 10 dataset 
- AutoGoal 
    - Accuracy: 96.5% in 85min 
- Auto-kerasAccuracy: 
    - 92% in 90min
Evalutae H2O AutoML vs Auto-keras on binary classification 
- H2O 
    - accuracy: 80.8% in 13.4min 
- Auto-Keras
    - accuracy: 65% in 6min

