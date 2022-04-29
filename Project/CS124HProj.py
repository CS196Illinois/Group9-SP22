'''
CS 124 Honors Project: Bifactor Prediction of Stock Market Prices
Authors (Group 9): Shivjeet Chanan, Joe Dirion, William Terry, Thejas Kadiri, Aryan Gaurav, Bryant Zhang

(LSTM/Neural Network Code) Credits
Adapted from
Title: stock-prediction-pytorch
Author: Rodolfo Saldanha
Date: June 2nd, 2020
Code Version: 7.0
Availability: https://www.kaggle.com/code/rodsaldanha/stock-prediction-pytorch/notebook

API/Stock Data Retrieval Credits: Yahoo Fianance API, Pandas, and Numpy Array

News Article Decomposition/Retrieval Credits: BeautifulSoup

'''
import torch as tc
import pandas as pd
import yfinance as yf
import numpy as np
import torch.nn as nn
import time
from sklearn.preprocessing import MinMaxScaler
from bs4 import BeautifulSoup
import requests
from sentence_transformers import SentenceTransformer, util

'''creates the model that vectorizes text, used for articles that we pass in'''
model = SentenceTransformer('all-MiniLM-L6-v2')

''' set the loopback time, number of Input dimensions, Hidden dimensions, Number of Hidden Layers, and Output dimension, and Epochs to train over'''
lookback = 20  # choose sequence length
input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 100

''' Rudolfo's Code of LSTM Neural Network'''
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        #sets the following data from set global variables
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        #runs the Neural Network with the LSTM constraints defined
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    #performs the backpropogation using gradient descent on the Neural Network model
    def forward(self, x):
        h0 = tc.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = tc.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(),c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

'''gets the Title, and returns the Title of the article '''
def getTitle(link):
    html_txt = requests.get(link).text
    soup2 = BeautifulSoup(html_txt, 'lxml')
    title_element = soup2.find_all('h1', class_ = 'ArticleHeader-headline')[0].text
    return title_element

'''gets the Time of when the article is written'''
def getTime(link):
    #uses request import to get the link of the text
    html_txt = requests.get(link).text

    #uses request import to get the link of the text
    soup2 = BeautifulSoup(html_txt, 'lxml')

    #strips the text and searches for the time when the article is written and returns this
    timeElement = [x.text.strip() for x in soup2.find_all('time')]
    return timeElement

'''get the Main Text of the article (Limited to First 5 Lines of the Article)'''
def getMainText(link):
    # uses request import to get the link of the text
    html_txt = requests.get(link).text

    # uses request import to get the link of the text
    soup2 = BeautifulSoup(html_txt, 'lxml')

    #loops through the article to find all article breaks/paragraphs
    text2 = soup2.find_all('p')

    #initializes the list of Sentences, starts loop through first 5 sentences
    sentenceList = []
    for x in range(6):
        #loop through first 5 sentences
        temptxt2 = text2[x].text

        #conditional to check if there is an empty sentence and not include in the lis
        if (len(str(temptxt2)) > 0):
            sentenceList.append(str(temptxt2))
    return sentenceList

'''gets Specific Arrays for a Specific Company, Period, Interval, and info needed'''
def getArrays(comp, period, interval,array_Needed):
    #initialize the boolean value that will check if Info needed is a RETRIEVABLE info
    checker = False

    #create a List of Retrievable Info from Yahoo Finance API
    dtype_List = ["Open", "High", "Low", "AdjClose", "Volume"]
    final_Series = pd.Series(dtype=float)

    #loop through valid retrievable list of info, and check if the passed arg matches, change checker from false to true
    for item in dtype_List:
        if array_Needed == item:
            checker = True
    #assertion catcher that checks if passed arg is retrievable, if it makes it here false, means its not retrievable
    assert checker != False, "You have called for Data that does not exist! Retype Array Name."

    #retrieve data from Yahoo Fianance API  and return the data from API as a Pandas Series
    data = yf.download(tickers=comp, period=period, interval=interval)
    final_Series = data[array_Needed]
    return final_Series

''' gets Numpy Array and converts to Torch Tensors to prep Data for ML'''
def changeDataType(Numpy_Tensor):
    Torch_Tensor = tc.from_numpy(Numpy_Tensor)
    return Torch_Tensor

'''retrieves all the Stock Data retrieves the Data closest to the current state of the Stock Market'''
def retrieveCurrentData(Overall_Stock):
    #convers the Pandas Series Object passed to a Numpy Array in order to remove all the past time data
    #required to do this in order to get current data since smallest period is 1 day in 1 minute smallest intervals
    OS_Numpy = Overall_Stock.to_numpy()
    OS_Numpy = np.reshape(OS_Numpy,(OS_Numpy.shape[0],1))

    #loop through the data and retrieve the last 30 minutes worth of Stock data (ie. remove the data before last 30 minutes)
    for i in range(OS_Numpy.shape[0] - 30):
        OS_Numpy = np.delete(OS_Numpy,0,0)

    #return the data as a pandas series, requires array shape (N, )
    return pd.Series(np.reshape(OS_Numpy,(OS_Numpy.shape[0], )))

''' Combines the Stock Data as a Numpy Array to the Text Data as a Numpy Array'''
def getFinalData(Stock, Text):
    #Convert the Text tensor into a Numpy Array
    text_numpy = Text.numpy()

    #Loop through the text numpy array and append the data from text data on to initial stock array
    for i in range(text_numpy.shape[0]):
        Stock = np.append(Stock,text_numpy[i,:])

    #reshape the array into a (N,1) array and then return it with all the data
    Stock = np.reshape(Stock,(Stock.shape[0],1))
    return Stock

''' Rodolfo's Code that splits the data into different sets, Train Sets and Testing Sets'''
def split_data(stock, lookback):
    data_raw = stock  # convert to numpy array
    data = []
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback):
        data.append(data_raw[index: index + lookback])
    data = np.array(data);
    test_set_size = int(np.round(0.2 * data.shape[0]));
    train_set_size = data.shape[0] - (test_set_size);

    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]

    x_test = data[train_set_size:, :-1]
    y_test = data[train_set_size:, -1, :]
    return [x_train, y_train, x_test, y_test]

''' sets the Training Data by coonverting the x_train, y_test, and x_train, and y_train'''
def setTrainingData(x_train, y_train, x_test, y_test):
    x_tensor_train = tc.from_numpy(x_train).type(tc.Tensor)
    x_tensor_test = tc.from_numpy(x_test).type(tc.Tensor)
    y_train_lstm = tc.from_numpy(y_train).type(tc.Tensor)
    y_test_lstm = tc.from_numpy(y_test).type(tc.Tensor)
    return x_tensor_train, x_tensor_test, y_train_lstm, y_test_lstm

'''Rudolfo's Code for Training the LSTM NN Model, uses the num_epochs global variable set at the top of file'''
def training_Prediction(num_epochs,x_train,y_train_lstm):
    hist = np.zeros(num_epochs)
    start_time = time.time()
    lstm = []
    for t in range(num_epochs):
        y_train_pred = model(x_train)
        loss = criterion(y_train_pred, y_train_lstm)
        hist[t] = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    training_time = time.time() - start_time
    trained = pd.DataFrame(scaler.inverse_transform(y_train_pred.detach().numpy()))
    return trained

#Gets Data, Initializes Data, converts Data, and sets Data for Machine Learning
sentence1 = getMainText('https://www.cnbc.com/2022/04/28/apple-aapl-earnings-q2-2022.html')

#Vectorizes the 5 Sentences of the Text using the Model initilized above
embeddings1 = model.encode(sentence1, convert_to_tensor=True)

#gets the Data from a specific company, on specific period, and specific interval, and specific info type (provided by Yahoo Fianance API)
some_Data = getArrays('AAPL','1d','1m','Open')

#takes in the data from some_Data, and removes all the data before 30 minutes since running program and returns the data within the 30 minutes
recent_Data = retrieveCurrentData(some_Data)

#sets the Scaler up to convert values in recent_Data into values that are to be passed into the LSTM/NN Model
scaler = MinMaxScaler(feature_range=(-1, 1))

#fits the data into Min Max scaler range
recent_Data = scaler.fit_transform(recent_Data.values.reshape(-1,1))

#combines the recent Stock Data with the Text Data
real_input = getFinalData(recent_Data,embeddings1)

#splits the x_train, y_train data, and sets the data in right format
x_train, y_train, x_test, y_test = split_data(real_input, lookback)
x_tensor_train, x_tensor_test, y_train_lstm, y_test_lstm = setTrainingData(x_train, y_train, x_test, y_test)

#initizalies and creates the LSTM Model given the global parameters defined above
model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

#defines the criterion of Mean Error and sets the gradient descent optimizer
criterion = tc.nn.MSELoss(reduction='mean')
optimiser = tc.optim.Adam(model.parameters(), lr=0.01)

#runs the training_prediction function to train the LSTM/NN , and returns original data in order to compare
training_predict = training_Prediction(num_epochs,x_tensor_train,y_train_lstm)
original = pd.DataFrame(scaler.inverse_transform(y_train_lstm.detach().numpy()))

#returns the test function AFTER set has been trained, and then returns a 1-D tensor with values of the Stock Market Fluctuating
y_test_pred = model(x_tensor_test)
y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
y_test_final = scaler.inverse_transform(y_test_lstm.detach().numpy())