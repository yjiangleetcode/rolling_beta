import datetime
from datetime import date
import urllib
import numpy as np
import matplotlib.pyplot as plt
import argparse

class rolling_beta(object):
    def __init__(self, symbol, benchmark, look_back, aggregation_period, num_days_past, trimming, intercept):
        '''Parameter setup
        # symbol: stock symbol (X in OLS)
        # benchmark: benchmark of beta (Y in OLS)
        # look_back: look back period to calculate betas
        # aggregation_period: number of days of return (daily, n-days, weekly, monthly)
        # trimming: WinSorize data eg: trimming = 90, set those data in top and bottom 10% to be upper bound or lower bound
        # intercept: with or without intercept
        # beta: rolling beta
        # beta_table: a dictionary for searching for single beta
        '''
        #Initialization
        self.symbol = symbol.upper()
        self.date = []
        self.symbol_close = []
        self.symbol_return = []
        self.benchmark = benchmark
        self.benchmark_close = []
        self.benchmark_return = []
        self.look_back = look_back
        self.aggregation_period = aggregation_period
        self.num_days_past = num_days_past
        self.current_date = date.today()
        self.current_date= self.current_date.strftime('%Y-%m-%d')
        self.trimming = trimming
        self.intercept = intercept
        self.beta = []
        self.beta_table={}
        
        #dealing with date
        self.get_start_date()
        
        #download data
        self.url_read()
        
        #get returns sampled with aggregation period
        self.returns()
        
        #trimming outliers
        self.trimming_outliers()
        #calculate beta
        self.CalculateBeta()
        
    #############################################################################
    #Functions
    #build start and end date
    def get_start_date(self):
        self.start_date = date.today() - datetime.timedelta(self.num_days_past + self.look_back + self.aggregation_period)
        self.start_date = self.start_date.strftime('%Y-%m-%d')
        self.start_year, self.start_month, self.start_day = self.start_date.split('-')
        self.start_month = str(int(self.start_month) - 1)
        self.end_year, self.end_month, self.end_day = self.current_date.split('-')
        self.end_month = str(int(self.end_month) - 1)
    #download data from Yahoo finance using URL    
    def url_read(self):
        #download symbol data
        url_string = "http://ichart.finance.yahoo.com/table.csv?s={0}".format(self.symbol)
        url_string += "&a={0}&b={1}&c={2}".format(self.start_month, self.start_day, self.start_year)
        url_string += "&d={0}&e={1}&f={2}".format(self.end_month, self.end_day, self.end_year)
        csv = urllib.urlopen(url_string).readlines()
        csv.reverse()
        for bar in xrange(0, len(csv) - 1):
            ds, open_, high, low, close, volume, adjc = csv[bar].rstrip().split(',')
            open_, high, low, close, adjc = [float(x) for x in [open_, high, low, close, adjc]]
            factor = 1
            if close != adjc:
                factor = adjc / close
            open_, high, low, close = [x * factor for x in [open_, high, low, close]]
            dt = datetime.datetime.strptime(ds, '%Y-%m-%d')
            self.symbol_close.append(close)
            self.date.append(dt.date())
            self.beta_table[dt.strftime('%Y-%m-%d')] = bar
        #download benchmark data
        url_string = "http://ichart.finance.yahoo.com/table.csv?s={0}".format(self.benchmark)
        url_string += "&a={0}&b={1}&c={2}".format(self.start_month, self.start_day, self.start_year)
        url_string += "&d={0}&e={1}&f={2}".format(self.end_month, self.end_day, self.end_year)
        csv = urllib.urlopen(url_string).readlines()
        csv.reverse()
        for bar in xrange(0, len(csv) - 1):
            ds, open_, high, low, close, volume, adjc = csv[bar].rstrip().split(',')
            open_, high, low, close, adjc = [float(x) for x in [open_, high, low, close, adjc]]
            factor = 1
            if close != adjc:
                factor = adjc / close
            open_, high, low, close = [x * factor for x in [open_, high, low, close]]
            dt = datetime.datetime.strptime(ds, '%Y-%m-%d')
            self.benchmark_close.append(close)
    #aggregating returns
    def returns(self):
        n = self.aggregation_period
        for i in range(n):
            self.symbol_return.append(0)
            self.benchmark_return.append(0)
        for i in range(n,len(self.symbol_close)):
            self.symbol_return.append((self.symbol_close[i]-self.symbol_close[i-n])/self.symbol_close[i])
            self.benchmark_return.append((self.benchmark_close[i]-self.benchmark_close[i-n])/self.benchmark_close[i])
    #calculate betas using memoization 
    def CalculateBeta(self):
        '''The beta from OLS can be calculated as beta=cov(X,Y)/Var(X)=cov(X,Y)/cov(X,X)
        Then, use xySum to store the sum of x(i) times y(i).
        So, the sum of x*y from i to j is xySum(j)-xySum(i)
        Finally calculate the beta
        In case of no intersection, beta=xySum/x2Sum
        Time complexity: O(n)        
        '''
        length = len(self.symbol_return)
        n = self.aggregation_period
        xySum, xSum, ySum, x2, x2Sum = ([None]*length for _ in range(5))
        xySum[n-1]=self.symbol_return[n-1]*self.benchmark_return[n-1]
        xSum[n-1]=self.symbol_return[n-1]
        ySum[n-1]=self.benchmark_return[n-1]
        x2[n-1]=self.symbol_return[n-1]**2
        x2Sum[n-1]=x2[n-1]
        for i in range(n):
            self.beta.append(0)
        if (self.intercept == True):
            for i in range(n,length):
                xySum[i] = xySum[i-1]+self.symbol_return[i]*self.benchmark_return[i]
                xSum[i] = xSum[i-1]+self.symbol_return[i]
                ySum[i] = ySum[i-1]+self.benchmark_return[i]  
                x2[i] = self.symbol_return[i]**2
                x2Sum[i] = x2Sum[i-1]+x2[i]
                if (i<n+self.look_back-1):
                    self.beta.append(0)
                elif (i==n+self.look_back-1):
                    cov = xySum[i]/self.look_back-xSum[i]/self.look_back*ySum[i]/self.look_back
                    var = x2Sum[i]/self.look_back-(xSum[i]/self.look_back)**2
                    self.beta.append(cov/var)
                else:
                    cov = (xySum[i]-xySum[i-self.look_back])/self.look_back-(xSum[i]-xSum[i-self.look_back])/self.look_back*(ySum[i]-ySum[i-self.look_back])/self.look_back
                    var = (x2Sum[i]-x2Sum[i-self.look_back])/self.look_back-((xSum[i]-xSum[i-self.look_back])/self.look_back)**2
                    self.beta.append(cov/var)
        else:
            for i in range(n,length):
                xySum[i] = xySum[i-1]+self.symbol_return[i]*self.benchmark_return[i]
                xSum[i] = xSum[i-1]+self.symbol_return[i]
                ySum[i] = ySum[i-1]+self.benchmark_return[i]  
                x2[i] = self.symbol_return[i]**2
                x2Sum[i] = x2Sum[i-1]+x2[i]
                if (i<n+self.look_back-1):
                    self.beta.append(0)
                elif (i==n+self.look_back-1):
                    self.beta.append(xySum[i]/x2Sum[i])
                else:
                    self.beta.append((xySum[i]-xySum[i-self.look_back])/(x2Sum[i]-x2Sum[i-self.look_back]))
    #WinSorization
    def trimming_outliers(self):
        if (self.trimming < 50):
            self.trimming += 50
        lowerS, upperS = np.percentile(self.symbol_return, [100-self.trimming, self.trimming])
        lowerB, upperB = np.percentile(self.benchmark_return, [100-self.trimming, self.trimming])
        for i in range(len(self.symbol_return)):
            if (self.symbol_return[i]<lowerS):
                self.symbol_return[i] = lowerS
            elif (self.symbol_return[i]>upperS):
                self.symbol_return[i] = upperS
            if (self.benchmark_return[i]<lowerB):
                self.benchmark_return[i] = lowerB
            elif (self.benchmark_return[i]>upperB):
                self.benchmark_return[i] = upperB
    #Make plot of betas
    def make_plot(self):
        plt.plot(self.beta[self.aggregation_period+self.look_back:len(self.benchmark_close)])
        plt.legend=False
        plt.title("%s day history rolling beta for %s sampled every %s day(s) with benchmark %s" % (len(self.benchmark_close),self.symbol, self.aggregation_period, self.benchmark.upper()))
        plt.ylabel(u'\u03B2', fontsize = 'large', rotation='horizontal')
        plt.show()
    #Search for single beta of a given day
    def searchBeta(self,date):
        return self.beta[self.beta_table[date]]   
    #Return beta list
    def returnBetaList(self):
        return self.beta
    
    #functions for printing and saving data
    def __repr__(self):
        return self.to_csv()
    def to_csv(self):
        return ''.join(["{0},{1},{2:.2f},{3:.2f},{4:.2f}\n".format(self.symbol,self.date[bar],self.symbol_close[bar],self.benchmark_close[bar],self.beta[bar]) for bar in xrange(len(self.symbol_close))])
    def write_csv(self, filename):
        with open(filename, 'w') as f:
            f.write(self.to_csv())

#Parser for command-line options
'''parser = argparse.ArgumentParser(description='Get parameters.')
parser.add_argument('symbol',type=str)
parser.add_argument('benchmark',type=str)
parser.add_argument('look_back',type=int)
parser.add_argument('aggregation_period',type=int)
parser.add_argument('num_days_past',type=int)
parser.add_argument('trimming',type=int)
parser.add_argument('intercept',type=int)
args = parser.parse_args()
print(rolling_beta(args.symbol, args.benchmark, args.look_back, args.aggregation_period, args.num_days_past, args.trimming, args.intercept))'''
