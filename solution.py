import rolling_beta


if __name__ == '__main__':
    #read input parameters from input.txt file
    '''
    symbol = 'msft'
    benchmark = 'spy'
    look_back = 30
    aggregation_period = 1
    num_days_past = 600
    trimming = 90
    intercept=0   (False)'''
    for line in open('input.txt', 'r'):
        symbol, benchmark, look_back, aggregation_period, num_days_past, trimming, intercept = line.strip().split(',')
        look_back = int(look_back)
        aggregation_period = int(aggregation_period)
        num_days_past = int(num_days_past)
        trimming = int(trimming)
        intercept = int(intercept)
        beta=rolling_beta.rolling_beta(symbol, benchmark, look_back, aggregation_period, num_days_past, trimming, intercept)
        #beta.write_csv('beta.csv')
        #Methods
    print(beta) #print all betas
    beta.make_plot() #make plot for betas
    print(beta.searchBeta('2014-08-04')) #search for beta of a given day
    betas = beta.returnBetaList() #return a list of betas
    
