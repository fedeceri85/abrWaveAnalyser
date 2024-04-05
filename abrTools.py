from pylab import *
import os
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from scipy import signal, stats
#Global variables
frequencies = [100,3000,6000,12000,18000,24000,30000,36000,42000]
lowestInt = 15 # dB

freqDict = {'100':'Click','3000':'3 kHz','6000':'6 kHz','12000':'12 kHz','18000':'18 kHz','24000':'24 kHz','30000':'30 kHz','36000':'36 kHz','42000':'42 kHz'}
intensityDict = {'0':0,'5':1,'10':2,'15':3,'20':4,'25':5,'30':6,'35':7,'40':8,'45':9,#
        '50':10,'55':11,'60':12,'65':13,'70':14,'75':15,'80':16,'85':17,'90':18,'95':19}   

fs = 195000.0/2.0 # Acquisition sampling rate



def extractABR(filename,removeDuplicates = True,saveConverted=False):
    '''
    Extract ABR data from csv files (output of Biosig)
    If saveConverted, the converted file is saved as an ordered csv file in the same folder
    '''
    f = open(filename,'r')
    l = f.readlines()
    out=[]
    header1=[]
    header2 = []
    nclicks = 0 # number of clicks recordings at 70 dB -> this is to avoid the last recording done at the end.
    for i,line in enumerate(l):

        if line.startswith('[Trace_'):
            nextL = l[i+1]
            s = nextL.split(',')
            try: 
                indicator = float(s[1])
                nextindex=2
            except:
                indicator = float(s[2])
                nextindex=3
            if s[0].endswith('Cal')==False and s[1].endswith('Cal')==False and indicator!=42001:

                frequency = indicator#float(s[1]) 
                
                intensity = float(s[nextindex])

                if frequency == 100 and intensity == 70:
                    nclicks = nclicks + 1
                if nclicks<3 or  not ((frequency==100) and (intensity == 70)):
                    header1.append(frequency)
                    header2.append(intensity)
                    nextL = l[i+2]
                    j=0
                    column =[]
                    while nextL.startswith('[Trace_')==False:

                        if nextL.startswith('TraceData_'):
                            s0 = nextL.split('=')[1]
                            s = s0.split(',')[:]
                            
                            for el in s:
                                try:
                                    column.append(float(el))
                                except ValueError:
                                    print("weird stuff goin on in "+filename)
                        j=j+1
                        try:
                            nextL=l[i+2+j]
                        except:
                            break

                    if column==[]:
                        print(frequency)
                    out.append(column)

        else:
            pass
    if saveConverted:
        table = np.vstack((header1,header2,np.array(out).T))
        np.savetxt(filename+'converted.csv',table,delimiter=',')

    pdOut = pd.DataFrame(out,index=[header1,header2]) 
    if removeDuplicates: # remove duplicated data
        t2 = pdOut.reset_index()
        pdOut['levels']=(t2['level_0'].astype(str)+'_'+t2['level_1'].astype(str)).values
        pdOut.drop_duplicates(keep='last',subset='levels',inplace=True)
        pdOut.drop('levels',inplace=True,axis=1)
    return pdOut


def makeFigure(h1,h2,out,title,thresholds = None):
    '''
    Make a figure from ABR trace data
    '''
    frequency = list(set(h1))#[100,3000,6000, 12000,18000,24000,30000,36000,42000 ]
    frequency.sort()
    intensity = list(set(h2))#arange(0,100,5)
    intensity.sort()
    nint = len(intensity)
    nfreq=len(frequency)
    freqmap=dict(zip(frequency,np.arange(len(frequency))))
    imap = dict(zip(intensity,np.arange(len(intensity))))
    if nint==1:
        nint=2
    if nfreq==1:
        nfreq=2
    fig,axs=plt.subplots(nint,nfreq,sharex=False, sharey=False,subplot_kw={'xticks': [], 'yticks': []},figsize=np.array([ 15.8 ,  16.35]))
    for i in range(len(h1)):
        column = freqmap[int(h1[i])]
        row = imap[int(h2[i])]
        #plotn = i+row*len(frequency)
        linecol = 'k'
        if thresholds is not None:
            if h2[i]>=thresholds[h1[i]]:
                linecol = 'r'
            else:
                linecol = 'k'

        axs[nint-row-1,column].plot(np.array(out)[i,:],c=linecol)
        #axs[nint-row-1,column].set_ylim((array(out).min(),array(out).max()))
        if nint-row-1==0:
            tit1 = int(h1[i])
            tit = str(tit1)+' Hz'
            if tit1 == 100:
                tit='Click'
            axs[nint-row-1,column].set_title(tit)
        if column==0:
            axs[nint-row-1,column].set_ylabel(str(int(h2[i]))+' dB')
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def find_peaks(ys,xs,fss, distance=0.5e-3, prominence=50, wlen=None,
               invert=False, detrend=True):

    '''
Utility function for ABR wave 1 extraction
    '''
    y = -ys if invert else ys
    if detrend:
        y = signal.detrend(ys)
    x = xs
    fs = fss
    prominence = np.percentile(y, prominence)
    i_distance = round(fs*distance)
    if wlen is not None:
        wlen = round(fs*wlen)
    kwargs = {'distance': i_distance, 'prominence': prominence, 'wlen': wlen}
    indices, metrics = signal.find_peaks(y, **kwargs)

    metrics.pop('left_bases')
    metrics.pop('right_bases')
    metrics['x'] = x[indices]
    metrics['y'] = y[indices]
    metrics['index'] = indices
    metrics = pd.DataFrame(metrics)
    return metrics

def guess_peaks(metrics, latency):
    '''
    Initial guess in ABR wave 1 determination using find_peaks.
    '''
    p_score_norm = metrics['prominences'] / metrics['prominences'].sum()
    guess = {}
    for i in sorted(latency.keys()):
        l = latency[i]
        l_score = metrics['x'].apply(l.pdf)
        l_score_norm = l_score / l_score.sum()
        score = 5 * l_score_norm + p_score_norm
        m = score.idxmax()
        if np.isfinite(m):
            guess[i] = metrics.loc[m]
            metrics = metrics.loc[m+1:]
        else:
            guess[i] = {'x': l.mean(), 'y': 0}

    return pd.DataFrame(guess).T



def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
        scoring='accuracy'
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt



from sklearn.model_selection import train_test_split

def loadFiles(datafolder ='../data'):
    '''
    Return pandas datasets with all the excel files, and a string with the current version of the data.
    '''
    with open(os.path.join(datafolder,'Data-version.txt')) as f:
        lines = f.readlines()
        #print(lines)
        dataVersion = lines[0]

    print('The dataset version is: ' + str(dataVersion))


    dataRep = pd.read_excel(os.path.join(datafolder,'Repaired - MachineLearningABR_ExperimentList.xlsx'))
    thresholdsRep = pd.read_excel(os.path.join(datafolder,'Repaired - Thresholds.xlsx'))#
    data6N = pd.read_excel(os.path.join(datafolder,'6N - MachineLearningABR_ExperimentList.xlsx'))
    thresholds6N = pd.read_excel(os.path.join(datafolder,'6N - Thresholds.xlsx'))#


    data = pd.concat([dataRep,data6N],ignore_index=True)
    thresholds = pd.concat([thresholdsRep,thresholds6N],ignore_index=True)   # Substitute with a concatenation of 6N and repaired when necessary

    return (data,thresholds,dataVersion)
    
def createThresholdDataset(datafolder ='../data',test_size = 0.2,random_state = 42, loadOnlyOneMonth = False, returnValidation = False,val_size=0.2):
    '''
    Load the dataset and return train and test data for the threshold finder problem
    '''
    data,thresholds,dataVersion = loadFiles(datafolder)

    X = []
    y = []
    for j,el in data.iterrows():
        filename = el['Folder 1'].split('./')[1]
        filename = os.path.join(datafolder,filename)
        mouseID = str(el['ID'])#.split('-')[0]
        t = extractABR(filename)#[arange(1200)]

        for fr in frequencies:
            X.append(t.loc[[(fr,x) for x in range(lowestInt,100,5)],:].values.ravel())
            strFr = freqDict[str(fr)]
            y.append(thresholds.loc[thresholds['MouseN - AGE']== mouseID+' - 1month',strFr].values[0])
    if loadOnlyOneMonth == False:
        for j,el in data.iterrows():
            if pd.isna(el['Folder 2'])==False:
                filename = el['Folder 2'].split('./')[1]
                filename = os.path.join(datafolder,filename)
                mouseID = str(el['ID'])#.split('-')[0]
                t = extractABR(filename)#[arange(1200)]

                for fr in frequencies:
                    X.append(t.loc[[(fr,x) for x in range(lowestInt,100,5)],:].values.ravel())
                    strFr = freqDict[str(fr)]
                    y.append(thresholds.loc[thresholds['MouseN - AGE']== mouseID+' - 3month',strFr].values[0])

            if pd.isna(el['Folder 3'])==False:
                filename = el['Folder 3'].split('./')[1]
                filename = os.path.join(datafolder,filename)
                mouseID = str(el['ID'])#.split('-')[0]
                t = extractABR(filename)#[arange(1200)]

                for fr in frequencies:
                    X.append(t.loc[[(fr,x) for x in range(lowestInt,100,5)],:].values.ravel())
                    strFr = freqDict[str(fr)]
                    y.append(thresholds.loc[thresholds['MouseN - AGE']== mouseID+' - 6month',strFr].values[0])

            if pd.isna(el['Folder 4'])==False:
                filename = el['Folder 4'].split('./')[1]
                filename = os.path.join(datafolder,filename)
                mouseID = str(el['ID'])#.split('-')[0]
                t = extractABR(filename)#[arange(1200)]

                for fr in frequencies:
                    X.append(t.loc[[(fr,x) for x in range(lowestInt,100,5)],:].values.ravel())
                    strFr = freqDict[str(fr)]
                    y.append(thresholds.loc[thresholds['MouseN - AGE']== mouseID+' - 9month',strFr].values[0])

            if pd.isna(el['Folder 5'])==False:
                filename = el['Folder 5'].split('./')[1]
                filename = os.path.join(datafolder,filename)
                mouseID = str(el['ID'])#.split('-')[0]
                t = extractABR(filename)#[arange(1200)]

                for fr in frequencies:
                    X.append(t.loc[[(fr,x) for x in range(lowestInt,100,5)],:].values.ravel())
                    strFr = freqDict[str(fr)]
                    y.append(thresholds.loc[thresholds['MouseN - AGE']== mouseID+' - 12month',strFr].values[0])


    X = np.array(X)
    y = np.array(y)



    X_train,  X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,shuffle=True,random_state=random_state)
    if returnValidation:
        X_train,  X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=val_size,shuffle=True,random_state=random_state)
        return (X_train,  X_test,X_val,y_train,y_test,y_val,dataVersion)
    else:
        return (X_train,  X_test,y_train,y_test,dataVersion)

def createClassificationDataset(datafolder ='../data',test_size = 0.2,random_state = 42,  returnValidation = False,val_size=0.2,
                                oversample=False,ages = [1],frequencies = [100,3000,6000,12000,18000,24000,30000,36000,42000],xlimit=None,lowestInt =15,
                                highestInt = 95,verbose=1):

    from collections import Counter
    '''
    Return 6N vs repaired Dataset for classification
    This only loads one month-old data
    '''
    data,_,dataVersion = loadFiles(datafolder)

    X = []
    y = []
    pairs = []

    for fr in frequencies:
        for i in range(lowestInt,highestInt+5,5):
            pairs.append([fr,i])

    for j,el in data.iterrows():
        if 1 in ages:
            filename = el['Folder 1'].split('./')[1] # DAta at 1 month
            filename = os.path.join(datafolder,filename)
            #mouseID = str(el['ID'])#.split('-')[0]
            t = extractABR(filename)#[arange(1200)]
            if xlimit is not None:
                t = t.iloc[:,:xlimit]
            X.append(t.loc[[(p[0],p[1]) for p in pairs],:].values.ravel())
            y.append(el['Strain'])
        if 3 in ages:

            try:
                filename = el['Folder 2'].split('./')[1] # DAta at 1 month
                filename = os.path.join(datafolder,filename)
                t = extractABR(filename)#[arange(1200)]
                if xlimit is not None:
                    t = t.iloc[:,:xlimit]
                X.append(t.loc[[(p[0],p[1]) for p in pairs],:].values.ravel())
                y.append(el['Strain'])
            except:
                if verbose>0:
                    print('Can''t find 3 month old data')
                else:
                    pass
        if 6 in ages:

            try:
                filename = el['Folder 3'].split('./')[1] # DAta at 1 month
                filename = os.path.join(datafolder,filename)
                t = extractABR(filename)#[arange(1200)]
                if xlimit is not None:
                    t = t.iloc[:,:xlimit]
                X.append(t.loc[[(p[0],p[1]) for p in pairs],:].values.ravel())
                y.append(el['Strain'])
            except:
                if verbose>0:
                    print('Can''t find 6 month old data')
                else:
                    pass

        if 9 in ages:

            try:
                filename = el['Folder 4'].split('./')[1] # DAta at 1 month
                filename = os.path.join(datafolder,filename)
                t = extractABR(filename)#[arange(1200)]
                if xlimit is not None:
                    t = t.iloc[:,:xlimit]
                X.append(t.loc[[(p[0],p[1]) for p in pairs],:].values.ravel())
                y.append(el['Strain'])
            except:
                if verbose>0:
                    print('Can''t find 9 month old data')
                else:
                    pass

        if 12 in ages:

            try:
                filename = el['Folder 5'].split('./')[1] # DAta at 1 month
                filename = os.path.join(datafolder,filename)
                t = extractABR(filename)#[arange(1200)]
                if t.shape[1]!=1953:
                    t = t.dropna(axis=1) #I added this for a weird file with lots of missing values when loaded
                if xlimit is not None:
                    t = t.iloc[:,:xlimit]

                X.append(t.loc[[(p[0],p[1]) for p in pairs],:].values.ravel())
                y.append(el['Strain'])
            except:
                if verbose>0:
                    print('Can''t find 12 month old data')
                else:
                    pass
    
    X = np.array(X)
    y = np.array(y)
    if oversample:
        print("WARNING!")
        print("Oversampling dataset using BORDERLINE SMOTE")
        from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE
        from imblearn.under_sampling import RandomUnderSampler
        #oversample = SMOTE(sampling_strategy=0.5)
        oversample = BorderlineSMOTE(sampling_strategy=1,random_state=random_state)
        #oversample = SVMSMOTE(sampling_strategy=0.5)
        under = RandomUnderSampler(sampling_strategy=1.0,random_state=random_state)
        X,y = oversample.fit_resample(X,y)
        X,y = under.fit_resample(X,y)
        print('Classes')
        print(Counter(y))
    
    print(Counter(y))
    X_train,  X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,shuffle=True,random_state=random_state,stratify=y)


    if returnValidation:
        X_train,  X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=val_size,shuffle=True,random_state=random_state)
        return (X_train,  X_test,X_val,y_train,y_test,y_val,dataVersion)
    else:
        return (X_train,  X_test,y_train,y_test,dataVersion)


import wandb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import socket

def initWandb(project,name = str(datetime.datetime.now()), group=None,config = {},dataVersion=None, train_size = 0, test_size = 0):
    wandb.login()
    run = wandb.init(project=project,name=name,group=group,config=config)
    wandb.config.model = group
    wandb.config.data_version = dataVersion

    wandb.config.train_size = train_size
    wandb.config.test_size = test_size
    wandb.config.architecture = socket.gethostname()

    return run

def fitRegModel(model,X_train,y_train,X_test=None,y_test = None,makePlot=True,calculateScores=True,cv=10,plotLearningCurve = False,saveToWandb=False,
modelName = '',config_dict = {}, dataVersion=None, plotOutliers = False,n_jobs=-1,random_state=42,n_splits=5,n_repeats=5):

    '''
    Fit a regression model and return the scores from cross validation (neg mean squared error),
    the MSE on train and the mean squared error on test data
    '''
    from sklearn.model_selection import RepeatedKFold, cross_validate
    from sklearn.metrics import make_scorer, mean_squared_error,mean_absolute_error,r2_score,root_mean_squared_error,max_error

    if saveToWandb:
        run = initWandb(project='Threshold prediction',name=str(datetime.datetime.now()),group=modelName,config=config_dict, dataVersion=dataVersion,
        train_size = X_train.shape[0], test_size= X_test.shape[0])

    print('Fitting '+modelName+' model')

    model.fit(X_train,y_train)
    results = model.predict(X_train)
    
    mse = sqrt(mean_squared_error(y_train,results))

    if X_test is not None:
        test_results = model.predict(X_test)
        test_mse = np.sqrt(mean_squared_error(y_test,test_results))
    else:
        test_mse=None
    
    if makePlot:
        fig = figure()
        plot(y_train,results,'o')
        
        if X_test is not None:
            plot(y_test,test_results,'og')
            
        plot([15,120],[15,120],'-r')
        #xlim(0,7)
        #ylim(0,7)
        xlabel('Real threshold (dB)')
        ylabel('Estimated threshold (dB')
    else:
        fig = None
    

    if plotLearningCurve:
        plot_learning_curve(model,'test',X_train,y_train,n_jobs=-1)

    if calculateScores:
        print('Cross validating')
        scorers = {'RMSE':make_scorer(root_mean_squared_error),
                    'mean_absolute_error':make_scorer(mean_absolute_error),
                    'r2':make_scorer(r2_score),
                    'max_error':make_scorer(max_error)
        }

        cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        c1 = cross_validate(model,X_train,y_train,scoring=scorers,n_jobs=n_jobs,cv=cv)

        scores = -c1['test_RMSE']**2
        print('MSE on train '+ str(mse))
        print('CV RMSE : ' + str( np.sqrt(-scores).mean()))
        print('CV RMSE STD : ' + str( np.sqrt(-scores).std()))
        print('CV R2 : ' + str(c1['test_r2'].mean()))
        print('CV R2 STD : ' + str(c1['test_r2'].std()))
        res = {
            'RMSE':c1['test_RMSE'],
            'MAE':c1['test_mean_absolute_error'],
            'R2':c1['test_r2'],
            'MaxError':c1['test_max_error']
        }
        return res

    else:
        scores = []

    if saveToWandb:
        wandb.log({'Train MSE': mse,
        'CV neg MSE scores': scores,
        'CV MSE':np.sqrt(-scores).mean(),
        'CV std':np.sqrt(-scores).std(),
        'Test MSE':test_mse,
        'plot':fig
         })
        #wandb.sklearn.plot_regressor(model, X_train, X_test, y_train, y_test,  model_name=modelName)
        if plotOutliers:
            wandb.sklearn.plot_outlier_candidates(model, X_train, y_train)
            wandb.sklearn.plot_residuals(model, X_train, y_train)

        run.finish()
    
    return scores, mse, test_mse


def fitClassificationModel(model,X_train,y_train,X_test=None,y_test = None,makePlot=True,crossValidation=False,plotLearningCurve = False,saveToWandb=False,
                modelName = '',config_dict = {}, dataVersion=None,random_state=42,calculatePValue=False,njobs= -1,n_splits=5,n_repeats=5):

    '''
    TODO : WANDB not implemented
    '''

    if saveToWandb:
        run = initWandb(project='6N vs repaired',name=str(datetime.datetime.now()),group=modelName,config=config_dict, dataVersion=dataVersion,
        train_size = X_train.shape[0], test_size= X_test.shape[0])

    from sklearn.model_selection import cross_val_score, cross_validate,RepeatedStratifiedKFold,permutation_test_score
    from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix,classification_report,make_scorer,\
                                f1_score, accuracy_score, roc_auc_score,balanced_accuracy_score , precision_score,recall_score
    from sklearn.feature_selection import f_classif,mutual_info_classif

    if crossValidation == False:
        model.fit(X_train,y_train)




        if makePlot:
            print('CLASSIFICATION REPORT ON TRAIN')
            print(classification_report(y_train, model.predict(X_train)))

            if X_test is not None:
                print('CLASSIFICATION REPORT ON TEST')
                print(classification_report(y_test, model.predict(X_test)))

            print('Confusion matrix on train')
            cm = confusion_matrix(y_train,model.predict(X_train),normalize=None,labels=['6N','Repaired'])
            ConfusionMatrixDisplay(cm).plot()
            show()

            print('Confusion matrix on test')
            cm = confusion_matrix(y_test,model.predict(X_test),normalize=None,labels=['6N','Repaired'])
            fig = ConfusionMatrixDisplay(cm).plot()
            show()


        if saveToWandb:
            wandb.sklearn.plot_classifier(model, X_train, X_test, y_train, y_test, model.predict(X_test), 
                                        model.predict_proba(X_test), labels = ['6N','Repaired'], model_name=modelName, feature_names=None)
            wandb.summary['Class report test'] = classification_report(y_test, model.predict(X_test), output_dict=True)
        
            #wandb.sklearn.plot_regressor(model, X_train, X_test, y_train, y_test,  model_name=modelName)
        
            run.finish()

  
    if crossValidation == True:
        print('Cross validating...')
        scorer = {'accuracy':make_scorer(balanced_accuracy_score),
        'f1_scorer':make_scorer(f1_score,pos_label='6N'),
        'f1_scorer_r':make_scorer(f1_score,pos_label='Repaired'),
        'f1_scorer_avg':make_scorer(f1_score,average='weighted'),
        'auc': make_scorer(roc_auc_score,average='weighted',needs_proba=True),
        'precision':make_scorer(precision_score,pos_label='6N'),
        'recall':make_scorer(recall_score,pos_label='6N'),
        'precision_avg':make_scorer(precision_score,average='weighted'),
        'recall_avg':make_scorer(recall_score,average='weighted'),
        'precision_r':make_scorer(precision_score,pos_label='Repaired'),
        'recall_r':make_scorer(recall_score,pos_label='Repaired'),
        }
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        cv2 = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=1, random_state=random_state) # for p value calculation we use n=1 for speed
        #c1 = pd.DataFrame()
        #c2 = pd.DataFrame()
        #accuracies = []
        #f1_6Ns = []
        #f1_Repaireds = []
        #rocs = []
        # for el in cv.split(X_train,y_train):
        #     X_train_val = X_train[el[0],:]
        #     y_train_val = y_train[el[0]]
        #     X_val = X_train[el[1],:]
        #     y_val = y_train[el[1]]

        #     model.fit(X_train_val,y_train_val)
        #     y_pred = model.predict(X_val)
            
        #     accuracies.append(accuracy_score(y_val,y_pred))
        #     f1_6Ns.append(f1_score(y_val,y_pred, pos_label='6N'))
        #     f1_Repaireds.append(f1_score(y_val,y_pred, pos_label='Repaired'))

        #     rocs.append(roc_auc_score(y_val,model.predict_proba(X_val)[:,1]))
        
        c1 = cross_validate(model,X_train,y_train,cv=cv,scoring=scorer,n_jobs=njobs)
        #c2 = cross_validate(model,X_train,y_train,cv=cv,scoring='roc_auc',n_jobs=-1) 
        # c1['test_f1_scorer'] = f1_6Ns
        # c1['test_f1_scorer_r'] = f1_Repaireds
        # c1['test_accuracy'] = accuracies
        # c2['test_score'] = rocs   # we use c1 and c2 so the code is back compatible 

        if calculatePValue == True :
            #TODO figure out how to include feature selection with the p value
            pts = permutation_test_score(model, X_train, y_train, groups=None, cv=cv2, n_permutations=100, n_jobs=njobs, random_state=random_state, verbose=2, scoring='accuracy', fit_params=None)
        else:
            pts = [None,None,None]
        print('Average CV F1 score: '+ str(mean(c1['test_f1_scorer_avg']))+ '   STD: ' + str(std(c1['test_f1_scorer_avg'])))
        print('Average CV F1 score: 6N: '+ str(mean(c1['test_f1_scorer'])) + '   STD: ' + str(std(c1['test_f1_scorer'])))
        print('Average CV F1 score: Repaired: '+ str(mean(c1['test_f1_scorer_r']))+ '   STD: ' + str(std(c1['test_f1_scorer_r'])))
        print('Average CV accuracy: '+ str(mean(c1['test_accuracy']))+ '   STD: ' + str(std(c1['test_accuracy'])))
        #print('Average CV ROC AUC: '+ str(mean(c2['test_score']))+ '   STD: ' + str(std(c2['test_score'])))
        print('Average CV ROC AUC: '+ str(mean(c1['test_auc']))+ '   STD: ' + str(std(c1['test_auc'])))
        print('Permutation test p value : '+ str(pts[2]))


        res = {
            'accuracy':c1['test_accuracy'],
            'test_f1_scorer_avg':c1['test_f1_scorer_avg'],
            'test_f1_scorer_6N':c1['test_f1_scorer'],
             'test_f1_scorer_Rep':c1['test_f1_scorer_r'],
            'roc_auc_score':c1['test_auc'],
            
            'test_precision_scorer_avg':c1['test_precision_avg'],
            'test_precision_scorer_6N':c1['test_precision'],
            'test_precision_scorer_Rep':c1['test_precision_r'],

            'test_recall_scorer_avg':c1['test_recall_avg'],
            'test_recall_scorer_6N':c1['test_recall'],
            'test_recall_scorer_Rep':c1['test_recall_r'],

            'p_value':pts[2]
            
        }
        return res

    return None
        #if saveToWandb:
            #


def createFutureThresholdDataset(datafolder ='../data',inputFreqs = None,inputs = ['1month'],target = '6month',strains = ['6N','Repaired'],test_size = 0.2,random_state = 42, targetFrequency = 100, mode = 'single',returnValidation = False,val_size=0.2):
    '''
    Load the dataset and return train and test data for the threshold finder problem
    '''
    data,thresholds,dataVersion = loadFiles(datafolder)

    X = []
    y = []

    pairs = []
    if inputFreqs is None:
        inputFreqs = frequencies
        
    for fr in inputFreqs:
        for i in range(lowestInt,100,5):
            pairs.append([fr,i])

    if (mode=='waveamp') or (mode=='wavelatency') or (mode=='waveampdiff') or (mode=='wavelatencydiff'):
        masterWave1 = createWave1Dataset(age=target)
        masterWave1 = masterWave1.query('Freq==@targetFrequency')
        masterWave1['ID'] = masterWave1['ID'].astype(str)
        #Add the missing latencies as the max of the lantencies per mouse
        for id in masterWave1['ID'].unique():
            for freq in masterWave1['Freq'].unique():
                el2 = masterWave1.query("ID==@id & Freq==@freq")

                masterWave1.loc[el2.index[pd.isna(el2['Wave1 latency'])],'Wave1 latency'] = el2['Wave1 latency'].max()
        masterWave1 = masterWave1.set_index(['ID','Freq','Intensity'])

        masterWave1month = createWave1Dataset(age='1month')
        masterWave1month = masterWave1month.query('Freq==@targetFrequency')
        masterWave1month['ID'] = masterWave1month['ID'].astype(str)
        #Add the missing latencies as the max of the lantencies per mouse
        for id in masterWave1month['ID'].unique():
            for freq in masterWave1month['Freq'].unique():
                el2 = masterWave1month.query("ID==@id & Freq==@freq")
                masterWave1month.loc[el2.index[pd.isna(el2['Wave1 latency'])],'Wave1 latency'] = el2['Wave1 latency'].max()
        masterWave1month = masterWave1month.set_index(['ID','Freq','Intensity'])

        masterWave1['Wave1 amp diff'] = (masterWave1['Wave1 amp'] - masterWave1month['Wave1 amp']).dropna()
        masterWave1['Wave1 latency diff'] = (masterWave1['Wave1 latency'] - masterWave1month['Wave1 latency']).dropna()
        masterWave1 = masterWave1.reset_index()
    #input = '1month'
    #target = '3month'

    for j,el in data.iterrows():
        if el['Strain'] in strains:
            mouseID = str(el['ID'])#.split('-')[0]
   
            try:
                if mode == 'mean': # Just calculate the average threshold at all frequency
                    this_tr = []
                    for fr in frequencies:
                        strFr = freqDict[str(fr)]
                        
                        this_tr.append(thresholds.loc[thresholds['MouseN - AGE']== mouseID+' - '+target,strFr].values[0])
                    y.append(mean(this_tr))
                elif mode == 'median': # Just calculate the average threshold at all frequency
                    this_tr = []
                    for fr in frequencies:
                        strFr = freqDict[str(fr)]
                        
                        this_tr.append(thresholds.loc[thresholds['MouseN - AGE']== mouseID+' - '+target,strFr].values[0])
                    y.append(median(this_tr))

                elif mode == 'diff':   #  calculate the diff threshold for the targetFrequency between the age and 1 month
                    this_tr = []
                    strFr = freqDict[str(targetFrequency)]
                    this_y = thresholds.loc[thresholds['MouseN - AGE']== mouseID+' - '+target,strFr].values[0] - thresholds.loc[thresholds['MouseN - AGE']== mouseID+' - 1month',strFr].values[0]
                    y.append(this_y)

                elif mode =='meandiff':#calculate the difference in average threshold between target and 1 month
                    this_tr = []
                    this_input = []
                    for fr in frequencies:
                        strFr = freqDict[str(fr)]
                        this_tr.append(thresholds.loc[thresholds['MouseN - AGE']== mouseID+' - '+target,strFr].values[0])
                        this_input.append(thresholds.loc[thresholds['MouseN - AGE']== mouseID+' - 1month',strFr].values[0])
                    y.append(mean(this_tr)-mean(this_input))

                elif mode =='mediandiff':#calculate the difference in average threshold between target and 1 month
                    this_tr = []
                    this_input = []
                    for fr in frequencies:
                        strFr = freqDict[str(fr)]
                        this_tr.append(thresholds.loc[thresholds['MouseN - AGE']== mouseID+' - '+target,strFr].values[0])
                        this_input.append(thresholds.loc[thresholds['MouseN - AGE']== mouseID+' - 1month',strFr].values[0])
                    y.append(median(this_tr)-median(this_input))


                elif mode=='single':  # calculate the threshold for the targetFrequency at the desired age
                    strFr = freqDict[str(targetFrequency)]
                    y.append(thresholds.loc[thresholds['MouseN - AGE']== mouseID+' - '+target,strFr].values[0])
                
                elif mode=='singlediff':  # calculate the threshold shift for the targetFrequency at the desired age
                    strFr = freqDict[str(targetFrequency)]

                    y.append(thresholds.loc[thresholds['MouseN - AGE']== mouseID+' - '+target,strFr].values[0] - 
                             thresholds.loc[thresholds['MouseN - AGE']== mouseID+' - 1month',strFr].values[0])


                elif mode=='allfreq':
                    this_thresh = []
                    for strFr in list(freqDict.values()):
                        this_thresh.append(thresholds.loc[thresholds['MouseN - AGE']== mouseID+' - '+target,strFr].values[0])    
                    y.append(this_thresh)

                elif mode=='waveampdiff':
                    values = masterWave1.query("ID==@mouseID").sort_values('Intensity')['Wave1 amp'].values
                    if len(values)==20:
                        y.append(masterWave1.query("ID==@mouseID").sort_values('Intensity')['Wave1 amp diff'].values.astype(float))
                    else:
                        print(len(values))
                        raise IndexError
                    
                elif mode=='wavelatencydiff':
                   
                    values = masterWave1.query("ID==@mouseID").sort_values('Intensity')['Wave1 amp'].values
                    if len(values)==20:  
                        y.append(masterWave1.query("ID==@mouseID").sort_values('Intensity')['Wave1 latency diff'].values.astype(float))
                    else:
                        raise IndexError
                
                elif mode=='waveamp':
                    values = masterWave1.query("ID==@mouseID").sort_values('Intensity')['Wave1 amp'].values
                    if len(values)==20:
                        y.append(masterWave1.query("ID==@mouseID").sort_values('Intensity')['Wave1 amp'].values.astype(float))
                    else:
                        print(len(values))
                        raise IndexError
                    
                elif mode=='wavelatency':
                   
                    values = masterWave1.query("ID==@mouseID").sort_values('Intensity')['Wave1 amp'].values
                    if len(values)==20:  
                        y.append(masterWave1.query("ID==@mouseID").sort_values('Intensity')['Wave1 latency'].values.astype(float))
                    else:
                        raise IndexError
                else:
                    print('Mode not supported')
                    return
                ts = []
                monthFolderDict = {'1month':'1','3month':'2','6month':'3','9month':'4','12month':'5'}
                for input in inputs:
                    filename = el['Folder {}'.format(monthFolderDict[input])].split('./')[1] # DAta at 1 month
                    filename = os.path.join(datafolder,filename)
                    t = extractABR(filename)#[arange(1200)]
                    ts.append(t.loc[[(p[0],p[1]) for p in pairs],:].values.ravel())

                t = np.hstack(ts)
                X.append(t)
            except IndexError:
                print('Cannot find data for '+mouseID)




    X = np.array(X)
    y = np.array(y)


    X_train,  X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,shuffle=True,random_state=random_state)
    if returnValidation:
        X_train,  X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=val_size,shuffle=True,random_state=random_state)
        return (X_train,  X_test,X_val,y_train,y_test,y_val,dataVersion)
    else:
        return (X_train,  X_test,y_train,y_test,dataVersion)

def createWave1Dataset(datafolder = '../data/',waveanalysisFolder = 'waveAnalysisResults',age='1month',addMissingAmplitudes=True):
    
    
    sixN = pd.read_excel(os.path.join(datafolder,'6N - MachineLearningABR_MouseList.xlsx'))
    rep = pd.read_excel(os.path.join(datafolder,'Repaired - MachineLearningABR_MouseList.xlsx'))

    masterAll = pd.DataFrame()
    rows = []
    for j,el in sixN.iterrows():
        try:
            filename = str(el['ID']) + ' - '+age+'.csv'
            fullpath = os.path.join(datafolder,waveanalysisFolder,filename)
            a = pd.read_csv(fullpath)
            a['Strain'] = '6N'
            a['Age'] = age
            a['ID'] = el['ID']
            rows.append(a)
        except FileNotFoundError:
            pass
          #  print('File not found')
    #masterAll = masterAll.append(a)
        
    for j,el in rep.iterrows():
        try:
            filename = str(el['ID']) + ' - '+age+'.csv'
            fullpath = os.path.join(datafolder,waveanalysisFolder,filename)
            a = pd.read_csv(fullpath)
            a['Strain'] = 'Repaired'
            a['Age'] = age
            a['ID'] = el['ID']
            rows.append(a)
        except FileNotFoundError:
            pass
#            print('File not found')
    masterAll = pd.concat(rows,ignore_index=True)

    masterAll['Wave1 amp'] = masterAll['P1_y']-masterAll['N1_y']
    masterAll['Wave1 latency'] = masterAll['P1_x']
    masterAll['Wave1 amp'] = masterAll['Wave1 amp'].fillna(0) 
    
    if addMissingAmplitudes:#Add amplitudes below 0
        allIntensities = set(np.arange(0,100,5))
        rowsToAdd = []
        for id in masterAll['ID'].unique():
            el = masterAll.query("ID==@id")
            for freq in el['Freq'].unique():
                el2 = el.query("Freq==@freq")
                ints = set(el2['Intensity'].unique())
                missingIntensityies = allIntensities - ints
                for intensity in missingIntensityies:
                    row = pd.Series(index=el2.columns,dtype='object')
                    row['Freq'] = freq
                    row['Intensity'] = intensity
                    row['Strain'] = el2['Strain'].values[0]
                    row['Age'] = el2['Age'].values[0]
                    row['ID'] = id
                    row['Wave1 amp'] = 0
                    rowsToAdd.append(row)
        rowsToAddDf = pd.concat(rowsToAdd,axis=1).T
    
        masterAll = pd.concat([masterAll,rowsToAddDf],ignore_index=True)
        
    return masterAll

def plotFeatureImportance(fi,abr=None,savgolOrder = 51,ylims=(-5.5,10)):
    from scipy.signal import savgol_filter
    if abr is None:
        abr = extractABR('../data/20220520/Mouse #1-[226251-LE].csv')

    fi = savgol_filter(fi,savgolOrder,1)
    ntraces = 153
    ppt = 1953#int(fi.size/ntraces)

    
    fig = makeFigure(abr.reset_index().values[:,0],abr.reset_index().values[:,1],abr.values,title='')
    for column in range(9):
        for row in range(17):
            tr = fi[(16-row+column*17)*ppt:(16-row+1)*ppt + column*17*ppt]
            currAx = row*9 + column
            ax2 = fig.axes[currAx].twinx()
            ax2.plot(tr*100000/2-5,'r')

    for i in range(180,333):
        ax = fig.axes[i]
        ax.set_ylim(ylims[0],ylims[1])
        ax.axis('off')

    for i in range(0,180):
        ax = fig.axes[i]
        ax.set_ylim(-4,4)
        ax.axis('off')
        ax.set_xlim(0,12*fs/1000)

    fig.patch.set_facecolor('white')
    fig.subplots_adjust(wspace=0.05,hspace=0)
    #tight_layout()
    fig.show()

def collectResults(savefolder,experimentType='',age=1):
    suffices = ['Global','NoHighFreq','OnlyLowFreq','OnlyBadFreq','Click','3000','6000','12000','18000','24000','30000','36000','42000']
    if age!=1:
        suffices = [el+'_'+str(age)+'months' for el in suffices]

    realSuff = ['Global','NoHighFreq','OnlyLowFreq','OnlyBadFreq','Click','3kHz','6kHz','12kHz','18kHz','24kHz','30kHz','36kHz','42kHz']
    master = pd.DataFrame()
    rows = []
    for i,suff in enumerate(suffices):


        try:
            res = pd.read_csv(os.path.join(savefolder,f'forest{experimentType}_kFoldCrossValidation_'+suff+'.csv'))
            res['Frequency'] =  realSuff[i]
            res['Model'] = 'Random Forest'
            rows.append(res)
        except:
            pass
        try:
            res = pd.read_csv(os.path.join(savefolder,f'forest{experimentType}_kFoldCrossValidation_AnovaFS10percent'+suff+'.csv'))
            res['Frequency'] =  realSuff[i]
            res['Model'] = 'Random Forest Anova FS'
            rows.append(res)
        except:
            pass
        try:
            res = pd.read_csv(os.path.join(savefolder,f'SVC{experimentType}_kFoldCrossValidation_AnovaFS10percent_'+suff+'.csv'))
            res['Frequency'] =  realSuff[i]
            res['Model'] = 'SVC Anova FS'
            rows.append(res)
        except:
            pass
        try:
            res = pd.read_csv(os.path.join(savefolder,f'XGBOOST{experimentType}_kFoldCrossValidation_AnovaFS10percent_'+suff+'.csv'))
            res['Frequency'] =  realSuff[i]
            res['Model'] = 'XGBOOST Anova FS'
            rows.append(res)
        except:
            pass
        try:
            res = pd.read_csv(os.path.join(savefolder,f'Rocket{experimentType}_kFoldCrossValidation_AnovaFS10percent_'+suff+'.csv'))
            res['Frequency'] =  realSuff[i]
            res['Model'] = 'Rocket Anova FS'
            rows.append(res)
        except:
            pass
        try:
            res = pd.read_csv(os.path.join(savefolder,f'hivecote{experimentType}_kFoldCrossValidation_AnovaFS10percent_'+suff+'.csv'))
            res['Frequency'] =  realSuff[i]
            res['Model'] = 'HiveCote Anova FS'
            rows.append(res)
        except:
            pass
        try:
            res = pd.read_csv(os.path.join(savefolder,f'MLP{experimentType}_kFoldCrossValidation_AnovaFS10percent_'+suff+'.csv'))
            res['Frequency'] =  realSuff[i]
            res['Model'] = 'MLP Anova FS'
            rows.append(res)
    #########################################################
            res = pd.read_csv(os.path.join(savefolder,f'forest{experimentType}_kFoldCrossValidation_MutualInfo10percent_'+suff+'.csv'))
            res['Frequency'] =  realSuff[i]
            res['Model'] = 'Random Forest MI FS'
            rows.append(res)
        except:
            pass
        try:
            res = pd.read_csv(os.path.join(savefolder,f'SVC{experimentType}_kFoldCrossValidation_MutualInfo10percent_'+suff+'.csv'))
            res['Frequency'] =  realSuff[i]
            res['Model'] = 'SVC MI FS'
            rows.append(res)
        except:
            pass
        try:
            res = pd.read_csv(os.path.join(savefolder,f'XGBOOST{experimentType}_kFoldCrossValidation_MutualInfo10percent_'+suff+'.csv'))
            res['Frequency'] =  realSuff[i]
            res['Model'] = 'XGBOOST MI FS'
            rows.append(res)

        except:
            pass
        try:
            res = pd.read_csv(os.path.join(savefolder,f'Rocket{experimentType}_kFoldCrossValidation_MutualInfo10percent_'+suff+'.csv'))
            res['Frequency'] =  realSuff[i]
            res['Model'] = 'Rocket MI FS'
            rows.append(res)
        except:
            pass
        try:
            res = pd.read_csv(os.path.join(savefolder,f'hivecote{experimentType}_kFoldCrossValidation_MutualInfo10percent_'+suff+'.csv'))
            res['Frequency'] =  realSuff[i]
            res['Model'] = 'HiveCote MI FS'
            rows.append(res)
        except:
            pass
        try:
            res = pd.read_csv(os.path.join(savefolder,f'MLP{experimentType}_kFoldCrossValidation_MutualInfo10percent_'+suff+'.csv'))
            res['Frequency'] =  realSuff[i]
            res['Model'] = 'MLP MI FS'
            rows.append(res)
        except:
            pass
        try:
            res = pd.read_csv(os.path.join(savefolder,f'SVC{experimentType}_kFoldCrossValidation_'+suff+'.csv'))
            res['Frequency'] = realSuff[i]
            res['Model'] = 'SVC'
            rows.append(res)

        except:
            pass
        try:
            res = pd.read_csv(os.path.join(savefolder,f'XGBOOST{experimentType}_kFoldCrossValidation_'+suff+'.csv'))
            res['Frequency'] = realSuff[i]
            res['Model'] = 'XGBOOST'
            rows.append(res)

            
        except:
            pass
        try:
            res = pd.read_csv(os.path.join(savefolder,f'forest{experimentType}-featureselection_kFoldCrossValidation_'+suff+'.csv'))
            res['Frequency'] =  realSuff[i]
            res['Model'] = 'Random Forest-feat.select.'
            rows.append(res)
        except:
            pass
        try:
            res = pd.read_csv(os.path.join(savefolder,f'svc{experimentType}-featureselection_kFoldCrossValidation_'+suff+'.csv'))
            res['Frequency'] =  realSuff[i]
            res['Model'] = 'SVC-feat.select.'
            rows.append(res)
        except:
            pass
        try:
            res = pd.read_csv(os.path.join(savefolder,f'XGBOOST{experimentType}-featureselection_kFoldCrossValidation_'+suff+'.csv'))
            res['Frequency'] =  realSuff[i]
            res['Model'] = 'XGBOOST-feat.select.'
            rows.append(res)
        except:
            pass
        try:
            res = pd.read_csv(os.path.join(savefolder,f'Rocket{experimentType}_kFoldCrossValidation_'+suff+'.csv'))
            res['Frequency'] =  realSuff[i]
            res['Model'] = 'Rocket'
            rows.append(res)
        except:
            pass
        try:
            res = pd.read_csv(os.path.join(savefolder,f'hivecote{experimentType}_kFoldCrossValidation_'+suff+'.csv'))
            res['Frequency'] =  realSuff[i]
            res['Model'] = 'HiveCote'
            rows.append(res)
        except:
            pass
        try:
            res = pd.read_csv(os.path.join(savefolder,f'MLP{experimentType}_kFoldCrossValidation_'+suff+'.csv'))
            res['Frequency'] =  realSuff[i]
            res['Model'] = 'MLP'
            rows.append(res)
        except:
            pass
        try:
            res = pd.read_csv(os.path.join(savefolder,f'forest{experimentType}_kFoldCrossValidation_10ms_'+suff+'.csv'))
            res['Frequency'] =  realSuff[i]
            res['Model'] = 'Random Forest limited'
            rows.append(res)
        except:
            pass
        try:
            res = pd.read_csv(os.path.join(savefolder,f'SVC{experimentType}_kFoldCrossValidation_10ms_'+suff+'.csv'))
            res['Frequency'] = realSuff[i]
            res['Model'] = 'SVC limited'
            rows.append(res)


        except:
            pass

    master = pd.concat(rows)
    
    try:
        master = master.drop('Unnamed: 0',axis=1)
    except:
        pass
    return master

def interFunc(x,a,b,c,d):
    
    return a*exp(-(x-c)/b)+d 


def loadKingsData(shift=54,scaling=False,filename = '../data/Kings - MAchineLEarningABR_ExperimentList.xlsx',dataFolder = '../data'):
    
    lowestInt = 15
    highestInt = 85
    kingsData = pd.read_excel(filename)
    kingsData.loc[kingsData['Status']=='Ahl-Repaired','Strain']='Repaired'
    kingsData.loc[kingsData['Status']=='UnRepaired','Strain']='6N'

    popt = np.array([ 3.26223496e+04,  2.12954754e+03, -1.34506980e+04,  1.11239997e+00]) # Standard parameters for scaling
    X_kings = []
    y_kings  = []
    index = 0
    for j,el in kingsData.iterrows():
        fname = el['Folder 1']
        t = extractABR(os.path.join(dataFolder,fname))

        pairs = []

        for fr in [100]:
            for ii in range(lowestInt,highestInt+5,5):
                pairs.append([fr,ii])

        try:
            X_kings.append(t.loc[[(p[0],p[1]) for p in pairs],:].values.ravel())
            y_kings.append(el['Strain'])
        except KeyError as e:
            print(e)
            index = index+1
            print(index)
    X_kings = np.array(X_kings)
    y_kings = np.array(y_kings)
    X_kings = X_kings[:,shift:]

    if scaling:
        X_kings_scaled = X_kings.copy()
        for i in range(X_kings_scaled.shape[0]):
            #X_kings_scaled[i,matchingPoints[0,0]:matchingPoints[-1,0]]=X_kings_scaled[i,matchingPoints[0,0]:matchingPoints[-1,0]]/ f(arange(matchingPoints[0,0],(matchingPoints[-1,0])))
            X_kings_scaled[i,100:]=X_kings_scaled[i,100:]/ interFunc(np.arange(100,X_kings_scaled.shape[1]),*popt)
        X_kings = X_kings_scaled

    X_train,X_test, y_train,y_test = train_test_split(X_kings,y_kings,test_size=0.25,random_state=42)

    return X_train,X_test,y_train,y_test,X_kings,y_kings

def loadSheffieldData(shift=54,dataFolder='../data'):
    lowestInt = 15
    highestInt = 85
    X_train,  X_test,y_train,y_test,dataVersion = createClassificationDataset(test_size=0.25,oversample=False,ages=[1,],frequencies=[100],lowestInt=lowestInt,highestInt=highestInt,datafolder = dataFolder)
    X_train = X_train[:,:-shift]
    X_test = X_test[:,:-shift]
    X_full = np.vstack([X_train,X_test])
    y_full = np.hstack([y_train,y_test])

    return X_train,X_test,y_train,y_test,X_full,y_full,dataVersion