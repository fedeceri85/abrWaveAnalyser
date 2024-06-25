import os
import pandas as pd
import numpy as np
import pandas as pd
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
    try:
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
    except: #If an error is thrown, assume the file is already converted. 
        pdOut = pd.read_csv(filename) 

    if removeDuplicates: # remove duplicated data
        t2 = pdOut.reset_index()
        pdOut['levels']=(t2['level_0'].astype(str)+'_'+t2['level_1'].astype(str)).values
        pdOut.drop_duplicates(keep='last',subset='levels',inplace=True)
        pdOut.drop('levels',inplace=True,axis=1)
    
    fs = 195000.0/2.0 #So far this is the frequency of all our files.
    return pdOut,fs

def extractABRDS(filenames,folder='.'):
    '''
    Extract ABR data from csv files (Dwayne Simmons Lab - Lieberman system)
    '''
    out = []
    fss = []
    for filename in filenames:
        print(filename)
        df = pd.read_csv(os.path.join(folder,filename), encoding='unicode_escape',sep='\t',skiprows=5,header=0).reset_index()  
        f = open(os.path.join(folder,filename), encoding='unicode_escape')
        l = f.readlines()
        for line in l:
            if line.startswith(':LEVELS'):
                ints = line.split(':')[-1].split(';')
                intensities = [int(i) for i in ints if i !='\n']    

            elif line.startswith(':SW EAR'):
                asd = line
                params = asd.split('\t')
                for p in params:
                    if p.startswith('SW FREQ'):
                        freq = int(1000*float(p.split(':')[1]))
                    elif p.startswith('SAMPLE'):
                        fs = 1e6/float(p.split(':')[-1]) 
                        fss.append(fs)

        df.columns = intensities
        header1 = [freq]*df.shape[1]
        header2 = intensities
        df2 = pd.DataFrame(df.T.values,index = [header1,header2])
        out.append(df2)

        if len(set(fss))!=1:
            raise NotImplementedError('All the traces should have the same sampling frequency')
        else:
            fs = fss[0]
        
    return pd.concat(out),fs

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
    



