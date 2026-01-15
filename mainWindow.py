from PyQt5.Qt import QApplication
from PyQt5.QtCore import Qt
from PyQt5 import QtGui,QtWidgets
from pyqtgraph.parametertree import Parameter, ParameterTree
import numpy as np
import sys
sys.path.append('../')
import abrTools as at
import os
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore,QtWidgets
from wavePeaksWindow import myGLW, findNearestPeak
import pandas as pd
import itertools

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

import seaborn as sns
from resultsWindows import resultWindow,resultAverageTraceWindow,resultThresholdWindow

# Set white graph
pg.setConfigOptions(antialias=True)
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


_instance = QApplication.instance()
if not _instance:
    _instance = QApplication([])
app = _instance



# frequencies = ['Click', '3 kHz', '6 kHz', '12 kHz', '18 kHz', '24 kHz',
#        '30 kHz', '36 kHz', '42 kHz']
intensitiesL = [str(x)+'dB' for x in range(0,100,5)]
     
frequenciesDict = {
    100:'Click',
    3000: '3 kHz',
    6000: '6 kHz',
    12000: '12 kHz',
    18000: '18 kHz',
    24000: '24 kHz',
    30000: '30 kHz',
    36000: '36 kHz',
    42000: '42 kHz',
}

#fs = 195000.0/2.0 # Acquisition sampling rate
# dataFolder = '../../data'
# waveAnalysisFolder = os.path.join(dataFolder,'waveAnalysisResults')

# data,thresholds,dataversion = at.loadFiles(datafolder=dataFolder)
# thresholds['ID'] = thresholds['MouseN - AGE'].str.split(' - ',expand=True)[0].astype(int)
# thresholds['Age (months)'] = thresholds['MouseN - AGE'].str.split(' - ',expand=True)[1].str.split('month',expand=True)[0].astype(int)
# strain = []
# for el in thresholds['ID']:
#     strain.append(data.loc[data['ID']==el,'Strain'].values[0])
# thresholds['Strain'] = strain
# dates = data['ID'].unique()

def makeFigureqt(h1,h2,out,layout,title,fs,wavePoints = None,plotDict = None,wavePointsPlotDict = None,thresholds = None,xlim=8):
    '''
    Make a figure from ABR trace data using pyqtgraph. Modifies an existing figure if the pyqtgraphs plots are passed through plotDict
    '''
    frequency = list(set(h1))#[100,3000,6000, 12000,18000,24000,30000,36000,42000 ]
    frequency.sort()
    intensity = list(set(h2))#arange(0,100,5)
    intensity.sort()
    nint = len(intensity)
    nfreq=len(frequency)
    freqmap=dict(zip(frequency,np.arange(len(frequency))))
    imap = dict(zip(intensity,np.arange(len(intensity))))
    time = np.arange(out.shape[1])/fs*1000

    ymin = out.min()
    ymax = out.max()
    if nint==1:
        nint=2
    if nfreq==1:
        nfreq=2
    plots = {}
    wavePointsPlot = {}
    plotToFreqIntMap = {}  # Dictionary that maps (row,col) of the plots to (freq, intensity) pairs

    if plotDict is not None:
        for el in plotDict.keys():
            plotDict[el].setData([],[])
    if wavePointsPlotDict is not None:
        for el in wavePointsPlotDict.keys():
            wavePointsPlotDict[el].setData([],[])


    for i in range(len(h1)):
        column = freqmap[int(h1[i])]
        row = imap[int(h2[i])]
        plotToFreqIntMap[(nint-row-1,column)] = (int(h1[i]),int(h2[i]))
        #plotn = i+row*len(frequency)
        linecol ='k'
        if thresholds is not None:
            try:
                if h2[i]>=thresholds[str(int(h1[i]))]:
                    linecol = 'k'
                else:
                    linecol = 'r'
            except KeyError:
                linecol = 'k'

        ## Add the traces to the plots
        plotID = str(nint-row-1)+ ' ' + str(column)
        if plotDict is None:
            p2 = layout.addPlot(nint-row-1,column)

            #p2.addLegend(offset=(50, 0))
            pl = p2.plot(time,np.array(out)[i,:], pen=linecol, name='p1.1')
            p2.hideAxis('left')
            p2.hideAxis('right')
            p2.hideAxis('bottom')
            #p2.setYRange(ymin,ymax,padding = 0.1)
            p2.setXRange(0,xlim,padding = 0)
            
            plots[plotID] = pl
            p2.setObjectName(plotID)


        else:
            try:
                pl = plotDict[str(nint-row-1)+ ' ' + str(column)]
                pl.setData(time,np.array(out)[i,:],pen=linecol)

               # pl.setPen(pg.mkPen(linecol,width=5))    
                layout.getItem(nint-row-1,column).setXRange(0,xlim,padding = 0)

            except KeyError:
                p2 = layout.addPlot(nint-row-1,column)

                #p2.addLegend(offset=(50, 0))
                pl = p2.plot(time,np.array(out)[i,:], pen=linecol, name='p1.1')
                p2.hideAxis('left')
                p2.hideAxis('right')
                p2.hideAxis('bottom')
                #p2.setYRange(ymin,ymax,padding = 0)
                p2.setXRange(0,xlim,padding = 0)
               
                plots[plotID] = pl
                p2.setObjectName(plotID)

        #Add the points for the wave peaks and trough to the plots
        if wavePoints is not None:
            slice = wavePoints.loc[(wavePoints['Freq']==int(h1[i])) & (wavePoints['Intensity']==int(h2[i]) )]
            if slice.shape[0]>1:
                print("More than one entry for this frequency-intensity combination. Serious error")
                print(h1[i])
                print(h2[i])
            if slice.shape[0]==1:
                x,y = wavePointsToScatter(slice)

                if wavePointsPlotDict is None:
                    p2 = layout.getItem(nint-row-1,column)
                    p1Point = pg.ScatterPlotItem(x=[],y=[],symbol = '+',pen=pg.mkPen('r'),)
                            
                    p2.addItem(p1Point)
                    

                    p1Point.setData(x,y)
                    wavePointsPlot[plotID]= p1Point
                else:
                    try:
                        p1Point = wavePointsPlotDict[str(nint-row-1)+ ' ' + str(column)]

                        p1Point.setData(x,y)
                    except KeyError: 
                        p2 = layout.getItem(nint-row-1,column)
                        p1Point = pg.ScatterPlotItem(x=[],y=[],symbol = '+',size=4,pen=pg.mkPen('r'),)
                        p2.addItem(p1Point)
                        

                        p1Point.setData(x,y)
                        wavePointsPlot[plotID]= p1Point
        #axs[nint-row-1,column].plot(np.array(out)[i,:],c=linecol)
        #axs[nint-row-1,column].set_ylim((array(out).min(),array(out).max()))
    if plotDict is not None:
        plots.update(plotDict)

    if wavePointsPlotDict is not None:
        wavePointsPlot.update(wavePointsPlotDict)

    return plots,wavePointsPlot,plotToFreqIntMap

def wavePointsToScatter(wavePoints):
    xlabels = ['P'+str(i)+'_x' for i in [1,2,3,4,5]] + ['N'+str(i)+'_x' for i in [1,2,3,4,5]]
    ylabels = ['P'+str(i)+'_y' for i in [1,2,3,4,5]]  + ['N'+str(i)+'_y' for i in [1,2,3,4,5]]  
    x = wavePoints[xlabels].values[0,:]
    y = wavePoints[ylabels].values[0,:]
    return (x,y)



class abrWindow(pg.GraphicsView):
    def __init__(self, parent=None, useOpenGL=None, background='default'):
        super().__init__(parent, useOpenGL, background)

        self.setWindowTitle('ABR Wave Analysis - Sheffield Hearing research group') 
        self.setGeometry(0,0,1300,1000)
        self.move(0,0)


        screen = QApplication.primaryScreen()
        print('Screen: %s' % screen.name())
        size = screen.size()
        print('Size: %d x %d' % (size.width(), size.height()))
        rect = screen.availableGeometry()
        print('Available: %d x %d' % (rect.width(), rect.height()))
        self.resize(int(rect.width()*0.66),rect.height())

        params = [
        # {'name':'Strain','type':'list','values':['6N','Repaired']},
            {'name':'Reverse polarity','type':'bool','value':False},
            {'name':'Set threshold','type':'action'},
            {'name':'Set no threshold','type':'action'},
            {'name':'Guess Wave Peak positions','type':'action'},
            {'name':'Guess Wave Peak higher intensities','type':'action'},
            {'name':'Guess Wave Peak lower intensities','type':'action'},
            {'name':'ML Wave 1 (experimental)','type':'action'},
            {'name':'ML threshold (experimental)','type':'action'},
            {'name':'X-axis lim (ms)','type':'float','value':8.0},
            {'name':'Plot wave analysis','type':'action'},                        
            {'name':'Frequencies','type':'str','value':'100,3000,6000,12000,18000,24000,30000,36000,42000'},
            {'name':'Plot thresholds','type':'action'},
            ]
            
        paramsFile = [
            {'name': 'Single file mode', 'type': 'group', 'children': [
            {'name': 'Open file', 'type': 'action'},
            {'name': 'Save ABR traces', 'type': 'action'},
            {'name': 'Save results', 'type': 'action'},
            ]},
            {'name': 'Multiple file mode', 'type': 'group', 'children': [
            {'name': 'Open experiment list', 'type': 'action'},
            {'name': 'Previous file', 'type': 'action', 'enabled': False},
            {'name': 'Next file', 'type': 'action', 'enabled': False},
            {'name': 'Selected File', 'type': 'str', 'value': '0/0'},
            {'name': 'Plot avg ABR traces', 'type': 'action'},
            {'name': 'Export avg ABR traces', 'type': 'action'},
            {'name': 'Export results', 'type': 'action'},
            ]},
        ]

        ## Create tree of Parameter objects
        self.p = Parameter.create(name='params', type='group', children=params)
        self.t = ParameterTree()
        self.t.setParameters(self.p, showTop=False)
        self.t.setStyleSheet("QWidget { font-size: 10pt; }")

        self.t.move(int(rect.width()*0.66+rect.width()*0.34/4),int(rect.height()/1.7))
        self.t.resize(int(rect.width()*0.34/3),int(rect.height()/2.5))
        #Create tree of parameters for file handling
        self.pFile = Parameter.create(name='paramsFile', type='group', children=paramsFile)
        self.tFile = ParameterTree()
        self.tFile.setParameters(self.pFile, showTop=False)
        self.tFile.move(int(rect.width()*0.66),int(rect.height()/1.7))
        self.tFile.resize(int(rect.width()*0.34/4),int(rect.height()/2.5))
        self.tFile.setStyleSheet("QWidget { font-size: 10pt; }")


        self.waveAnalysisWidget = myGLW(show=True)
        self.waveAnalysisWidget.move(int(rect.width()*0.66),0)
        self.waveAnalysisWidget.resize(int(rect.width()*0.34),int(rect.height()/2))

        self.activeRowCol = (0,0)
        self.waveAnalysisWidget.t.move(int(rect.width()*0.66+rect.width()*0.2),int(rect.height()/1.7))
        self.waveAnalysisWidget.t.resize(int(rect.width()*0.34/2.5),int(rect.height()/2.5))
        self.waveAnalysisWidget.t.setStyleSheet("QWidget { font-size: 10pt; }")

        self.makeConnections()
        self.show()
        self.t.show()
        self.tFile.show()

    def openFileCb(self):
        self.multipleFilesMode = False
        dlg = pg.widgets.FileDialog.FileDialog()
        #dlg.setFileMode(QFileDialog.AnyFile)
    #    dlg.setFilter("CSV files (*.csv)")
        #filenames = QStringList()
        #If we open a single file, we disable the next and previous file buttons
        self.pFile.param('Multiple file mode').param('Next file').setOpts(enabled=False) 
        self.pFile.param('Multiple file mode').param('Previous file').setOpts(enabled=False)
        self.pFile.param('Multiple file mode').param('Selected File').setOpts(value = str('0/0'))

        self.currentFileIndex = 0
        self.experimentList = None

        if dlg.exec_():
            filenames = dlg.selectedFiles()
        fullpath = filenames[0]
        self.folder, self.currentFile = os.path.split(fullpath)
        self.initData()

    def openMultipleFilesCb(self):

        self.multipleFilesMode = True
        dlg = pg.widgets.FileDialog.FileDialog()
        if dlg.exec_():
            filenames = dlg.selectedFiles()
            print(filenames)
        self.experimentList = pd.read_csv(filenames[0])
        self.pFile.param('Multiple file mode').param('Next file').setOpts(enabled=True) 
        self.pFile.param('Multiple file mode').param('Previous file').setOpts(enabled=True)

        folder = os.path.split(filenames[0])[0]
        print(folder)
        
        self.totalFiles = self.experimentList.shape[0]
        self.currentFileIndex = 0
        self.pFile.param('Multiple file mode').param('Selected File').setOpts(value = str(self.currentFileIndex+1)+'/'+str(self.totalFiles))
        
        #If the file name is not a full path, we add the folder of the selected file to it.
        for j,el in self.experimentList.iterrows():
            if not os.path.isabs(el['Filename']):
                self.experimentList.loc[j,'Filename'] = os.path.join(folder,os.path.relpath(el['Filename']))
        
        print(self.experimentList['Filename'].loc[0])

        self.openFileFromMultiple()

    def openFileFromMultiple(self):
        self.folder, self.currentFile = os.path.split(self.experimentList['Filename'].loc[self.currentFileIndex])
        self.initData()

    def nextFileCb(self):
        self.currentFileIndex+=1
        if self.currentFileIndex == self.totalFiles:
            self.currentFileIndex = 0
        self.pFile.param('Multiple file mode').param('Selected File').setOpts(value = str(self.currentFileIndex+1)+'/'+str(self.totalFiles))
        self.openFileFromMultiple()

    def prevFileCb(self):
        self.currentFileIndex-=1
        if self.currentFileIndex == -1:
            self.currentFileIndex = self.totalFiles-1
        self.pFile.param('Multiple file mode').param('Selected File').setOpts(value = str(self.currentFileIndex+1)+'/'+str(self.totalFiles))
        self.openFileFromMultiple()
    
    def initData(self):
        try:
            self.abr,self.fs = at.extractABR(os.path.join(self.folder,self.currentFile))
        except:
            #Try with DS files:
            prefix,_,postfix = self.currentFile.rpartition('-')
            files = []
            for dirpath, dirnames, filenames in os.walk(self.folder):
                for filename in filenames:
                    if (prefix in filename) and ('_waveAnalysisResults' not in filename) and ('_thresholds' not in filename):
                        f = open(os.path.join(self.folder,filename), encoding='unicode_escape')
                        l = f.readlines()
                        if l[0].startswith(':RUN'):
                            files.append(filename)
                        f.close()
            files.sort()
            self.abr,self.fs = at.extractABRDS(files,folder=self.folder)
            self.currentFile = prefix

        if self.p['Reverse polarity']:
            self.abr = -self.abr
        self.waveAnalysisWidget.fs = self.fs

        freqs = []
        intens = []
        for el in self.abr.index:
            freqs.append(el[0])
            intens.append(el[1])
        self.frequencies = np.sort(list(set(freqs)))
        self.intensities = np.sort(list(set(intens)))
        
        self.layout = pg.GraphicsLayout()
        #layout.layout.setContentsMargins(-100,-100,-100,-100)
        self.outerlayout = pg.GraphicsLayout()

        self.titleLabel = self.outerlayout.addLabel('Title',color='k',size='16pt',bold=True,row=0,col=0,colspan=10)

        for i,freq in enumerate(self.frequencies):
            try:
                self.outerlayout.addLabel(frequenciesDict[int(freq)],color='k',size='10pt',col=i+1,row=1)
            except KeyError:
                self.outerlayout.addLabel(int(freq),color='k',size='10pt',col=i+1,row=1)
        
        for i,intens2 in enumerate( self.intensities[::-1]):
            self.outerlayout.addLabel(str(int(intens2))+' dB',color='k',size='10pt',col=0,row=i+2)

        self.outerlayout.addItem(self.layout,colspan=len(self.frequencies),rowspan=len( self.intensities),row=2,col=1)

        self.setCentralItem(self.outerlayout)
        
        self.sc = self.scene()
        self.sc2 = self.layout.scene()
        self.sc2.sigMouseClicked.connect(self.onMouseClicked)
        

       # self.wavePoints = pd.DataFrame(columns=['Freq',	'Intensity','P1_x','P1_y','N1_x','N1_y','P2_x','P2_y','N2_x','N2_y','P3_x','P3_y','N3_x','N3_y','P4_x','P4_y','N4_x','N4_y'])


        self.plotDict,self.wavePointsPlotDict, self.plotToFreqIntMap = makeFigureqt(freqs,intens,self.abr.values,self.layout,'',fs=self.fs,wavePoints=None,xlim=self.p['X-axis lim (ms)'])
        
        self.waveAnalysisWidget.p['Peak type'] = 'P1'
        self.loadWaveAnalysis()
        self.updateCurrentPlotCb()
        self.setActivePlot(0,0)


    def loadWaveAnalysis(self):
        try: 
            filename = os.path.splitext(self.currentFile)[0]+'_waveAnalysisResults.csv'
            self.wavePoints = pd.read_csv(os.path.join(self.folder,filename))
            
            # Handle backward compatibility: add P5/N5 columns if missing (for old files)
            for col in ['P5_x', 'P5_y', 'N5_x', 'N5_y']:
                if col not in self.wavePoints.columns:
                    self.wavePoints[col] = np.nan
                    
        except FileNotFoundError:
            self.wavePoints = pd.DataFrame(columns=['Freq',	'Intensity','P1_x','P1_y','N1_x','N1_y','P2_x','P2_y','N2_x','N2_y','P3_x','P3_y','N3_x','N3_y','P4_x','P4_y','N4_x','N4_y','P5_x','P5_y','N5_x','N5_y'])
            print('Wave analysis not found')


        if self.multipleFilesMode:
            #in multipleFilesMode, the thresholds are not saved in the file, but in the experiment list
            thresholds = self.experimentList.loc[self.currentFileIndex,self.frequencies.astype(int).astype(str)]
            self.threshDict = pd.Series(thresholds,index=self.frequencies.astype(int).astype(str))
            #replace the NaNs with 0
            self.threshDict = self.threshDict.fillna(0)
            
        else:
            try:
                filename = os.path.splitext(self.currentFile)[0]+'_thresholds.csv'
                self.threshDict = pd.read_csv(os.path.join(self.folder,filename)).dropna(axis=1).T.squeeze()
                #Make sure the index is the correct format
                self.threshDict.index = [str(int(float(el))) for el in self.threshDict.index]
            except FileNotFoundError:  
                self.threshDict = pd.Series([0]*len(self.frequencies),index=self.frequencies.astype(int).astype(str))
                print('Thresholds not found')         

    def setActivePlot(self,row=0,col=0):
        
        # reset previous active plot
        try:
            self.highlightTraceAt(self.activeRowCol[0],self.activeRowCol[1],1)
        except KeyError: # If the trace was not highlighted move on
            pass
        
        try:
            self.highlightTraceAt(row,col,3)
        except KeyError:
            #if the trace not present go back to the previous one and break
            self.highlightTraceAt(self.activeRowCol[0],self.activeRowCol[1],3)
            return
        #set the new active plot
        self.activeRowCol=(row,col)


        data = self.getTraceAt(row,col)

        freq,intens = self.plotToFreqIntMap[(row,col)]
        print('Selected trace Freq: '+str(freq)+'   Intensity: '+str(intens))
        if self.wavePoints is not None:
            selectedWavePoints = self.wavePoints.loc[(self.wavePoints['Freq']==freq) & (self.wavePoints['Intensity']==intens) ]
            
            if selectedWavePoints.shape[0] ==0:
                selectedWavePoints = None
        else:
            selectedWavePoints = None
        
        self.waveAnalysisWidget.p['Peak type'] = 'P1'
        self.waveAnalysisWidget.setData(data[0],data[1],wavePoints = selectedWavePoints)

    def highlightTraceAt(self,row,col,pensize=3):
        msg =  str(row)+' '+str(col)
        try:
            penColor = self.plotDict[msg].opts['pen'].color()
        except AttributeError:
            penColor = self.plotDict[msg].opts['pen']

        self.plotDict[msg].setPen(pg.mkPen(penColor,width=pensize))
    
    def getTraceAt(self,row,col):
        msg =  str(row)+' '+str(col)
        return self.plotDict[msg].getData()
        
    def makeConnections(self):
        #p.keys()['Strain'].sigValueChanged.connect(changeStrainCb)
        self.pFile.param('Single file mode').param('Open file').sigActivated.connect(self.openFileCb)
        self.p.keys()['Reverse polarity'].sigValueChanged.connect(self.reversePolarityCb)
        self.p.keys()['Set threshold'].sigActivated.connect(self.setThresholdCb)
        self.p.keys()['Set no threshold'].sigActivated.connect(self.setAboveThresholdCb)
        self.p.keys()['Plot wave analysis'].sigActivated.connect(self.plotResultsCb)
        self.p.keys()['Plot thresholds'].sigActivated.connect(self.plotThresholdsCb)
        self.p.keys()['Guess Wave Peak positions'].sigActivated.connect(lambda: self.guessWavePoints('both'))
        self.p.keys()['Guess Wave Peak higher intensities'].sigActivated.connect(lambda: self.guessWavePoints('higher'))
        self.p.keys()[ 'Guess Wave Peak lower intensities'].sigActivated.connect(lambda: self.guessWavePoints('lower'))
        self.p.keys()['ML Wave 1 (experimental)'].sigActivated.connect(self.MLGuessCB)
        self.p.keys()['ML threshold (experimental)'].sigActivated.connect(self.MLGuessThresholdsCb)
        self.p.keys()['X-axis lim (ms)'].sigValueChanged.connect(self.changeXlimCb)
        self.pFile.param('Single file mode').param('Save ABR traces').sigActivated.connect(self.saveABRTracesCb)
        self.pFile.param('Single file mode').param('Save results').sigActivated.connect(self.saveResultsCb)
        self.pFile.param('Multiple file mode').param('Open experiment list').sigActivated.connect(self.openMultipleFilesCb)
        self.pFile.param('Multiple file mode').param('Next file').sigActivated.connect(self.nextFileCb)
        self.pFile.param('Multiple file mode').param('Previous file').sigActivated.connect(self.prevFileCb)
        self.pFile.param('Multiple file mode').param('Export results').sigActivated.connect(self.saveResultsMultipleCb)
        self.pFile.param('Multiple file mode').param('Plot avg ABR traces').sigActivated.connect(self.plotAverageABRTracesCb)
        self.pFile.param('Multiple file mode').param('Export avg ABR traces').sigActivated.connect(self.saveAverageABRTracesCb)

        self.waveAnalysisWidget.finishSignal.connect(self.retrieveResultsCb)
        self.waveAnalysisWidget.changeTraceSignal.connect(self.navigateTraces)
        self.waveAnalysisWidget.guessAboveSignal.connect(lambda: self.guessWavePoints('higher'))
        self.waveAnalysisWidget.guessBelowSignal.connect(lambda: self.guessWavePoints('lower'))
        self.waveAnalysisWidget.propagatePointSignal.connect(lambda peak: self.guessWavePoints('lower', peaks=[peak]))
        self.waveAnalysisWidget.resetPointBelowSignal.connect(self.resetPointBelowCb)



    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_W:
            self.navigateTraces('Up')
        elif ev.key() == Qt.Key_S:
            self.navigateTraces('Down')
        elif ev.key() == Qt.Key_A:
            self.navigateTraces('Left')        
        elif ev.key() == Qt.Key_D:
            self.navigateTraces('Right')
        elif ev.key() == Qt.Key_R:
            self.guessWavePoints("higher")
        elif ev.key() == Qt.Key_F:
            self.guessWavePoints("lower")
        # elif ev.key() == Qt.Key_Z:
        #     self.prevMouseCb()
        # elif ev.key() == Qt.Key_X:
        #     self.nextMouseCb()
        elif ev.key() == Qt.Key_Z:
            self.setThresholdCb()

    def navigateTraces(self,a):
        row, col = self.activeRowCol
        if a == 'Up':
            self.setActivePlot(row-1,col)
        elif a == 'Down':
            self.setActivePlot(row+1,col)
        elif a =='Left':
            self.setActivePlot(row,col-1)
        elif a == 'Right':
            self.setActivePlot(row,col+1)

    def saveResultsCb(self):
       # dat = dates[self.p['ID']]
        filename = os.path.splitext(self.currentFile)[0]+'_waveAnalysisResults.csv'
        self.wavePoints.sort_values(['Freq','Intensity'])
        self.wavePoints['Wave 1 amplitude (uV)'] = self.wavePoints['P1_y'] - self.wavePoints['N1_y']
        self.wavePoints['Wave 1 latency (ms)'] = self.wavePoints['P1_x']

        wavePoints = self.wavePoints.copy()
        wavePoints = self.addMissingIntensities(wavePoints)

        wavePoints.to_csv(os.path.join(self.folder,filename),index=False)

        filename2 = os.path.splitext(self.currentFile)[0]+'_thresholds.csv'
        self.threshDict.to_frame().T.to_csv(os.path.join(self.folder,filename2),index=None)
        
    def saveResultsMultipleCb(self):
        rows = []
        for j,el in self.experimentList.iterrows():
            self.folder, self.currentFile = os.path.split(el['Filename'])
            wavepoints = pd.read_csv(os.path.join(self.folder,os.path.splitext(self.currentFile)[0]+'_waveAnalysisResults.csv'))
            wavepoints['Group'] = el['Group']
            wavepoints['MouseID'] = el['MouseID']
            rows.append(wavepoints)
        # Create dialog to select save folder
        allWavepoints = pd.concat(rows,ignore_index=True)

        avgs = allWavepoints.groupby(['Group','Freq','Intensity'])[['Wave 1 amplitude (uV)','Wave 1 latency (ms)']].agg(['mean','std','count'])
        # Sort the aggregated dataframe by Group, Frequency, and Intensity
        avgs = avgs.sort_index(level=['Group', 'Freq', 'Intensity'])
        

        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.Directory)

        if dlg.exec_():
            saveFolder = dlg.selectedFiles()[0]
            # Save combined results
            allWavepoints.to_csv(os.path.join(saveFolder, 'all_waveAnalysisResults.csv'), index=False)
            avgs.to_csv(os.path.join(saveFolder, 'all_waveAnalysisResults_average.csv'))
          

    def retrieveResultsCb(self):
    
        points = self.waveAnalysisWidget.getPoints()


        #Check if the combination of freq and intensity exists already in the wavePoints DataFrame
        freq,intens = self.plotToFreqIntMap[self.activeRowCol]

        selectedWavePoints = self.wavePoints.loc[(self.wavePoints['Freq']==freq) & (self.wavePoints['Intensity']==intens) ]
        if selectedWavePoints.shape[0] ==0:
            self.wavePoints = pd.concat([self.wavePoints,
                    pd.DataFrame({
                    'Freq':freq,
                    'Intensity':intens,
                    'P1_x': points['P1'][0],
                    'P1_y' : points['P1'][1],
                    'N1_x': points['N1'][0],
                    'N1_y' : points['N1'][1],

                    'P2_x': points['P2'][0],
                    'P2_y' : points['P2'][1],
                    'N2_x': points['N2'][0],
                    'N2_y' : points['N2'][1],

                    'P3_x': points['P3'][0],
                    'P3_y' : points['P3'][1],
                    'N3_x': points['N3'][0],
                    'N3_y' : points['N3'][1],

                    'P4_x': points['P4'][0],    
                    'P4_y' : points['P4'][1],
                    'N4_x': points['N4'][0],
                    'N4_y' : points['N4'][1],

                    'P5_x': points['P5'][0],    
                    'P5_y' : points['P5'][1],
                    'N5_x': points['N5'][0],
                    'N5_y' : points['N5'][1],                

            },index=[0])],ignore_index=True,axis=0)
      

        elif selectedWavePoints.shape[0] ==1:
            for ii in [1,2,3,4,5]:
                self.wavePoints.loc[(self.wavePoints['Freq']==freq) & (self.wavePoints['Intensity']==intens),'P'+str(ii)+'_x'] =  points['P'+str(ii)][0]
                self.wavePoints.loc[(self.wavePoints['Freq']==freq) & (self.wavePoints['Intensity']==intens) ,'N'+str(ii)+'_x'] =  points['N'+str(ii)][0]
                self.wavePoints.loc[(self.wavePoints['Freq']==freq) & (self.wavePoints['Intensity']==intens) ,'P'+str(ii)+'_y'] =  points['P'+str(ii)][1]
                self.wavePoints.loc[(self.wavePoints['Freq']==freq) & (self.wavePoints['Intensity']==intens) ,'N'+str(ii)+'_y'] =  points['N'+str(ii)][1]

        #print(self.wavePoints)

        self.updateCurrentPlotCb() 






    def updateCurrentPlotCb(self):
        #dat = dates[self.p['ID']]
    
      
       
     #   abr = at.extractABR(os.path.join(self.folder,self.currentFile))
        #abr2 = at.extractABR(os.path.join(dataFolder,data.loc[data['ID']==dat,'Folder 2'].values[0]))
        #abr = pd.concat([abr,abr2])
        freqs = []
        intens = []
        for el in self.abr.index:
            freqs.append(el[0])
            intens.append(el[1])

      #  frequencies2 =np.array(np.sort(list(set(freqs))))#[ 100,3000,6000,12000,18000,24000,30000,36000,42000]

        #try:
      #  self.threshDict = self.thresholds#dict(zip(frequencies2,self.thresholds[frequencies2.astype(str)].values[0])) 

        #except:
        #    self.threshDict =  dict(zip(frequencies2,[0]*9))
        

        self.plotDict,self.wavePointsPlotDict, self.plotToFreqIntMap  = makeFigureqt(freqs,intens,self.abr.values,self.layout,'',fs=self.fs,plotDict=self.plotDict ,thresholds=self.threshDict,wavePoints=self.wavePoints,wavePointsPlotDict=self.wavePointsPlotDict,xlim=self.p['X-axis lim (ms)'])
        self.titleLabel.setText(self.currentFile)
        self.highlightTraceAt(self.activeRowCol[0],self.activeRowCol[1],3)


    def addMissingIntensities(self,wavePoints):
        wavePoints = wavePoints.dropna(subset=['Wave 1 amplitude (uV)'])

        #Add amplitude = 0 for missing intensities below threshold
        rowsToAppend = []
        for freq in self.frequencies:
            threshold = self.threshDict[str(int(freq))]
            intensities = wavePoints.loc[wavePoints['Freq']==freq,'Intensity']
            missingIntensities = np.setdiff1d(self.intensities,intensities)


            for intens in    missingIntensities:
                if intens < threshold:
                    row = pd.Series(index=wavePoints.columns,dtype='object')
                    row['Freq'] = freq
                    row['Intensity'] = intens
                    row['P1_x'] = np.nan
                    row['P1_y'] = 0
                    row['N1_x'] = np.nan 
                    row['N1_y'] = 0
                    row['Wave 1 amplitude (uV)'] = 0
                    row['Wave 1 latency (ms)'] = np.nan
                    rowsToAppend.append(row)
                else:
                    #check whether the particular intensity/freq combination is in the abr, if it is than the amplitude is 0, if it is no than we don't add it
                    if (freq,intens) in self.abr.index:
                        row = pd.Series(index=wavePoints.columns,dtype='object')
                        row['Freq'] = freq
                        row['Intensity'] = intens
                        row['P1_x'] = np.nan
                        row['P1_y'] = 0
                        row['N1_x'] = np.nan 
                        row['N1_y'] = 0
                        row['Wave 1 amplitude (uV)'] = 0
                        row['Wave 1 latency (ms)'] = np.nan
                        rowsToAppend.append(row)

        try:
            rowsToAppend = pd.concat(rowsToAppend,axis=1,ignore_index=True).T
            wavePoints = pd.concat([wavePoints,rowsToAppend],ignore_index=True)
        except ValueError:
            pass
        
        return wavePoints
    
    def plotResultsCb(self):
        if not self.multipleFilesMode:
            frequenciesToPlot = np.array(self.p.keys()['Frequencies'].value().split(',')).astype(int)
            wavePoints = self.wavePoints.loc[self.wavePoints['Freq'].isin(frequenciesToPlot)]
            #find the wavepoints with nan as wave 1 amplitude and remove the row from the dataframe
            wavePoints = self.addMissingIntensities(wavePoints)

            self.resultPlot = resultWindow(wavepoints=wavePoints,kind='scatter')
            self.resultPlot.show()
        else:
            rows = []
            for j,el in self.experimentList.iterrows():
                folder, currentFile = os.path.split(el['Filename'])
                wavepoints = pd.read_csv(os.path.join(folder,os.path.splitext(currentFile)[0]+'_waveAnalysisResults.csv'))
                wavepoints['Group'] = el['Group']
                wavepoints['MouseID'] = el['MouseID']
                rows.append(wavepoints)

            allWavepoints = pd.concat(rows,ignore_index=True)
            frequenciesToPlot = np.array(self.p.keys()['Frequencies'].value().split(',')).astype(int)
            wavePoints = allWavepoints.loc[allWavepoints['Freq'].isin(frequenciesToPlot)]
            self.resultPlot = resultWindow(wavepoints=wavePoints,group='Group',kind='line')
            self.resultPlot.show()

    def plotThresholdsCb(self):
        if not self.multipleFilesMode:
            allThresholds = pd.DataFrame({'Freq': self.threshDict.index.astype(int), 
                                        'Threshold': self.threshDict.values})
            print(allThresholds)
            self.thresholdPlot = resultThresholdWindow(allThresholds, group=None)
            self.thresholdPlot.show()
        else:
            rows = []
            for j,el in self.experimentList.iterrows():
                #in multipleFilesMode, the thresholds are not saved in the file, but in the experiment list
                thresholds = self.experimentList.loc[j,self.frequencies.astype(int).astype(str)]
                threshDict = pd.Series(thresholds,index=self.frequencies.astype(int).astype(str))
                #replace the NaNs with 0
                threshDict = threshDict.fillna(0)
            
                allThresholds = pd.DataFrame({'Freq': threshDict.index.astype(int), 
                                                        'Threshold': threshDict.values})
                allThresholds['Group'] = el['Group']
                allThresholds['MouseID'] = el['MouseID']
                rows.append(allThresholds)

            allThresholds = pd.concat(rows,ignore_index=True)
            self.thresholdPlot = resultThresholdWindow(allThresholds,group='Group')
            self.thresholdPlot.show()    

    def onMouseClicked(self,evt):
        if evt.double():
            pass#
            '''
            items = self.sc.items(evt.scenePos())
            for item in items:
                if isinstance(item,pg.PlotItem):
                    plItem = self.plotDict [item.objectName()]
                    row, col = [int(ee) for ee in item.objectName().split(' ')]
                    plItem.setPen('r')
                    for i in range(row):
                        msg = str(i)+' '+str(col)
                        plItem = self.plotDict[msg]
                        plItem.setPen('r')
                    for i in range(row+1,20):
                        msg = str(i)+' '+str(col)
                        plItem = self.plotDict[msg]
                        plItem.setPen('k')
            '''
        else:
            items = self.sc.items(evt.scenePos())
            for item in items:
                if isinstance(item,pg.PlotItem):
                    #msg =  str(self.activeRowCol[0])+' '+str(self.activeRowCol[1])
                    plItem = self.plotDict[item.objectName()]
                    
              
            
                    row, col = [int(ee) for ee in item.objectName().split(' ')]
            self.setActivePlot(row,col)


    
    def guessWavePoints(self, direction = 'both', peaks=None):
        """Propagate wave points to other traces.
        
        Args:
            direction: 'both', 'higher', or 'lower' - which intensities to propagate to
            peaks: Optional list of peak names (e.g., ['P1']) to propagate. 
                   If None, propagates all peaks (P1-P4, N1-N4).
        """
        freq,initialIntens = self.plotToFreqIntMap[self.activeRowCol]
        threshold = self.threshDict[str(int(freq))]
        
        if direction == 'both':
            lowerLimit = threshold
            higherLimit = 100
        elif direction =='higher':
            lowerLimit = initialIntens
            higherLimit = 100
        elif direction == 'lower':
            lowerLimit = threshold
            higherLimit = initialIntens

        # Determine which peaks to process
        if peaks is None:
            peaksToProcess = ['P1', 'N1', 'P2', 'N2', 'P3', 'N3', 'P4', 'N4', 'P5', 'N5']
        else:
            peaksToProcess = peaks

        activeRow = self.activeRowCol[0]
        activeCol = self.activeRowCol[1]
        for row in np.arange(20):
            if row!=activeRow:
                
                initialWavePoints = self.wavePoints.loc[(self.wavePoints['Freq']==freq) & (self.wavePoints['Intensity']==initialIntens) ]

                try:
                    trace = self.getTraceAt(row,activeCol)[1]
                    freq, intens =  self.plotToFreqIntMap[(row,activeCol)]

                    
                    if (intens>=lowerLimit) & (intens<=higherLimit):
                        print((freq,intens))
                        points = {}
                        for peak in [1,2,3,4,5]:
                            # Only process P peaks that are in peaksToProcess
                            if 'P'+str(peak) in peaksToProcess:
                                try:
                                    initialGuess_x = int(initialWavePoints['P'+str(peak)+'_x'].values[0]*self.fs/1000)
                                    guess_x = findNearestPeak(trace, initialGuess_x)    
                                    points['P'+str(peak)] = (guess_x/self.fs*1000,trace[guess_x])
                                except ValueError:
                                    points['P'+str(peak)] = (np.nan,np.nan)

                            # Only process N peaks that are in peaksToProcess
                            if 'N'+str(peak) in peaksToProcess:
                                try:
                                    initialGuess_x = int(initialWavePoints['N'+str(peak)+'_x'].values[0]*self.fs/1000)
                                    guess_x = findNearestPeak(trace, initialGuess_x, negative=True  )
                                    points['N'+str(peak)] = (guess_x/self.fs*1000,trace[guess_x])
                                except ValueError:
                                    points['N'+str(peak)] = (np.nan,np.nan)

                        # check if the entry exists and update it or create it :
                        selectedWavePoints = self.wavePoints.loc[(self.wavePoints['Freq']==freq) & (self.wavePoints['Intensity']==intens) ]

                        initialIntens = intens
                        
                        if selectedWavePoints.shape[0] ==0:
                            # Create new row with only the peaks we processed
                            newRow = {'Freq':freq, 'Intensity':intens}
                            for peakName in points:
                                newRow[peakName+'_x'] = points[peakName][0]
                                newRow[peakName+'_y'] = points[peakName][1]
                            self.wavePoints = pd.concat([self.wavePoints,
                                pd.DataFrame(newRow, index=[0])], ignore_index=True, axis=0)
                        elif selectedWavePoints.shape[0] ==1:
                            # Only update the peaks that were processed
                            for peakName in points:
                                self.wavePoints.loc[(self.wavePoints['Freq']==freq) & (self.wavePoints['Intensity']==intens), peakName+'_x'] = points[peakName][0]
                                self.wavePoints.loc[(self.wavePoints['Freq']==freq) & (self.wavePoints['Intensity']==intens), peakName+'_y'] = points[peakName][1]
                except KeyError:
                    pass

        self.updateCurrentPlotCb()

        #print(sc.items())
    def setThresholdCb(self):
        freq,initialIntens = self.plotToFreqIntMap[self.activeRowCol]
        self.threshDict[str(int(freq))] = initialIntens
        self.updateCurrentPlotCb()

    def setAboveThresholdCb(self):
        freqs = []
        intens = []
        for el in self.abr.index:
            freqs.append(el[0])
            intens.append(el[1])

        freq,initialIntens = self.plotToFreqIntMap[self.activeRowCol]
        self.threshDict[str(int(freq))] = max(intens)+5
        self.updateCurrentPlotCb()

    def resetPointBelowCb(self, peakName):
        """Reset a specific peak for the current intensity and all lower intensities.
        
        Args:
            peakName: The peak to reset, e.g., 'P1', 'N1', etc.
        """
        freq, initialIntens = self.plotToFreqIntMap[self.activeRowCol]
        threshold = self.threshDict[str(int(freq))]
        
        # Reset the peak for all intensities from threshold up to (and including) current intensity
        for intens in self.intensities:
            if intens >= threshold and intens <= initialIntens:
                # Check if entry exists
                mask = (self.wavePoints['Freq'] == freq) & (self.wavePoints['Intensity'] == intens)
                if mask.any():
                    # Set the specific peak to NaN
                    self.wavePoints.loc[mask, peakName + '_x'] = np.nan
                    self.wavePoints.loc[mask, peakName + '_y'] = np.nan
        
        self.updateCurrentPlotCb()
        self.setActivePlot(self.activeRowCol[0], self.activeRowCol[1])

    def reversePolarityCb(self):
        
        self.abr = -self.abr
        self.updateCurrentPlotCb()
        self.setActivePlot(self.activeRowCol[0],self.activeRowCol[1])
    
    def MLGuessCB(self):
 
        import pickle
        m1model = pickle.load(open('./models/Wave1LatencyModel.pkl','rb'))
        m2model = pickle.load(open('./models/Wave1m2xModel.pkl','rb'))

        for freq in self.frequencies:
            abr = self.abr.loc[freq]
            m1x = m1model.predict(abr.values)
            m2x = m2model.predict(abr.values)


            ii = 0
            for j,el in abr.iterrows():
                if j>= self.threshDict[str(int(freq))]:
                    selectedWavePoints = self.wavePoints.loc[(self.wavePoints['Freq']==freq) & (self.wavePoints['Intensity']==j) ]
                    trace = el.values
                    #Adjust values on a maximum minimum
                    m1 = findNearestPeak(trace,int(m1x[ii]),30)
                    m2 = findNearestPeak(trace,int(m2x[ii]),30,negative=True)
                    if selectedWavePoints.shape[0] ==0:

                        row = pd.DataFrame({
                            'Freq':freq,
                            'Intensity':j,
                            'P1_x':m1/self.fs*1000,
                            'P1_y' : el.values[m1],
                            'N1_x':m2/self.fs*1000,
                            'N1_y' : el.values[m2],
                        },index=[0])
                        self.wavePoints = pd.concat([self.wavePoints,row],ignore_index=True)
                    else:
                        self.wavePoints.loc[(self.wavePoints['Freq']==freq) & (self.wavePoints['Intensity']==j) ,'P1_x'] = m1/self.fs*1000
                        self.wavePoints.loc[(self.wavePoints['Freq']==freq) & (self.wavePoints['Intensity']==j) ,'N1_x'] = m2/self.fs*1000
                        self.wavePoints.loc[(self.wavePoints['Freq']==freq) & (self.wavePoints['Intensity']==j) ,'P1_y'] = el.values[m1]
                        self.wavePoints.loc[(self.wavePoints['Freq']==freq) & (self.wavePoints['Intensity']==j) ,'N1_y'] = el.values[m2]
                ii = ii+1

        self.updateCurrentPlotCb()
        self.setActivePlot(self.activeRowCol[0],self.activeRowCol[1])
    
    def MLGuessThresholdsCb(self):
        import pickle
        model = pickle.load(open('./models/ThresholdModel.pkl','rb'))
        print('Threshold predictions\n')
        for freq in self.frequencies:
            abr = self.abr.loc[freq]
            thresholds = model.predict(abr)
            thresholdProba = model.predict_proba(abr)[:,1]
            # print(thresholdProba)
            # print(abr.index)
            # print(thresholds)
            #Define the threshold as the lowest predicted as threshold
            aboveThresh = abr.index[thresholds==1]
            belowThresh = abr.index[thresholds==0]

            aboveThreshProba = thresholdProba[thresholds==1]
            belowThreshProba = thresholdProba[thresholds==0]

            if aboveThresh.size>1:
                thresh = min(aboveThresh)
                threshprob = aboveThreshProba[np.argmin(aboveThresh)]
                threshProbaNext = aboveThreshProba[np.argpartition(aboveThresh,1)][1]
                threshProbaPrev = belowThreshProba[np.argmax(belowThresh)]
                # thresh = max(belowThresh)
                # threshprob = belowThreshProba[np.argmax(belowThresh)]
                # threshProbaNext = aboveThreshProba[np.argmin(aboveThresh)]
                # threshProbaPrev = np.nan

                print(freq,' Threshold:',thresh,' Probability: ',threshprob*100,'% (Previous lvl:',threshProbaPrev*100,'%; next lvl:',threshProbaNext*100,'%)')
            else:
                threshProbaPrev = belowThreshProba[np.argmax(belowThresh)]
                thresh = max(abr.index)+5
                print(freq,max(abr.index)+5,'(Previous lvl:',threshProbaPrev*100,'%)' )
            self.threshDict[str(int(freq))] = thresh
        self.updateCurrentPlotCb()
    
    def saveABRTracesCb(self,_):
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.Directory)

        if dlg.exec_():
            folder = dlg.selectedFiles()[0]
            # Save to csv
            filename = os.path.join(folder, os.path.splitext(self.currentFile)[0] + 'converted.csv')
            self.abr.to_csv(filename)


    def saveAverageABRTracesCb(self,_):
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.Directory)
        rows = []
        for j,el in self.experimentList.iterrows():
            self.folder, self.currentFile = os.path.split(el['Filename'])
            abr,fs = at.extractABR(os.path.join(self.folder,self.currentFile))
            abr['Group'] = el['Group']
            abr=abr.reset_index()
            rows.append(abr)
        abr_all = pd.concat(rows,ignore_index=True)
        avg_abr = abr_all.groupby(['Group','level_0','level_1']).agg(['mean','std','count'])
        # Reset the index to make the MultiIndex become regular columns
        df_reset = avg_abr.reset_index()
        # Reset index to get rid of second level

        # Melt the DataFrame
        try:
            melted_df = pd.melt(df_reset, 
                            id_vars=['Group', 'level_0', 'level_1'],
                            var_name=['time', 'metric'])
        except KeyError:
            melted_df = pd.melt(df_reset, 
                            id_vars=[('Group',''), ('level_0',''), ('level_1','')],
                          )
            melted_df.columns = ['Group', 'level_0', 'level_1', 'time', 'metric', 'value']


        # Pivot to get mean and std in separate columns
        final_df = melted_df.pivot_table(
            index=['Group', 'level_0', 'level_1', 'time'],
            columns='metric',
            values='value'
        ).reset_index()

        # Rename the columns
        final_df.columns.name = None
        final_df = final_df.rename(columns={
            'level_0': 'Frequency',
            'level_1': 'Sound level (dB SPL)',
            'time':'Time (ms)'
        })
        final_df['Time (ms)'] = final_df['Time (ms)'].astype(int)/fs*1000
        final_df.sort_values(['Group', 'Frequency', 'Sound level (dB SPL)', 'Time (ms)'], inplace=True)
        final_df

        # Get unique groups
        groups = final_df['Group'].unique()

        # For each group
        if dlg.exec_():
            folder = dlg.selectedFiles()[0]
            for group in groups:
                # Create an Excel writer object for this group
                # Remove any / or \ from group name for valid filename
                safe_group = group.replace('/', '_').replace('\\', '_')
                
                # Create Excel file with sanitized group name
                with pd.ExcelWriter(os.path.join(folder,f'{safe_group}.xlsx')) as writer:

                    # Get unique frequencies for this group
                    frequencies = final_df[final_df['Group'] == group]['Frequency'].unique()
                    
                    # For each frequency
                    for freq in frequencies:
                        # Filter data for this group and frequency
                        df_sheet = final_df[
                            (final_df['Group'] == group) & 
                            (final_df['Frequency'] == freq)
                        ]
                        # Swap the order of levels in the columns MultiIndex
                        pivoted_df = df_sheet.pivot(index=['Group', 'Frequency', 'Time (ms)'], columns='Sound level (dB SPL)', values=['mean', 'std','count'])
                        pivoted_df.columns = pivoted_df.columns.swaplevel(0, 1)
                        pivoted_df
                        # Sort the columns to group mean and std for each sound level together
                        pivoted_df = pivoted_df.sort_index(axis=1, level=0)
                        pivoted_df = pivoted_df.reorder_levels([1, 0], axis=1)
                    #    pivoted_df.columns = ['{}_{}'.format(col[0], col[1]) for col in pivoted_df.columns]

                        
                        # Save to a sheet named after the frequency
                        pivoted_df.to_excel(writer, sheet_name=f'{freq}', index=True)


    def plotAverageABRTracesCb(self,_):

        rows = []
        for j,el in self.experimentList.iterrows():
            self.folder, self.currentFile = os.path.split(el['Filename'])
            abr,fs = at.extractABR(os.path.join(self.folder,self.currentFile))
            xlim = int(self.p['X-axis lim (ms)']*fs/1000)
            abr = abr.loc[:,0:xlim]
            abr['Group'] = el['Group']
            abr=abr.reset_index()
            rows.append(abr)
        
        abr_all = pd.concat(rows,ignore_index=True)
        
        avg_abr = abr_all.groupby(['Group','level_0','level_1']).mean()
        std_abr = abr_all.groupby(['Group','level_0','level_1']).std()
        avg_abr_to_plot = []
        std_abr_to_plot = []
        for g in avg_abr.index.get_level_values('Group').unique():
            avg_abr_to_plot.append(avg_abr.loc[avg_abr.index.get_level_values('Group')==g].droplevel('Group'))
            std_abr_to_plot.append(std_abr.loc[std_abr.index.get_level_values('Group')==g].droplevel('Group'))

        self.averageABRwindow = resultAverageTraceWindow(avg_abr_to_plot,std_abr_to_plot,group_labels=avg_abr.index.get_level_values('Group').unique())
        self.averageABRwindow.show()

    def changeXlimCb(self):
        self.updateCurrentPlotCb()
        self.waveAnalysisWidget.p1.setXRange(0,self.p['X-axis lim (ms)'])

if __name__ == '__main__':
    win = abrWindow()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtWidgets.QApplication.instance().exec_()