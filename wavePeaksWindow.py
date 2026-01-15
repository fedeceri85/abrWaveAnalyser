from math import nan
from PyQt5.QtCore import Qt
import pyqtgraph as pg
#from PyQt5.Qt import QApplication,QGridLayout , QWidget
from pyqtgraph.Qt import QtGui,QtCore 
import numpy as np
from pyqtgraph.parametertree import Parameter, ParameterTree
import pandas as pd

pg.setConfigOptions(antialias=True)
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

fs = 195000.0/2.0 # Acquisition sampling rate


def findNearestPeak(trace,guess,window=20,negative=False):
    if negative == False:
        return np.argmax(trace[guess-window:guess+window])+(guess-window)
    else:
        return np.argmin(trace[guess-window:guess+window])+(guess-window)




#generate layout

class myGLW(pg.GraphicsLayoutWidget,QtCore.QObject):
    finishSignal = QtCore.pyqtSignal()
    changeTraceSignal = QtCore.pyqtSignal(str)
    guessAboveSignal = QtCore.pyqtSignal()
    guessBelowSignal = QtCore.pyqtSignal()
    propagatePointSignal = QtCore.pyqtSignal(str)
    resetPointBelowSignal = QtCore.pyqtSignal(str)

    def __init__(self, parent=None, show=True, size=None, title=None, **kargs):
        super().__init__(parent, show, size, title, **kargs)
        self.setWindowTitle('Current waveform')
        self.setGeometry(1300,0,600,500)
        self.label = pg.LabelItem(justify='right')
        self.addItem(self.label)

        self.fs = fs
        self.p1 = self.addPlot(row=1, col=0)
        self.p1.setAutoVisible(y=True)
        self.initPlot()
        self.vb = self.p1.vb



        self.initParameterWindow()
        self.makeConnections()

        
    def initPlot(self):


        self.p1.setXRange(0,8)

        self.data1 = pd.read_csv('testTrace.csv').values[:,1]#abr.values[155,:]

        self.times = np.arange(self.data1.shape[0])/self.fs*1000

        self.linePlot = self.p1.plot(self.times,self.data1, pen="k")
        self.initPlotLabels()

    def initPlotLabels(self):
        font=QtGui.QFont()
        font.setPixelSize(20)
        p1Label = pg.TextItem("P1",color = 'k')
        p1Label.setFont(font)
        p1Point = pg.ScatterPlotItem(x=[],y=[])
        p1Label.setPos(np.nan,np.nan)
        self.p1.addItem(p1Label)
        self.p1.addItem(p1Point)
        p1Point.setData([0],[0])

        p2Label = pg.TextItem("P2",color = 'k')
        p2Label.setFont(font)
        p2Point = pg.ScatterPlotItem(x=[],y=[])
        p2Label.setPos(np.nan,np.nan)
        self.p1.addItem(p2Label)
        self.p1.addItem(p2Point)
        p2Point.setData([0],[0])

        p3Label = pg.TextItem("P3",color = 'k')
        p3Label.setFont(font)
        p3Point = pg.ScatterPlotItem(x=[],y=[])
        p3Label.setPos(np.nan,np.nan)
        self.p1.addItem(p3Label)
        self.p1.addItem(p3Point)
        p3Point.setData([0],[0])

        p4Label = pg.TextItem("P4",color = 'k')
        p4Label.setFont(font)
        p4Point = pg.ScatterPlotItem(x=[],y=[])
        p4Label.setPos(np.nan,np.nan)
        self.p1.addItem(p4Label)
        self.p1.addItem(p4Point)
        p4Point.setData([0],[0])

        n1Label = pg.TextItem("N1",color = 'k')
        n1Label.setFont(font)
        n1Point = pg.ScatterPlotItem(x=[],y=[],symbol='+',pen=pg.mkPen('r'),size=13,brush=pg.mkBrush('r'))
        n1Label.setPos(np.nan,np.nan)
        self.p1.addItem(n1Label)
        self.p1.addItem(n1Point)
        n1Point.setData([0],[0])

        n2Label = pg.TextItem("N2",color = 'k')
        n2Label.setFont(font)
        n2Point = pg.ScatterPlotItem(x=[],y=[],symbol='+',pen=pg.mkPen('r'),size=13,brush=pg.mkBrush('r'))
        n2Label.setPos(np.nan,np.nan)
        self.p1.addItem(n2Label)
        self.p1.addItem(n2Point)
        n2Point.setData([0],[0])

        n3Label = pg.TextItem("N3",color = 'k')
        n3Label.setFont(font)
        n3Point = pg.ScatterPlotItem(x=[],y=[],symbol='+',pen=pg.mkPen('r'),size=13,brush=pg.mkBrush('r'))
        n3Label.setPos(np.nan,np.nan)
        self.p1.addItem(n3Label)
        self.p1.addItem(n3Point)
        n3Point.setData([0],[0])

        n4Label = pg.TextItem("N4",color = 'k')
        n4Label.setFont(font)
        n4Point = pg.ScatterPlotItem(x=[],y=[],symbol='+',pen=pg.mkPen('r'),size=13,brush=pg.mkBrush('r'))
        n4Label.setPos(np.nan,np.nan)
        self.p1.addItem(n4Label)
        self.p1.addItem(n4Point)
        n4Point.setData([0],[0])

        p5Label = pg.TextItem("P5",color = 'k')
        p5Label.setFont(font)
        p5Point = pg.ScatterPlotItem(x=[],y=[])
        p5Label.setPos(np.nan,np.nan)
        self.p1.addItem(p5Label)
        self.p1.addItem(p5Point)
        p5Point.setData([0],[0])

        n5Label = pg.TextItem("N5",color = 'k')
        n5Label.setFont(font)
        n5Point = pg.ScatterPlotItem(x=[],y=[],symbol='+',pen=pg.mkPen('r'),size=13,brush=pg.mkBrush('r'))
        n5Label.setPos(np.nan,np.nan)
        self.p1.addItem(n5Label)
        self.p1.addItem(n5Point)
        n5Point.setData([0],[0])


        self.labelDict = {'P1':p1Label,'P2':p2Label,'P3':p3Label,'P4':p4Label,'P5':p5Label,'N1':n1Label,'N2':n2Label,'N3':n3Label,'N4':n4Label,'N5':n5Label}
        self.pointDict =  {'P1':p1Point,'P2':p2Point,'P3':p3Point,'P4':p4Point,'P5':p5Point,'N1':n1Point,'N2':n2Point,'N3':n3Point,'N4':n4Point,'N5':n5Point}

    def initParameterWindow(self):
        params = [
        {'name':'Peak type','type':'list','values':['P1','N1','P2','N2','P3','N3','P4','N4','P5','N5']},
        {'name':'Auto find peak','type':'bool','value':True},
        {'name':'RESET ALL POINTS','type':'action'},
        {'name':'RESET ALL LOWER INTENSITIES','type':'action'},
        {'name':'P1','type':'group','children':
            [   {'name':'x','type':'float','value':0},
                {'name':'y','type':'float','value':0},
                {'name':'RESET','type':'action'},
                {'name':'PROPAGATE','type':'action'},
                {'name':'RESET BELOW','type':'action'},
        ]
        
        },

                {'name':'N1','type':'group','children':
            [    {'name':'x','type':'float','value':0},
                {'name':'y','type':'float','value':0},
                {'name':'RESET','type':'action'},
                {'name':'PROPAGATE','type':'action'},
                {'name':'RESET BELOW','type':'action'},
        ]
        
        },

            {'name':'P2','type':'group','children':
            [    {'name':'x','type':'float','value':0},
                {'name':'y','type':'float','value':0},
                {'name':'RESET','type':'action'},
                {'name':'PROPAGATE','type':'action'},
                {'name':'RESET BELOW','type':'action'},
        ]
        
        },
                {'name':'N2','type':'group','children':
            [    {'name':'x','type':'float','value':0},
                {'name':'y','type':'float','value':0},
                {'name':'RESET','type':'action'},
                {'name':'PROPAGATE','type':'action'},
                {'name':'RESET BELOW','type':'action'},
        ]
        
        },

            {'name':'P3','type':'group','children':
            [    {'name':'x','type':'float','value':0},
                {'name':'y','type':'float','value':0},
                {'name':'RESET','type':'action'},
                {'name':'PROPAGATE','type':'action'},
                {'name':'RESET BELOW','type':'action'},
        ]
        
        },
                {'name':'N3','type':'group','children':
            [    {'name':'x','type':'float','value':0},
                {'name':'y','type':'float','value':0},
                {'name':'RESET','type':'action'},
                {'name':'PROPAGATE','type':'action'},
                {'name':'RESET BELOW','type':'action'},
     ]
        
        },

            {'name':'P4','type':'group','children':
            [    {'name':'x','type':'float','value':0},
                {'name':'y','type':'float','value':0},
                {'name':'RESET','type':'action'},
                {'name':'PROPAGATE','type':'action'},
                {'name':'RESET BELOW','type':'action'},
        ]
        
        },



            {'name':'N4','type':'group','children':
            [    {'name':'x','type':'float','value':0},
                {'name':'y','type':'float','value':0},
                {'name':'RESET','type':'action'},
                {'name':'PROPAGATE','type':'action'},
                {'name':'RESET BELOW','type':'action'},
        ]
        
        },

            {'name':'P5','type':'group','children':
            [    {'name':'x','type':'float','value':0},
                {'name':'y','type':'float','value':0},
                {'name':'RESET','type':'action'},
                {'name':'PROPAGATE','type':'action'},
                {'name':'RESET BELOW','type':'action'},
        ]
        
        },

            {'name':'N5','type':'group','children':
            [    {'name':'x','type':'float','value':0},
                {'name':'y','type':'float','value':0},
                {'name':'RESET','type':'action'},
                {'name':'PROPAGATE','type':'action'},
                {'name':'RESET BELOW','type':'action'},
        ]
        
        },


        ]
        ## Create tree of Parameter objects
        self.p = Parameter.create(name='params', type='group', children=params)
        self.t = ParameterTree()
        self.t.setParameters(self.p, showTop=False)
        self.t.setGeometry(1600,800,300,400)
        self.t.show()

    def makeConnections(self):
        self.p1.scene().sigMouseClicked.connect(self.onClick)
        
        self.p.keys()['RESET ALL POINTS'].sigActivated.connect(self.resetAllPoints)
        self.p.keys()['RESET ALL LOWER INTENSITIES'].sigActivated.connect(self.resetAllPointsLowerInt)


        self.p.keys()['P1'].keys()['RESET'].sigActivated.connect(lambda:self.resetPoint('P1'))
        self.p.keys()['P2'].keys()['RESET'].sigActivated.connect(lambda:self.resetPoint('P2'))
        self.p.keys()['P3'].keys()['RESET'].sigActivated.connect(lambda:self.resetPoint('P3'))
        self.p.keys()['P4'].keys()['RESET'].sigActivated.connect(lambda:self.resetPoint('P4'))
        self.p.keys()['P5'].keys()['RESET'].sigActivated.connect(lambda:self.resetPoint('P5'))
        self.p.keys()['N1'].keys()['RESET'].sigActivated.connect(lambda:self.resetPoint('N1'))
        self.p.keys()['N2'].keys()['RESET'].sigActivated.connect(lambda:self.resetPoint('N2'))
        self.p.keys()['N3'].keys()['RESET'].sigActivated.connect(lambda:self.resetPoint('N3'))
        self.p.keys()['N4'].keys()['RESET'].sigActivated.connect(lambda:self.resetPoint('N4'))
        self.p.keys()['N5'].keys()['RESET'].sigActivated.connect(lambda:self.resetPoint('N5'))

        self.p.keys()['P1'].keys()['PROPAGATE'].sigActivated.connect(lambda:self.propagatePointSignal.emit('P1'))
        self.p.keys()['P2'].keys()['PROPAGATE'].sigActivated.connect(lambda:self.propagatePointSignal.emit('P2'))
        self.p.keys()['P3'].keys()['PROPAGATE'].sigActivated.connect(lambda:self.propagatePointSignal.emit('P3'))
        self.p.keys()['P4'].keys()['PROPAGATE'].sigActivated.connect(lambda:self.propagatePointSignal.emit('P4'))
        self.p.keys()['P5'].keys()['PROPAGATE'].sigActivated.connect(lambda:self.propagatePointSignal.emit('P5'))
        self.p.keys()['N1'].keys()['PROPAGATE'].sigActivated.connect(lambda:self.propagatePointSignal.emit('N1'))
        self.p.keys()['N2'].keys()['PROPAGATE'].sigActivated.connect(lambda:self.propagatePointSignal.emit('N2'))
        self.p.keys()['N3'].keys()['PROPAGATE'].sigActivated.connect(lambda:self.propagatePointSignal.emit('N3'))
        self.p.keys()['N4'].keys()['PROPAGATE'].sigActivated.connect(lambda:self.propagatePointSignal.emit('N4'))
        self.p.keys()['N5'].keys()['PROPAGATE'].sigActivated.connect(lambda:self.propagatePointSignal.emit('N5'))

        self.p.keys()['P1'].keys()['RESET BELOW'].sigActivated.connect(lambda:self.resetPointBelowSignal.emit('P1'))
        self.p.keys()['P2'].keys()['RESET BELOW'].sigActivated.connect(lambda:self.resetPointBelowSignal.emit('P2'))
        self.p.keys()['P3'].keys()['RESET BELOW'].sigActivated.connect(lambda:self.resetPointBelowSignal.emit('P3'))
        self.p.keys()['P4'].keys()['RESET BELOW'].sigActivated.connect(lambda:self.resetPointBelowSignal.emit('P4'))
        self.p.keys()['P5'].keys()['RESET BELOW'].sigActivated.connect(lambda:self.resetPointBelowSignal.emit('P5'))
        self.p.keys()['N1'].keys()['RESET BELOW'].sigActivated.connect(lambda:self.resetPointBelowSignal.emit('N1'))
        self.p.keys()['N2'].keys()['RESET BELOW'].sigActivated.connect(lambda:self.resetPointBelowSignal.emit('N2'))
        self.p.keys()['N3'].keys()['RESET BELOW'].sigActivated.connect(lambda:self.resetPointBelowSignal.emit('N3'))
        self.p.keys()['N4'].keys()['RESET BELOW'].sigActivated.connect(lambda:self.resetPointBelowSignal.emit('N4'))
        self.p.keys()['N5'].keys()['RESET BELOW'].sigActivated.connect(lambda:self.resetPointBelowSignal.emit('N5'))

    def nextPeakType(self):
        if self.p['Peak type'] == 'P1':
            self.p['Peak type'] = 'N1'
        elif self.p['Peak type'] == 'N1':
            self.p['Peak type'] = 'P2'
        elif self.p['Peak type'] == 'P2':
           self.p['Peak type'] = 'N2'
        elif self.p['Peak type'] == 'N2':
            self.p['Peak type'] = 'P3'
        elif self.p['Peak type'] == 'P3':
            self.p['Peak type'] = 'N3'
        elif self.p['Peak type'] == 'N3':
            self.p['Peak type'] = 'P4'
        elif self.p['Peak type'] == 'P4':
            self.p['Peak type'] = 'N4'
        elif self.p['Peak type'] == 'N4':
            self.p['Peak type'] = 'P5'
        elif self.p['Peak type'] == 'P5':
            self.p['Peak type'] = 'N5'
        elif self.p['Peak type'] == 'N5':
            self.p['Peak type'] = 'P1'

    def onClick(self,event):
        if event.button()==1:
            items = self.p1.scene().items(event.scenePos())
            mousePoint = self.vb.mapSceneToView(event._scenePos)
            #print(newX, mousePoint.y())
            if self.p1.sceneBoundingRect().contains(event._scenePos):
                mousePoint = self.vb.mapSceneToView(event._scenePos)
                if self.p['Auto find peak']:
                    if self.p['Peak type'].startswith('P'): 
                        index = findNearestPeak(self.data1,int(mousePoint.x()/1000*self.fs),int(206E-6*self.fs))
                    elif self.p['Peak type'].startswith('N'): 
                        index = findNearestPeak(self.data1,int(mousePoint.x()/1000*self.fs),int(206E-6*self.fs),negative=True)
                    newX = index*1000/self.fs
                else:
                    index = int(mousePoint.x()/1000*self.fs)
                    newX = mousePoint.x()

                if index > 0 and index < len(self.data1):
                    self.label.setText(
                        "<span style='font-size: 12pt'>x=%0.1f,   <span style='color: red'>y1=%0.1f</span>" % (
                        newX, self.data1[index],))

                    #ai = pg.TextItem("P1")
                    self.setPoint(self.p['Peak type'],newX,self.data1[index])
            self.finishSignal.emit()
            
    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_W:
            self.changeTraceSignal.emit('Up')
        elif ev.key() == Qt.Key_S:
            self.changeTraceSignal.emit('Down')
        elif ev.key() == Qt.Key_A:
            self.changeTraceSignal.emit('Left')        
        elif ev.key() == Qt.Key_D:
            self.changeTraceSignal.emit('Right')
        elif ev.key() == Qt.Key_R:
            self.guessAboveSignal.emit()
        
        elif ev.key() == Qt.Key_F:
            self.guessBelowSignal.emit()
        
        elif ev.key() == Qt.Key_1:
            self.p['Peak type'] = 'P1'
        elif ev.key() == Qt.Key_2:
            self.p['Peak type'] = 'N1'
        elif ev.key() == Qt.Key_3:
            self.p['Peak type'] = 'P2'
        elif ev.key() == Qt.Key_4:
            self.p['Peak type'] = 'N2'
        elif ev.key() == Qt.Key_5:
            self.p['Peak type'] = 'P3'
        elif ev.key() == Qt.Key_6:
            self.p['Peak type'] = 'N3'
        elif ev.key() == Qt.Key_7:
            self.p['Peak type'] = 'P4'
        elif ev.key() == Qt.Key_8:
            self.p['Peak type'] = 'N4'
        elif ev.key() == Qt.Key_9:
            self.p['Peak type'] = 'P5'
        elif ev.key() == Qt.Key_0:
            self.p['Peak type'] = 'N5'
        
        elif  ev.key() == Qt.Key_E:
            self.nextPeakType()

        elif (ev.key() == Qt.Key_Return) or  (ev.key() == Qt.Key_Enter) or  (ev.key() == Qt.Key_Space):
            self.finishSignal.emit()

    def resetPoint(self,point,emitSignal = True):
        self.p.keys()[point].keys()['x'].setValue(0)  
        self.p.keys()[point].keys()['y'].setValue(0)
        self.labelDict[point].setPos(np.nan,np.nan)
        self.pointDict[point].setData([0],[0])

        if emitSignal:
            self.finishSignal.emit()

    def resetAllPoints(self):
        for point in ['P1','N1','P2','N2','P3','N3','P4','N4','P5','N5']:
            self.resetPoint(point,emitSignal=False)
        self.finishSignal.emit()
    
    def resetAllPointsLowerInt(self):
        for i in range(22): # TODO: If there are more than 22 intensities this will fail
            self.changeTraceSignal.emit('Down')
            for point in ['P1','N1','P2','N2','P3','N3','P4','N4','P5','N5']:
                self.resetPoint(point,emitSignal=False)
            self.finishSignal.emit()


    def setPoint(self,point,x,y):
        self.labelDict[point].setPos(x,y)
        self.p.keys()[point].keys()['x'].setValue(x)
        self.p.keys()[point].keys()['y'].setValue(y)
        self.pointDict[point].setData(x=[x],y=[y])   

    def getPoint(self,point):
        x=self.p.keys()[point].keys()['x'].value()
        y=self.p.keys()[point].keys()['y'].value()
        if x == 0:
            x= nan
            y= nan
        return (x,y)

    def getPoints(self):
        outDict = {}
        for point in ['P1','N1','P2','N2','P3','N3','P4','N4','P5','N5']:
            outDict[point] = self.getPoint(point)
        return outDict

    def setData(self,times,trace,wavePoints=None):
 
        self.times = times
        self.data1 = trace

       

        if wavePoints is not None:
            for point in ['P1','N1','P2','N2','P3','N3','P4','N4','P5','N5']:
                if wavePoints[point+'_x'].isna().values[0]==False:
                    self.setPoint(point,wavePoints[point+'_x'].values[0],wavePoints[point+'_y'].values[0]) 
                else:
                    self.resetPoint(point,emitSignal=False)
        else:
            for point in self.pointDict.keys():
                self.resetPoint(point,emitSignal=False)

        self.linePlot.setData(self.times,self.data1)
        self.p1.setXRange(0,8)
        

        #self.p1.setXRange(0,8)
    



if __name__ == '__main__':
    import sys
    win = myGLW(show=True)
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()