from PyQt5.Qt import QApplication
from PyQt5 import QtWidgets
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class resultWindow(QtWidgets.QMainWindow):
    def __init__(self, wavepoints,group=None,kind='scatter'):
        '''
        Kind can be 'scatter' or 'line'
        '''

        super().__init__()

        self.setWindowTitle('ABR Wave Analysis - Results') 
        self.setGeometry(0,0,1300,1000)
        # Center the window on the screen
        screen = QApplication.primaryScreen().geometry()
        self.move(screen.center().x() - self.width()//2,
                 screen.center().y() - self.height()//2)
        self.main_widget = QtWidgets.QWidget(self)

        wavepoints2 = wavepoints.copy()
        wavepoints2['Latency (ms)'] = wavepoints2['P1_x']
        wavepoints2['Amplitude (uV)'] = np.abs(wavepoints2['P1_y']-wavepoints2['N1_y'])
        fg = sns.relplot(data=wavepoints2,x='Intensity',y='Latency (ms)',col='Freq',hue=group,kind=kind,errorbar='sd')
        # Superimpose a scatter plot using seaborn
        if kind == 'line':
            for ax in fg.axes.flat:
                freq = float(ax.get_title().split(' = ')[1])
                data = wavepoints2[wavepoints2['Freq'] == freq]
                sns.scatterplot(data=data, x='Intensity', y='Latency (ms)', hue=group, style='MouseID', ax=ax, legend=False)

#                ax.get_legend().remove()

        self.fig = fg.figure
        self.fig.tight_layout()
        
        self.canvas = FigureCanvas(self.fig)

        fg2 = sns.relplot(data=wavepoints2,x='Intensity',y='Amplitude (uV)',col='Freq',hue=group,kind=kind,errorbar='sd')
        if kind == 'line':
            for ax in fg2.axes.flat:
                freq = float(ax.get_title().split(' = ')[1])
                data = wavepoints2[wavepoints2['Freq'] == freq]
                sns.scatterplot(data=data, x='Intensity', y='Amplitude (uV)', hue=group, style='MouseID', ax=ax, legend=False)


        self.fig2 = fg2.figure
        self.fig2.tight_layout()
        self.canvas2 = FigureCanvas(self.fig2)

        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                       QtWidgets.QSizePolicy.Expanding)
        self.canvas.updateGeometry()

        self.canvas2.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                       QtWidgets.QSizePolicy.Expanding)
        self.canvas2.updateGeometry()
        # self.button = QtWidgets.QPushButton("Button")
        # self.label = QtWidgets.QLabel("A plot:")

        self.layout = QtWidgets.QGridLayout(self.main_widget)
        # self.layout.addWidget(self.button)
        # self.layout.addWidget(self.label)
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.canvas2)
        self.setCentralWidget(self.main_widget)

class resultThresholdWindow(QtWidgets.QMainWindow):
    def __init__(self,thresholds,group=None):
        super().__init__()

        self.setWindowTitle('ABR thresholds - Results') 
        self.setGeometry(0,0,1300,1000)
        # Center the window on the screen
        screen = QApplication.primaryScreen().geometry()
        self.move(screen.center().x() - self.width()//2,
                 screen.center().y() - self.height()//2)
        self.main_widget = QtWidgets.QWidget(self)

        fg = sns.catplot(data=thresholds,x='Freq',y='Threshold',hue=group,kind='point',errorbar='sd')
        self.fig = fg.figure
        if group is not None:
            thresholds2 = thresholds.copy()
            thresholds2['FreqCat'] = thresholds2['Freq'].map({k:i for i,k in enumerate(sorted(thresholds2['Freq'].unique()))})
            for ax in fg.axes.flat:
                sns.lineplot(data=thresholds2, x='FreqCat', y='Threshold', hue=group, ax=ax, units='MouseID', estimator=None, alpha=0.2, legend=False)
                sns.scatterplot(data=thresholds2, x='FreqCat', y='Threshold', hue=group, style='MouseID', s=100, ax=ax, alpha=0.2, legend=False) 

        self.fig.tight_layout()
        
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                       QtWidgets.QSizePolicy.Expanding)
        self.canvas.updateGeometry()

        self.layout = QtWidgets.QGridLayout(self.main_widget)
        self.layout.addWidget(self.canvas)
        self.setCentralWidget(self.main_widget)

class resultAverageTraceWindow(QtWidgets.QMainWindow):
    def __init__(self,abr_traces,abr_traces_err=None,group_labels=None):
        super().__init__()

        self.setWindowTitle('ABR avg traces - Results') 
        self.setGeometry(0,0,1300,1000)
        # Center the window on the screen
        screen = QApplication.primaryScreen().geometry()
        self.move(screen.center().x() - self.width()//2,
                 screen.center().y() - self.height()//2)
        self.main_widget = QtWidgets.QWidget(self)

        #fg = sns.relplot(data=abr_traces,x='Time (ms)',y='Amplitude (uV)',col='Frequency',row='Sound level (dB SPL)',hue=group,kind='line',errorbar='sd')
        fig = None
        axs = None
        colors = ['#DC3220', '#005AB5', '#FFA500', '#009E73', '#9370DB', '#E6AB02', '#56B4E9', '#8B0000', '#00BFFF', '#32CD32',
                  '#FF69B4', '#4B0082', '#FF4500', '#008080', '#FFD700', '#8B4513', '#00CED1', '#FF1493', '#556B2F', '#9932CC']
        for j,abr_trace in enumerate(abr_traces):
            fig,axs = makeFigureErrorBar(abr_trace,err=abr_traces_err[j],fig=fig,axs=axs,linecolor=colors[j])
        self.fig = fig
        if group_labels is not None:
            for i in range(len(group_labels)):
                self.fig.text(0.9, 0.1-0.015*i, group_labels[i], ha='left', va='center', fontsize=12, fontweight='bold',color=colors[i])
        self.fig.tight_layout()

        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                       QtWidgets.QSizePolicy.Expanding)
        self.canvas.updateGeometry()

        self.layout = QtWidgets.QGridLayout(self.main_widget)
        self.layout.addWidget(self.canvas)
        self.setCentralWidget(self.main_widget)


def makeFigureErrorBar(abr_df,title='',fig=None,axs=None,linecolor = 'k',err=None):
    '''
    Make a figure from ABR trace data
    '''

    h1 = abr_df.index.get_level_values(0)
    h2 = abr_df.index.get_level_values(1)
    out = abr_df.values

    frequency = list(set(h1))
    frequency.sort()
    intensity = list(set(h2))
    intensity.sort()
    nint = len(intensity)
    nfreq=len(frequency)
    freqmap=dict(zip(frequency,np.arange(len(frequency))))
    imap = dict(zip(intensity,np.arange(len(intensity))))
    if nint==1:
        nint=2
    if nfreq==1:
        nfreq=2
    if fig is None:
        fig,axs=plt.subplots(nint,nfreq,sharex=False, sharey=False,subplot_kw={'xticks': [], 'yticks': []},figsize=np.array([ 15.8 ,  16.35]))
    for i in range(len(h1)):
        column = freqmap[int(h1[i])]
        row = imap[int(h2[i])]
        #plotn = i+row*len(frequency)
        linecol = linecolor

 
        axs[nint-row-1,column].plot(np.array(out)[i,:],c=linecol,linewidth=2)

        if err is not None:
            axs[nint-row-1,column].fill_between(np.arange(out.shape[1]),np.array(out)[i,:]-np.array(err)[i,:],np.array(out)[i,:]+np.array(err)[i,:],
                                                alpha=0.15,color = linecol)
    #axs[nint-row-1,column].set_ylim((array(out).min(),array(out).max()))
      

    for i in range(len(h1)):
        row = imap[int(h2[i])]

        axs[nint-row-1,0].set_ylabel(str(int(h2[i]))+' dB')

    for i in range(len(h1)):
        tit1 = int(h1[i])
        column = freqmap[int(h1[i])]    
        if tit1 == 100:
            tit='Click'
        else:
            tit = str(int(tit1/1000))+' kHz'
        axs[0,column].set_title(tit,fontsize=14,ha='center')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(wspace=0.05,hspace=0,left=+.035)


    for ax in fig.axes:
       ax.axis('off')

    plt.tight_layout()
    return fig,axs

