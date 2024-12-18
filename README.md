# abrWaveAnalyser
Marcotti's lab ABR wave analysis GUI. The software can be used to semi-automatically label thresholds and waves (wave I-IV supported) of auditory brainstem responses (ABRs).

## Updates
### 12/2024
- Added support for multiple-file analysis. This requires creating a csv file containing the following columns: Date, MouseID, Strain, Genotype, Group, Age, Filename and one column for each frequency, with the same name as the label in the ABR files (e.g., 12000,24000, etc..). The Filename can be either an absolute path, or relaitve to the location of the "master" csv file.

## Installation
A python environment with the following libraries is required:
-   `python 3.10`
-   `pyqt 5`
-   `pyqtgraph 0.12.4`
-   `scikit-learn`
-   `pandas`

If using anaconda, an environment containing the appropriate library can be created with the following command:

    conda create -n waveAnalysis python=3.10 pyqtgraph=0.12.4 scikit-learn pyqt=5 pandas
    

## Running the software
Run in a command/terminal window:
`python mainWindow.py`

## Keyboard shortcuts
- Moving between traces:`W A S D`
- Select peak P1-N4: `1-8`
- Select next peak: `E`
- Guess peaks at lower intensities: `F`
- Guess peaks at higher intensities: `R`
- Set threshold: `Z`

## Supported file formats
Currently, the software reads data from csv files produced by the ABR systems used in a few labs (Sheffield: Walter Marcotti's lab; Baylor: Dwayne Simmons's lab). Internally, the software transform the data into a wide-format *pandas* dataframe (number of traces (rows) X number of timepoints (columns)) with pairs of frequency-intensity as (multi) index. The software can be adapted to read any file format by implementing a reader function that returns a similarly formatted dataframe. 

## Automatic detection of peaks
Initial manual labelling of ABR peaks can be used as a starting point to automatically label peaks at higher and lower intensities. Manual adjustment of the automatic detection is sometimes required.  
A machine-learning model can be used to semi-automatically determine wave 1 position (P1 and N1) and thresholds for all traces. Use of these methods currently requires traces containing exactly 1953 points. 

## Assumptions
Latency is not calculated for traces which do not have a annotated wave 1, whereas the software assumes that the amplitude is 0 for any sound level not esplicitly annotated, unless the abr trace is not present in the data. 