FOLDER=dupe
mkdir -p $FOLDER

# echo "Downloading UCI - Beijing PM2.5 Data Set"
# wget -P $FOLDER https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv

# echo "Downloading UCI - Individual household electric power consumption Data Set"
# wget -P $FOLDER https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip

# echo "Downloading UCI - Beijing Multi-Site Air-Quality Data Set"
# wget -P $FOLDER https://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip

# echo "Downloading UCI - Human Activity Recognition from Continuous Ambient Sensor Data Set"
# wget -P $FOLDER https://archive.ics.uci.edu/ml/machine-learning-databases/00506/casas-dataset.zip

# echo "Downloading UCI - ElectricityLoadDiagrams20112014 Data Set"
# wget -P $FOLDER https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip

# echo "Downloading UCI - PEMS-SF Data Set (traffic)"
# wget -P $FOLDER https://archive.ics.uci.edu/ml/machine-learning-databases/00204/PEMS-SF.zip

# echo "Downloading Physionet 2012"
# wget -P $FOLDER  https://physionet.org/files/challenge-2012/1.0.0/set-a.zip
# wget -P $FOLDER  https://physionet.org/files/challenge-2012/1.0.0/set-b.zip

# echo "Downloading Physionet 2019"
# wget -P $FOLDER https://archive.physionet.org/users/shared/challenge-2019/training_setA.zip
# wget -P $FOLDER https://archive.physionet.org/users/shared/challenge-2019/training_setB.zip

# echo "Downloading USHCN-daily Data Set"
# wget -P $FOLDER -r -np -nH --cut-dirs 1 https://cdiac.ess-dive.lbl.gov/ftp/ushcn_daily/

# echo "Downloading UWAVE Data Set"
# wget -P $FOLDER http://timeseriesclassification.com/Downloads/UWaveGestureLibrary.zip

# echo "Downloading M3 Data Set"
# wget -P $FOLDER https://forecasters.org/data/m3comp/M3C.xls

# echo "Downloading M4 Data Set"
# wget -P $FOLDER https://www.m4.unic.ac.cy/wp-content/uploads/2017/12/M4DataSet.zip
# wget -P $FOLDER https://www.m4.unic.ac.cy/wp-content/uploads/2018/07/M-test-set.zip


echo "Downloading M4 Data set"
svn export "https://github.com/Mcompetitions/M4-methods/trunk/Dataset" "$FOLDER/M4"

echo "Downloading Kaggle - M5 Data Set"
kaggle competitions download -p $FOLDER -c m5-forecasting-accuracy

echo "Downloading Kaggle - Tourism Forecasting Data Set"
kaggle competitions download -p $FOLDER -c tourism1

echo "Downloading Kaggle - Tourism Forecasting Data Set 2"
kaggle competitions download -p $FOLDER -c tourism2
kaggle datasets download -d yogesh94/m4-forecasting-competition-dataset