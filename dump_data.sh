echo "Download prediction data from moocdata.org"
wget http://lfs.aminer.cn/misc/moocdata/data/prediction_data.tar.gz
wget http://lfs.aminer.cn/misc/moocdata/data/user_info.csv
wget http://lfs.aminer.cn/misc/moocdata/data/course_info.csv

echo "Extract files from prediction_data.tar.gz"
tar -xzvf prediction_data.tar.gz 

