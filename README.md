This is the code for dropout prediction in our AAAI'19 paper:

Wenzheng Feng, Jie Tang, Tracy Xiao Liu, Shuhuai Zhang, Jian Guan. [Understanding Dropouts in MOOCs](http://keg.cs.tsinghua.edu.cn/jietang/publications/AAAI19-Feng-dropout-moocs.pdf). In Proceedings of the 33rd AAAI Conference on Artificial Intelligence (AAAI'19).

## How to run

```bash
# download data from www.moocdata.org
sh dump_data.sh

# extract basic activity features from log file
python feat_extract.py

# integrate different types of features
python preprocess.py

# run CFIN model
python main.py
```
