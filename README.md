Music generation


1． 根据1、2个小结音乐生成同样长度的音乐。
2． 生成的旋律与输入旋律听上去有相关性。


Step1: 安装 midi
(tensorflow_p27) [ec2-user@ip-172-31-31-213 test1]$ cd python-midi-master
$python setup.py install

Step2:在 generated 文件夹中放入 1、 2 个小结音乐

Step3: 运行
(tensorflow_p27) [ec2-user@ip-172-31-31-213 test1]$ python test1.py

Step4 在 generated 文件夹中生成 gen_song_0.mid

