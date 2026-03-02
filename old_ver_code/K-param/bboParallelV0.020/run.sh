test_num=100;

python bboMain.py --popu 50 --iter 100 --part-num 45 --dist-type l --test-num $test_num ;
python bboMain.py --popu 100 --iter 200 --part-num 45 --dist-type l --test-num $test_num ;
python bboMain.py --popu 100 --iter 400 --part-num 45 --dist-type l --test-num $test_num ;
