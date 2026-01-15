test_num=100;
   
#for mach in 9 11 13 15 17 19 ;
#do
#  python gaMain.py --mach-num $mach --part-num 35 --dist-type h --test-num $test_num ;
#done
python gaMain.py --part-num 15 --dist-type l --iter 285 --popu 285  --test-num $test_num ;
python gaMain.py --part-num 15 --dist-type m --iter 400 --popu 200  --test-num $test_num ;
#python gaMain.py --part-num 15 --dist-type h --iter 400 --popu 200  --test-num $test_num ;

