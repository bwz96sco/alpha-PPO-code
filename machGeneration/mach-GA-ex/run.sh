test_num=100;
   
for mach in 9 11 13 15 17 19 ;
do
  python gaMain.py --mach-num $mach --part-num 35 --dist-type h --test-num $test_num ;
done