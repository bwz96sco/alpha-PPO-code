test_num=20;

for mach in 9 11;
do
  python gaMain.py --mach-num $mach --part-num 35 --dist-type h --test-num $test_num ;
done
 
