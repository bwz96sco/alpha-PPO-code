test_num=100;
   
for mach in 9 11 13 15 17 19 ;
do
  for k in "wspt"  "wco" ;
  do
    python ruleTest.py --mach-num $mach --mode $k --part-num 35 --dist-type h --test-num $test_num ;
  done
done