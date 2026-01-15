test_num=20;

for mach in 9 11;
do
  for k in "wspt"  "wco" ;
  do
    python ruleTest.py --mach-num $mach --mode $k --part-num 35 --dist-type h --test-num $test_num ;
  done
done
 
