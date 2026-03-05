test_num=100;
   
for mach in 10 12 14 16 18 20 22;
do
  for k in "mp" ;
  do
    uv run python ruleTest.py --mach-num $mach --mode $k --part-num 50 --dist-type h --test-num $test_num ;
  done
done