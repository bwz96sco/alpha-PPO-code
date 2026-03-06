test_num=100;
   
for part in 20 30 40 50 60 70 80 90;
do
  for k in "mp" ;
  do
    uv run python ruleTest.py --mach-num 8 --mode $k --part-num $part --dist-type h --test-num $test_num ;
  done
done