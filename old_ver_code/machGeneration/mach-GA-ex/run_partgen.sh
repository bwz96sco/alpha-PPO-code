test_num=100;
   
for part in 20 30 40 50 60 70 80 90;
do
  uv run python gaMain.py --mach-num 8 --part-num $part --dist-type h --test-num $test_num ;
done