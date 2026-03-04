test_num=100;
   
for mach in 10 12 14 16 18 20 22;
do
  uv run python gaMain.py --mach-num $mach --part-num 50 --dist-type h --test-num $test_num ;
done