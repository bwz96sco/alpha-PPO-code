test_num=100;
   
for part in 20 30 40 50 60 70 80 90;
do
  uv run python bboMain.py --popu 150 --iter 400 --part-num $part --mach-num 8 --dist-type h --test-num $test_num ;
done