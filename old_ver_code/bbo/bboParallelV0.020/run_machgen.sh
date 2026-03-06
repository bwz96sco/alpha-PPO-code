test_num=100;

for mach in 10 12 14 16 18 20 22;
do
  uv run python bboMain.py --popu 150 --iter 400 --part-num 50 --mach-num $mach --dist-type h --test-num $test_num ;
done

   
