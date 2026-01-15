test_num=2;
   
for part in 15 65 125 ;
do
  python bboMain.py --popu 50 --iter 100 --part-num $part --dist-type h --test-num $test_num ;
  #python bboMain.py --popu 100 --iter 200 --part-num $part --dist-type h --test-num $test_num ;
  python bboMain.py --popu 150 --iter 400 --part-num $part --dist-type h --test-num $test_num ;
done