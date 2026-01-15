test_num=100;
dist=h

(for part in 125 ;
do
  python gaMain.py --popu 100 --iter 1200 --part-num  $part --dist-type $dist --test-num $test_num ;
  #python bboMain.py --popu 100 --iter 600 --part-num $part --dist-type $dist --test-num $test_num
done)&

(for part in 125 ;
do
  python gaMain.py --popu 100 --iter 1600 --part-num  $part --dist-type $dist --test-num $test_num ;
  #python bboMain.py --popu 100 --iter 600 --part-num $part --dist-type $dist --test-num $test_num
done)&



