test_num=100;
dist=h

(for part in 15 25 35 45 65 95 125 ;
do
  python bboMain.py --popu 100 --iter 600 --part-num $part --dist-type $dist --test-num $test_num
done)&
(for part in 15 25 35 45 65 95 125 ;
do
  python bboMain.py --popu 50 --iter 1200 --part-num $part --dist-type $dist --test-num $test_num
done)& 
wait