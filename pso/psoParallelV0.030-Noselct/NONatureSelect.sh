test_num=100;
dist=h

(for part in 15 25 35 45 65 125 ;
do
  python main.py --popu 100 --iter 400 --part-num  $part --dist-type $dist --test-num $test_num ;
  #python bboMain.py --popu 100 --iter 600 --part-num $part --dist-type $dist --test-num $test_num
done)&
(for part in 15 25 35 45 65 125 ;
do
  python main.py --popu 200 --iter 200 --part-num  $part --dist-type $dist --test-num $test_num ;
  #python bboMain.py --popu 100 --iter 600 --part-num $part --dist-type $dist --test-num $test_num
done)&
(for part in 15 25 35 45 65 125 ;
do
  python main.py --popu 400 --iter 100 --part-num  $part --dist-type $dist --test-num $test_num ;
  #python bboMain.py --popu 100 --iter 600 --part-num $part --dist-type $dist --test-num $test_num
done)&
wait



