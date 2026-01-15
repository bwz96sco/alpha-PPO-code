test_num=100;
for dist in l m h;
do
  for part in 15 25 35 45 65 95 125 ;
  do
    python bboMain.py --popu 150 --iter 400 --part-num $part --dist-type $dist --test-num $test_num ;
  done
done