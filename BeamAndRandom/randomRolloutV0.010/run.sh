test_num=100

env_name="random"
mode="mcts_random"
for part in 50;
do
  for mach in 8 16 32;
  do
    for dist_type in h;
    do
      python testPolicy.py --mode $mode --part-num $part --mach-num $mach --env-name $env_name --test-num $test_num --dist-type $dist_type
    done
  done
done