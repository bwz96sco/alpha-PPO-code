test_num=2

env_name="K-beam-ex"

for K in 2 4;
do
  python testPolicy.py --mode K2 --env-name $env_name  --beam-size $K --test-num $test_num
done


