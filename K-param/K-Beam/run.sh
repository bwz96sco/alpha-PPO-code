test_num=100

env_name="K-beam-ex"

for K in 2 4 7 10 13 17 20;
do
  python testPolicy.py --mode K2 --env-name $env_name  --beam-size $K --test-num $test_num
done


