
###========================================
### parameters settting examples
###========================================
python ReHash_CIFAR_10.py \
--name ReHash_cifar_alex_test \
--bits 12 24 32 48 \
--arch alexnet \
--batch-size 32 \
--max-iter 50 \
--epoch 3 \
--learning-rate 0.0001 \
--update-step 30 50 \
--discrete-iter 2 \
--weight_rh 1.0 \
--weight_sql 1.0 \
--temp 10 \
--alpha 5.5 \
--margin 4.0;
