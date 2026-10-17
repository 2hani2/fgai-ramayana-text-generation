### Training and test results

| model | params | epochs_run | best_epoch | train_time_s | sec_per_epoch | test_loss | test_ppl | test_acc | test_top5 |
|---|---|---|---|---|---|---|---|---|---|
| SimpleRNN | 2932545 | 8 | 6 | 78.300 | 9.800 | 4.240 | 69.422 | 0.270 | 0.469 |
| BiRNN | 4915521 | 5 | 3 | 72.700 | 14.500 | 4.379 | 79.722 | 0.256 | 0.451 |
| LSTM | 3228225 | 9 | 7 | 175.500 | 19.500 | 4.100 | 60.360 | 0.283 | 0.484 |
| GRU | 3130433 | 8 | 6 | 148.600 | 18.600 | 4.040 | 56.799 | 0.289 | 0.490 |

Unigram baseline (ignores context): perplexity 333.0, top-1 accuracy 0.117

### Text quality by temperature (mean over seeds and samples)

| model | temperature | distinct1 | distinct2 | rep_trigram | bigram_valid | copy4 |
|---|---|---|---|---|---|---|
| BiRNN | 0.500 | 0.551 | 0.850 | 0.028 | 0.946 | 0.333 |
| BiRNN | 1.000 | 0.729 | 0.957 | 0.000 | 0.730 | 0.097 |
| BiRNN | 1.500 | 0.836 | 0.986 | 0.000 | 0.483 | 0.014 |
| GRU | 0.500 | 0.564 | 0.846 | 0.056 | 0.966 | 0.435 |
| GRU | 1.000 | 0.738 | 0.955 | 0.002 | 0.773 | 0.125 |
| GRU | 1.500 | 0.827 | 0.980 | 0.000 | 0.512 | 0.009 |
| LSTM | 0.500 | 0.527 | 0.814 | 0.095 | 0.975 | 0.539 |
| LSTM | 1.000 | 0.709 | 0.964 | 0.000 | 0.812 | 0.125 |
| LSTM | 1.500 | 0.847 | 0.980 | 0.000 | 0.553 | 0.019 |
| SimpleRNN | 0.500 | 0.522 | 0.830 | 0.067 | 0.975 | 0.501 |
| SimpleRNN | 1.000 | 0.720 | 0.946 | 0.005 | 0.771 | 0.130 |
| SimpleRNN | 1.500 | 0.802 | 0.975 | 0.005 | 0.551 | 0.035 |

### Rankings (1 = best)

| model | ppl_rank (low best) | acc_rank (high best) | time_rank (low best) | bigram_valid@1.0 rank | repetition@1.0 rank (low best) |
|---|---|---|---|---|---|
| SimpleRNN | 3 | 3 | 2 | 3 | 4 |
| BiRNN | 4 | 4 | 1 | 4 | 1.5 |
| LSTM | 2 | 2 | 4 | 1 | 1.5 |
| GRU | 1 | 1 | 3 | 2 | 3 |
