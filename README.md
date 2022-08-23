# train-with-tsdae

[TSDAE](https://www.sbert.net/examples/unsupervised_learning/TSDAE/README.html)

[raw model](https://huggingface.co/Bingsu/my_reformer_untrained)

### Note

TSDAE로 모델을 학습하려면 모델이 다음 조건을 만족해야 한다.

1. `AutoModelForCausalLM`로 모델을 불러올 수 있어야 합니다.

2. 모델의 `forward` 함수가 `encoder_hidden_states`를 입력으로 받아야 합니다.

둘 모두를 만족하지 않는 모델을 TSDAE로 학습하려면,
`losses.DenoisingAutoEncoderLoss`에서 `tie_encoder_decoder=False`로 설정하고,
 디코더로 사용할 모델을 따로 설정해주어야 합니다.

이는 학습 성능이 낮아질 수 있습니다...
