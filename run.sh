CUDA_VISIBLE_DEVICES=0 python inference_local.py --model_name=openai-community/gpt2 --dataset_type=time_t --sanity_check

CUDA_VISIBLE_DEVICES=0 python inference_nli.py --model_name=facebook/bart-large-mnli --dataset_type=time_t --sanity_check

python inference.py \
    --model_name="gpt-35-turbo" \
    --dataset_type=time_t \
    --prompt_mode=ZS_vanilla \
    --api=azure \
    --model_on_api \
    --sanity_check

CUDA_VISIBLE_DEVICES=0 python inference_local.py --model_name=openai-community/gpt2 --dataset_type=time_t1 --sanity_check

CUDA_VISIBLE_DEVICES=0 python inference_nli.py --model_name=facebook/bart-large-mnli --dataset_type=time_t1 --sanity_check

python inference.py \
    --model_name="gpt-35-turbo" \
    --dataset_type=time_t1 \
    --prompt_mode=ZS_vanilla \
    --api=azure \
    --model_on_api \
    --sanity_check
