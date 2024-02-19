Requirements: requirements.txt

First, run bert_training.py :
python3 bert_training.py \
            --gpus 1 \
            --train_csv train_data_bert.csv \
            --test_csv test_data_bert.csv \
            --dev_csv val_data_bert.csv \
            --encoder_model bert-base-uncased \
            --min_epochs 5 \
            --nr_frozen_epochs 3 \
            --max_epochs 20 \
            --patience 3 \
            --out final_data/predictions

Then run final_prediction.py :
CUDA_VISIBLE_DEVICES="" python3 final_prediction.py \
    --input test_data_bert.csv \
    --experiment final_data/predictions

