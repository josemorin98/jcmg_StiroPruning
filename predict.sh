cd src
python predict.py --modelo use --params separate_grid --embeddings_path "../test/embeddings/use" --models_dir "../test/Modelos" --use_adjusted > ../test/prediction/log_use.txt
python predict.py --modelo st1 --params separate_grid --embeddings_path "../test/embeddings/st1" --models_dir "../test/Modelos" --use_adjusted > ../test/prediction/log_st1.txt
python predict.py --modelo st2 --params separate_grid --embeddings_path "../test/embeddings/st2" --models_dir "../test/Modelos" --use_adjusted > ../test/prediction/log_st2.txt
python predict.py --modelo st3 --params separate_grid --embeddings_path "../test/embeddings/st3" --models_dir "../test/Modelos" --use_adjusted > ../test/prediction/log_st3.txt
