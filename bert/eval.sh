#!/bin/sh
OUTPUT_DIR=/fs/class-projects/spring2021/cmsc723/c723g001/bert/evals
python /fs/class-projects/spring2021/cmsc723/c723g001/bert/transformers/examples/question-answering/run_qa.py \
--model_name_or_path 'fine_tune_output' \
--do_eval --dataset_name squad_v2  \
--max_seq_length 384   --doc_stride 128 \
--overwrite_output_dir \
--output_dir $OUTPUT_DIR \
--version_2_with_negative \
# --max_val_samples 10