import chz
from tinker_cookbook import model_info
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.renderers import TrainOnWhat

# Model configuration
model_name = "meta-llama/Llama-3.2-1B"
renderer_name = model_info.get_recommended_renderer_name(model_name) # Basically this will render messages into tokens and weights in correct format.

# Dataset configuration for custom jsonl file 
common_config = ChatDatasetBuilderCommonConfig(
    model_name_for_tokenizer=model_name,
    renderer_name=renderer_name,
    max_length=2048, # max token per example 
    batch_size=8, # batch size for training
    train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES, # what to train on
)

# create a dataset from the JSONL file
dataset_builder = FromConversationFileBuilder(
    common_config=common_config, 
    file_path="example-data/conversations.jsonl",
    test_size=24, # test split
    shuffle_seed=42, # for reproducibility
    )

# Training configuration
config = train.Config(
    log_path="/Users/aadityasubedi/Desktop/tinker/tinker-cookbook/training_logs/run_3", 
    model_name=model_name,
    dataset_builder=dataset_builder,
    learning_rate=5e-4, # learning rate
    lr_schedule="linear", # linear learning rate schedule
    num_epochs=3, # number of epochs
    lora_rank=32, # LoRA rank
    save_every=10, # checkpoint every 10 epochs
    eval_every=10, # evaluate every 10 epochs
    wandb_project="tinker-SFT", # weights and biases project
    wandb_name="training-example", # weights and biases run name
)

if __name__ == "__main__":
    train.main(config)
    
    