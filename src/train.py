from datetime import datetime

import pandas as pd
import torch
from datasets import load_dataset
from loguru import logger
from sentence_transformers import InputExample, SentenceTransformer, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader

from .dataset import DenoisingAutoEncoderIterableDataset


def train(
    model_name: str = "Bingsu/my_reformer_untrained",
    train_batch_size: int = 16,
    num_epochs: int = 1,
    steps_per_epoch: int = 0,
    max_seq_length: int = 16384,
    evaluation_steps: int = 5000,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 1000,
    scheduler: str = "warmuplinear",
):

    # Save path to store our model
    model_save_path = "tsdae/{}-{}-{}".format(
        model_name.split("/")[-1],
        train_batch_size,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )

    # Defining our sentence transformer model
    logger.info("Loading model")
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(), "cls"
    )
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # Load train dataset
    logger.info("Loading train dataset")
    dataset = load_dataset(
        "Bingsu/my-korean-training-corpus",
        split="train",
        use_auth_token=True,
        streaming=True,
    )

    # Read STSbenchmark dataset and use it as development set
    logger.info("Read STSbenchmark dev dataset")
    dev_samples = []
    test_samples = []

    dev_df = pd.read_csv(
        "https://raw.githubusercontent.com/Bing-su/klue-sts-csv-data/main/data/klue_sts_train.csv"
    )
    test_df = pd.read_csv(
        "https://raw.githubusercontent.com/Bing-su/klue-sts-csv-data/main/data/klue_sts_val.csv"
    )
    dev_df["label"] = dev_df["label"] / 5.0
    test_df["label"] = test_df["label"] / 5.0

    for _, row in dev_df.iterrows():
        dev_samples.append(
            InputExample(texts=[row["sentence1"], row["sentence2"]], label=row["label"])
        )

    for _, row in test_df.iterrows():
        test_samples.append(
            InputExample(texts=[row["sentence1"], row["sentence2"]], label=row["label"])
        )

    dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        dev_samples, batch_size=train_batch_size, name="sts-dev"
    )
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        test_samples, batch_size=train_batch_size, name="sts-test"
    )

    # We train our model using the MultipleNegativesRankingLoss
    train_dataset = DenoisingAutoEncoderIterableDataset(dataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        drop_last=True,
    )
    train_loss = losses.DenoisingAutoEncoderLoss(
        model, decoder_name_or_path=model_name, tie_encoder_decoder=True
    )

    logger.info(f"Training steps: {num_epochs * steps_per_epoch}")
    logger.info("Performance before training")
    dev_evaluator(model)

    # length of IterableDataset
    len_dataset = 213867576
    steps_per_epoch = len_dataset // train_batch_size

    # Train the model
    logger.info("Training start")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=dev_evaluator,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        evaluation_steps=evaluation_steps,
        output_path=model_save_path,
        weight_decay=weight_decay,
        scheduler=scheduler,
        warmup_steps=warmup_steps,
        optimizer_class=torch.optim.RAdam,
        optimizer_params={"lr": learning_rate},
        use_amp=True,  # Set to True, if your GPU supports FP16 cores
    )
    logger.info("Training end")

    ##############################################################################
    #
    # Load the stored model and evaluate its performance on STS benchmark dataset
    #
    ##############################################################################

    model = SentenceTransformer(model_save_path)
    test_evaluator(model, output_path=model_save_path)
