import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from census_config import *

def load_census_data():
    train_data_file = "census_data/n_train_data.csv"
    test_data_file = "census_data/n_test_data.csv"

    if os.path.exists(train_data_file) and os.path.exists(test_data_file):
        train_data = pd.read_csv('census_data/n_train_data.csv', names=CSV_HEADER)
        test_data = pd.read_csv('census_data/n_test_data.csv', names=CSV_HEADER)
    else:
        train_data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        test_data_url  = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
        train_data     = pd.read_csv(train_data_url, header=None, names=CSV_HEADER)
        test_data      = pd.read_csv(test_data_url, header=None, names=CSV_HEADER)
        test_data      = test_data[1:]
        train_data['income_bracket'] = train_data['income_bracket'].apply(lambda x: 0 if x.repalce('.', '') == ' <=50K' else 1)
        test_data['income_bracket'] = test_data['income_bracket'].apply(lambda x: 0 if x.repalce('.', '') == ' <=50K' else 1)
        train_data.to_csv(train_data_file, index=False, header=False)
        test_data.to_csv(test_data_file, index=False, header=False)

    return train_data_file, test_data_file, train_data, test_data


train_data_file, test_data_file, train_data, test_data = load_census_data()

numeric_feat     = NUMERIC_FEATURE_NAMES
categorical_feat = CATEGORICAL_FEATURE_NAMES
total_feat       = numeric_feat + categorical_feat
csv_header       = CSV_HEADER

def build_categorical_vocab(df):
    cate_feat_vocab = {}
    for col in categorical_feat:
        cate_feat_vocab[col] = []
        cate_feat_vocab[col].extend(sorted(list(df[col].unique()))) ,
    return cate_feat_vocab

cate_feat_vocab = build_categorical_vocab(train_data)

cfg = {
    # feat config
    'numeric_feat'        : numeric_feat,
    'categorical_feat'    : categorical_feat,
    'total_feat'          : total_feat,
    'csv_header'          : csv_header,
    'cate_feat_vocab'     : cate_feat_vocab,
    'target_col'          : TARGET_COL_NAME,

    # model config
    'num_transformer_blocks'    : NUM_TRANSFORMER_BLOCKS,
    'embedding_dims'            : EMBEDDING_DIMS,
    'num_heads'                 : NUM_HEADS,
    'mlp_hidden_units_factors'  : MLP_HIDDEN_UNITS_FACTORS,
    'num_mlp_blocks'            : NUM_MLP_BLOCKS,

    # training process config
    'learning_rate'             : LEARNING_RATE,
    'weight_decay'              : WEIGHT_DECAY,
    'dropout_rate'              : DROPOUT_RATE,
    'batch_size'                : BATCH_SIZE,
    'num_epochs'                : NUM_EPOCHS,
}


def get_tf_dataset_from_csv(csv_file_path, cfg, batch_size=128, shuffle=False):
    def prepare_example_update(features):
        target = features[cfg['target_col']]
        features.pop(cfg['target_col'])
        return features, target

    dataset = tf.data.experimental.make_csv_dataset(
        csv_file_path,
        batch_size=batch_size,
        column_names=cfg['csv_header'],
        header=True,
        num_epochs=1,
        na_value="?",
        shuffle=shuffle,
    ).map(prepare_example_update, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

    return dataset.cache()


def create_model_inputs(cfg):
    inputs = {}
    for feat in cfg['total_feat']:
        if feat in cfg['numeric_feat']:
            inputs[feat] = layers.Input(name=feat, shape=(), dtype=tf.float32)
        if feat in cfg['categorical_feat']:
            inputs[feat] = layers.Input(name=feat, shape=(), dtype=tf.string)
    return inputs


def encode_inputs(inputs, cfg):
    encoded_cate_feat, encoded_num_feat = [], []
    for feat in inputs:
        if feat in cfg['categorical_feat']:
            vocabulary = cfg['cate_feat_vocab'][feat]
            lookup = layers.StringLookup(vocabulary=vocabulary, mask_token=None, num_oov_indices=0, output_mode="int")
            encoded_feature = lookup(inputs[feat])
            embedding = layers.Embedding(input_dim=len(vocabulary), output_dim=cfg['embedding_dims'])
            encoded_categorical_feature = embedding(encoded_feature)
            encoded_cate_feat.append(encoded_categorical_feature)

        if feat in cfg['numeric_feat']:
            numerical_feature = tf.expand_dims(inputs[feat], -1)
            encoded_num_feat.append(numerical_feature)

    return encoded_cate_feat, encoded_num_feat


def create_mlp(hidden_units, dropout_rate, activation, normalization_layer, name=None):
    mlp_layers = []
    for units in hidden_units:
        mlp_layers.append(normalization_layer),
        mlp_layers.append(layers.Dense(units, activation=activation))
        mlp_layers.append(layers.Dropout(dropout_rate))

    return keras.Sequential(mlp_layers, name=name)

def create_baseline_model(cfg):
    inputs = create_model_inputs(cfg)
    encoded_cate_feat, encoded_num_feat = encode_inputs(inputs, cfg)

    features = layers.concatenate(encoded_cate_feat + encoded_num_feat)
    feedforward_units = [features.shape[-1]]

    for layer_idx in range(cfg['num_mlp_blocks']):
        features = create_mlp(
            hidden_units=feedforward_units,
            dropout_rate=cfg['dropout_rate'],
            activation=keras.activations.gelu,
            normalization_layer=layers.LayerNormalization(epsilon=1e-6),
            name=f"feedforward_{layer_idx}",
        )(features)

    mlp_hidden_units = [factor * features.shape[-1] for factor in cfg['mlp_hidden_units_factors']]

    features = create_mlp(
        hidden_units=mlp_hidden_units,
        dropout_rate=cfg['dropout_rate'],
        activation=keras.activations.selu,
        normalization_layer=layers.BatchNormalization(),
        name="MLP",
    )(features)

    outputs = layers.Dense(units=1, activation="sigmoid", name="sigmoid")(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def run_experiment(model, train_data_file, test_data_file, num_epochs, learning_rate, weight_decay, batch_size,):
    optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
    )

    train_dataset      = get_tf_dataset_from_csv(train_data_file, cfg, batch_size, shuffle=True)
    validation_dataset = get_tf_dataset_from_csv(test_data_file, cfg, batch_size)

    history = model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=validation_dataset,
        verbose=1
    )

    _, accuracy = model.evaluate(validation_dataset, verbose=1)

    print(f"Final Vali Acc: {round(accuracy * 100, 4)}%")
    return history

def create_tabtransformer_model(cfg, use_column_embedding=False,):
    inputs = create_model_inputs(cfg)
    encoded_cate_feat, encoded_num_feat = encode_inputs(inputs, cfg)

    encoded_categorical_features = tf.stack(encoded_cate_feat, axis=1)
    numerical_features = layers.concatenate(encoded_num_feat)

    if use_column_embedding:
        num_columns = encoded_categorical_features.shape[1]
        column_embedding = layers.Embedding(input_dim=num_columns, output_dim=cfg['embedding_dims'])
        column_indices = tf.range(start=0, limit=num_columns, delta=1)
        encoded_categorical_features = encoded_categorical_features + column_embedding(column_indices)

    for block_idx in range(cfg['num_transformer_blocks']):
        attention_output = layers.MultiHeadAttention(
            num_heads=cfg['num_heads'],
            key_dim=cfg['embedding_dims'],
            dropout=cfg['dropout_rate'],
            name=f"multihead_attention_{block_idx}",
        )(encoded_categorical_features, encoded_categorical_features)

        x = layers.Add(name=f"skip_connection1_{block_idx}")([attention_output, encoded_categorical_features])

        x = layers.LayerNormalization(name=f"layer_norm1_{block_idx}", epsilon=1e-6)(x)

        feedforward_output = create_mlp(
            hidden_units=[cfg['embedding_dims']],
            dropout_rate=cfg['dropout_rate'],
            activation=keras.activations.gelu,
            normalization_layer=layers.LayerNormalization(epsilon=1e-6),
            name=f"feedforward_{block_idx}",
        )(x)

        x = layers.Add(name=f"skip_connection2_{block_idx}")([feedforward_output, x])

        encoded_categorical_features = layers.LayerNormalization(
            name=f"layer_norm2_{block_idx}",
            epsilon=1e-6
        )(x)

    categorical_features = layers.Flatten()(encoded_categorical_features)

    numerical_features = layers.LayerNormalization(epsilon=1e-6)(numerical_features)

    features = layers.concatenate([categorical_features, numerical_features])

    mlp_hidden_units = [factor * features.shape[-1] for factor in cfg['mlp_hidden_units_factors']]

    features = create_mlp(
        hidden_units=mlp_hidden_units,
        dropout_rate=cfg['dropout_rate'],
        activation=keras.activations.selu,
        normalization_layer=layers.BatchNormalization(),
        name="MLP",
    )(features)

    outputs = layers.Dense(
        units=1,
        activation="sigmoid",
        name="sigmoid")(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

baseline_model = create_baseline_model(cfg)

# history = run_experiment(
#     model=baseline_model,
#     train_data_file=train_data_file,
#     test_data_file=test_data_file,
#     num_epochs=cfg['num_epochs'],
#     learning_rate=cfg['learning_rate'],
#     weight_decay=cfg['weight_decay'],
#     batch_size=cfg['batch_size'],
# )

tabtransformer_model = create_tabtransformer_model(cfg=cfg, use_column_embedding=True)


history = run_experiment(
    model=tabtransformer_model,
    train_data_file=train_data_file,
    test_data_file=test_data_file,
    num_epochs=cfg['num_epochs'],
    learning_rate=cfg['learning_rate'],
    weight_decay=cfg['weight_decay'],
    batch_size=cfg['batch_size'],
)
