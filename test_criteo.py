import pandas as pd
import tensorflow as tf
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from census_config import *
import os


file_path = '/media/psdz/hdd/Download/Criteo/'
train_path = file_path + 'train.txt'

numeric_feat     = ['I' + str(i) for i in range(1, 14)]
categorical_feat = ['C' + str(i) for i in range(1, 27)]
total_feat       = numeric_feat + categorical_feat
csv_header       = ['label'] + numeric_feat + categorical_feat

data = pd.read_csv(train_path, sep='\t', names=csv_header)
sample_data = data.sample(frac=0.01, random_state=2022)

train = sample_data[: int(len(sample_data) * 0.8)].reset_index(drop=True)
vali = sample_data[int(len(sample_data) * 0.8): ].reset_index(drop=True)

# fill missing values
for col in numeric_feat:
    avg = train[col].mean()
    train[col] = train[col].fillna(avg)
    train[col] = train[col].apply(lambda x: round(x, 6))
    vali[col] = vali[col].fillna(avg)
    vali[col] = vali[col].apply(lambda x: round(x, 6))

for col in categorical_feat:
    train[col] = train[col].fillna('na')
    vali[col] = vali[col].fillna('na')

train.to_csv(file_path + 'train.csv', index=False)
vali.to_csv(file_path + 'vali.csv', index=False)


def build_categorical_vocab(df):
    cate_feat_vocab, cate_feat_vocab_size = {}, {}
    for col in categorical_feat:
        cate_feat_vocab[col]      = sorted(list(df[col].unique())),
        cate_feat_vocab_size[col] = len(sorted(list(df[col].unique()))),
    return cate_feat_vocab, cate_feat_vocab_size


cate_feat_vocab, cate_feat_vocab_size = build_categorical_vocab(train)

cfg = {
    # feat config
    'numeric_feat'        : numeric_feat,
    'categorical_feat'    : categorical_feat,
    'total_feat'          : total_feat,
    'csv_header'          : csv_header,
    'cate_feat_vocab'     : cate_feat_vocab,
    'cate_feat_vocab_size': cate_feat_vocab_size,
    'target_col'          : 'label',

    # model config
    'embedding_dims'            : 16,
    'num_heads'                 : 4,
    'mlp_hidden_units_factors'  : [2, 1],
    'num_mlp_blocks'            : 2,

    # training process config
    'learning_rate'   : 0.001,
    'weight_decay'    : 0.0001,
    'dropout_rate'    : 0.2,
    'learning_rate'   : 0.001,
    'batch_size'      : 256,
    'num_epoch'       : 15,
}


# target_label_lookup = layers.StringLookup(
#     vocabulary=TARGET_LABELS,
#     mask_token=None,
#     num_oov_indices=0
# )


def get_tf_dataset_from_csv(csv_file_path, cfg, batch_size=128, shuffle=False):
    def process(features):
        feedids_string = features['feedids']
        seq_feedids = tf.strings.split(feedids_string, '|').to_tensor()
        features['feedids'] = seq_feedids[:, :]
        labels = features[label_name]
        return features, labels

    def prepare_example(features, target):
        target_index = target_label_lookup(target)
        return features, target_index

    def prepare_example_update(features, target):
        target = features['label']
        features = features.pop(['label'])
        return features, target_index

    dataset = tf.data.experimental.make_csv_dataset(
        csv_file_path,
        batch_size=batch_size,
        column_names=cfg['csv_header'],
        label_name=cfg['target_col'],
        num_epochs=1,
        # header=False,
        header=True,
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

def encode_inputs(inputs, embedding_dims, cfg):
    encoded_categorical_feature_list, numerical_feature_list = [], []
    for feat in inputs:
        if feat in cfg['categorical_feat']:
            vocabulary = cfg['cate_feat_vocab'][feat]
            lookup = layers.StringLookup(vocabulary=vocabulary, mask_token=None, num_oov_indices=0, output_mode="int")
            encoded_feature = lookup(inputs[feat])
            embedding = layers.Embedding(input_dim=len(vocabulary), output_dim=embedding_dims)
            encoded_categorical_feature = embedding(encoded_feature)
            encoded_categorical_feature_list.append(encoded_categorical_feature)

        if feat in cfg['numerical_feat']:
            numerical_feature = tf.expand_dims(inputs[feat], -1)
            numerical_feature_list.append(numerical_feature)

    return encoded_categorical_feature_list, numerical_feature_list


def create_mlp(hidden_units, dropout_rate, activation, normalization_layer, name=None):
    mlp_layers = []
    for units in hidden_units:
        mlp_layers.append(normalization_layer),
        mlp_layers.append(layers.Dense(units, activation=activation))
        mlp_layers.append(layers.Dropout(dropout_rate))

    return keras.Sequential(mlp_layers, name=name)


inputs = create_model_inputs(cfg)
encoded_categorical_feature_list, numerical_feature_list = encode_inputs(inputs, cfg['embedding_dims'])

features = layers.concatenate(encoded_categorical_feature_list + numerical_feature_list)
feedforward_units = [features.shape[-1]]

for layer_idx in range(cfg['num_mlp_blocks']):
    features = create_mlp(
        hidden_units=feedforward_units,
        dropout_rate=cfg['dropout_rate'],
        activation=keras.activations.gelu,
        normalization_layer=layers.LayerNormalization(epsilon=1e-6),
        name=f"feedforward_{layer_idx}",
    )(features)

mlp_hidden_units = [factor * features.shape[-1] for factor in mlp_hidden_units_factors]

features = create_mlp(
    hidden_units=mlp_hidden_units,
    dropout_rate=dropout_rate,
    activation=keras.activations.selu,
    normalization_layer=layers.BatchNormalization(),
    name="MLP",
)(features)

outputs = layers.Dense(units=1, activation="sigmoid", name="sigmoid")(features)
model = keras.Model(inputs=inputs, outputs=outputs)


def run_experiment(model, train_data_file, test_data_file, num_epochs, learning_rate, weight_decay, batch_size,):
    optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
    )

    train_dataset = get_tf_dataset_from_csv(train_data_file, batch_size, shuffle=True)
    validation_dataset = get_tf_dataset_from_csv(test_data_file, batch_size)

    history = model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=validation_dataset)
    _, accuracy = model.evaluate(validation_dataset, verbose=-1)

    print(f"Final Vali Acc: {round(accuracy * 99, 2)}%")
    return history

