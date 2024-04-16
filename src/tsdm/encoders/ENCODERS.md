# Technical Note

The encoders-library shares a lot of similarity with sklearn's `preprocessing` and `pipeline` module. The main difference is that the encoders-library is designed to work with `pandas` DataFrames and Series, while sklearn's `preprocessing` module is designed to work with numpy arrays.

There are a few differences:

- sklearn always carries through the `y`-values, even for transforms that are only applied to the `X`-values. This is not the case for the encoders-library.

## Goal

- compatiblity with `sklearn.preprocessing` and `sklearn.pipeline`

In order to train, evaluate, and perform inference on models, we need to be able to create end-to-end pipelines.
Note that for supervised learning, we need to be able to encode the features, and both en- and decode the targets.
Being able to decode the targets is critical in order to ensure that multiple models can be compared on the same metric,
even if they use different encoding schemes. Fundamentally, the encoding scheme is a part of the model, and should be treated as such.
Of course, there are also cases such as time-series forecasting, where the targets and features are the same variable, but at different points in time.

On the other hand, sometimes we might be interested in having target-aware encoders, for example[^1]. In the most
general case, any encoder, as it is part of the model, can make use of any information contained in the training data.

One specific consideration are non-decomposable batch encoders. Usually, elements in a batch are encoded
element-wise, however there are exceptions such as batch-normalization, for which the output depends on the entire batch.
These transformations are weird, since they will make the model predict differently depending on which other
elements happen to be present in the batch. This is not a problem for training, but it is awkard in inference, and
problematic in evaluation.

### Pipeline Examples

#### Training pipeline

1. Load samples sₙ=(xₙ,yₙ) from source
2. encode data $x' = \text{feature-encoder}(x)$, $y' = \text{target-encoder}(y)$
3. make model predictions $\hat{y}' = \text{model}(x')$
4. update model parameters based on (encoded domain) loss $ℓ(y', \hat{y}')$

#### Evaluation pipeline

1. Load data (x,y) from source
2. encode data $x' = \text{feature-encoder}(x)$
3. make model predictions $\hat{y}' = \text{model}(x')$
4. decode predictions $\hat{y} = \text{target-decoder}(\hat{y}')$
5. evaluate model predictions based on (real world domain) loss $ℓ(y, \hat{y})$

#### Inference pipeline

1. Load data (x) from source
2. encode data $x' = \text{feature-encoder}(x)$
3. make model predictions $\hat{y}' = \text{model}(x')$
4. decode predictions $\hat{y} = \text{target-decoder}(\hat{y}')$

--

[^1]: On Embeddings for Numerical Features in Tabular Deep Learning: <https://proceedings.neurips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html>
