# Technical Note

The encoders-library shares a lot of similarity with sklearn's `preprocessing` and `pipeline` module. The main difference is that the encoders-library is designed to work with `pandas` DataFrames and Series, while sklearn's `preprocessing` module is designed to work with numpy arrays.

There are a few differences:

- sklearn always carries through the `y`-values, even for transforms that are only applied to the `X`-values. This is not the case for the encoders-library.

## Goal

- compatiblity with `sklearn.preprocessing` and `sklearn.pipeline`

In order to train, evaluate, and perform inference on models, we need to be able to create end-to-end pipelines. Note that for supervised learning, we need to be able to encode the features, and both en- and decode the targets. Being able to decode the targets is critical in order to ensure that multiple models can be compared on the same metric, even if they use different encoding schemes. Fundamentally, the encoding scheme is a part of the model, and should be treated as such. Of course, there are also cases such as time-series forecasting, where the targets and features are the same variable, but at different points in time.

On the other hand, sometimes we might be interested in having target-aware encoders, for example[^1]. In the most general case, any encoder, as it is part of the model, can make use of any information contained in the training data.

One specific consideration are non-decomposable batch encoders. Usually, elements in a batch are encoded element-wise, however there are exceptions such as batch-normalization, for which the output depends on the entire batch. These transformations are weird, since they will make the model predict differently depending on which other elements happen to be present in the batch. This is not a problem for training, but it is awkward in inference, and problematic in evaluation.

Another complication is that, for instance, for time series forecasting, we need to both transform variables, and also perform selection operations. Typically, we have a segment of the time series $S=(tₙ, xₙ, yₙ)_{n=-L:F}$,  and the goal is to predict $y_{1:F}$ based on $x_{-L:0}$. So, the sample in this case is input=$(t_{1:F}，(tₙ, xₙ)_{n=-L:0})$ and the target is $y_{1:F}$.

The question here is: what comes first: encoding or selection? The selection is model-independent, while the encoding is model-dependent. Moreover, encoding can potentially generate additional features, or be batch-dependent as discussed above. On the other hand, fitting and applying encoders post-selection is a bit awkward, as initially, the table is a simple tabular form, but later, it is segmented into different components.

## What Encoders should and shouldn't do

Generally, there are 4 things we need to do inside the pipeline:

1. Select data and create samples
2. Encode data
3. Batch data
4. Decode the model predictions

Remarks:

- Encoding could potentially be performed on the entire dataset, which would provide efficiency gains.
  However, this can make the sample creation more difficult, because encoding can create additional features.
  Also, this is only really useful in academic settings, not in production where data samples are created on-the-fly.
- Encoders should use canonical vectorization to act on a batch of samples.

## Why not just `sklearn`?

Scikit-learn is a great library, but there are several limitations that lead us to develop a (partially compatible) alternative:

1. No support for complex variables. (What about a FFT encoder?!)
2. Restricted to numpy (what if we need to encode a `torch.Tensor`?)
3. No explicit duck-typing. Sklearn claims to use duck-typing, but it does not provide any actual `typing.Protocol` definitions.
4. Awkward design choices: For instance, always providing separate X and y values, even for unsupervised tasks.

## Question

- Should batching happen before encoding, or after encoding?

### Pipeline Examples

#### Training pipeline

1. Initialize model, encoder, task, etc.
2. Fit encoder on dataset
3. While not converged:
   1. Load samples sₙ=(xₙ,yₙ) from source
   2. encode+batch data $x' = \text{feature-encoder}(x)$, $y' = \text{target-encoder}(y)$
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
