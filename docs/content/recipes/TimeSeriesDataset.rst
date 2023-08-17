TimeseriesDataset & TimeSeriesCollection
========================================

We focus on two abstractions

1. TimeSeriesDataset
2. TimeseriesCollection
   3. non-equimodal
   4. equimodal

We currently only consider the equimodal, single table variant, i.e. both the timeseries and the metadata can be
represented by a single tensor (for example for video data with images+autio+subtitles, they can't)


    +-----+-----------+------------+-------------+
    |     | equimodal | multitable | implemented |
    +=====+===========+============+=============+
    | TSD | -         | y          | n           |
    +-----+-----------+------------+-------------+
    |     | -         | n          | y           |
    +-----+-----------+------------+-------------+
    | TSC | y         | y          | n           |
    +-----+-----------+------------+-------------+
    |     | y         | n          | y           |
    +-----+-----------+------------+-------------+
    |     | n         | y          | n           |
    +-----+-----------+------------+-------------+
    |     | n         | n          | n           |
    +-----+-----------+------------+-------------+


For training models on forecasting tasks, we need suitable DataLoaders.

- torch.DataLoader needs a **sampler**, that produces indices
- torch.Dataset uses the index to produce a **sample**

So in the Pipeline we need

- Generic Dataset class that takes care of setting up the dataset such that `dataset[key] = subset`
    - For a TSC, we use index i∈𝓘 as input and return a TSD. Globals are ingored (they are only used for model setup)
    - For a TSD, we use index t∈𝓣 as input and return time series data (t, vₜ), ignoring metadata.
    - So in both cases selecting with an index only returns data that depends on that index, and not the static data that
      is independent of the index.
    - given a key `(i, t)` this should look up the data `(S[i][t], M[i])` and return `(S[i][t], M[i])`
- A specialized task-specific dataset class such that `dataset[key] = sample`
    - It used the returned data `(S[i][t], M[i])` and performs the necessary **slicing and masking** to produce a sample
      of the form `(t_input, x_input, t_target, y_target, md_input, md_target)`
    - **Open Design Question:** Should covariates be given as separate inputs `(t_x, x, t_y, y, t_u, u, md_input, md_target)`?
        - Advantage: clear separation
        - Distadvantage: model needs to merge that data itself
- A collate function such that `collate(samples: list[sample]) -> batch`
- `sample` should be a `NamedTuple` of `DataFrames`
- `batch` should be a `NamedTuple` of `Tensors`
