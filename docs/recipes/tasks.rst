datasets
========


Mental Model
------------

dataset: raw data

create dataloader



Train-variant
1. raw data
2. task specific preprocessing ``data = preprocess(data)``
3. model specific encoding  ``data = encode(data)``
4. sampling of batch  ``indices: list[int] = sample()da``
5. Collating batch ``batch = collate(data[indices])``
6. Seperate batch ``inputs, targets = batch``
5. feeding batch to model ``outputs=model(*inputs)``
6. comparing model outputs to targets ``loss(targets, outputs)``

Inference
1. raw data
2. task specific preprocessing ``data = preprocess(data)``
3. model specific encoding  ``data = encode(data)``
4. sampling of batch  ``indices: list[int] = sample()da``
5. Collating batch ``batch = collate(data[indices])``
6. Seperate batch ``inputs, targets = batch``
5. feeding batch to model ``outputs=model(*inputs)``
5b. Transforming results: ``prediction = make_preds(outputs)``
6. comparing model outputs to targets ``metric(targets, predictions)``

There are multiple ways of doing things:

1. Model has both a `forward` and a `predict`
    - ⊖: little code sharing, mapping to predictions is often the same process
    - ⊖: How to incorporate model/task specific encoding/preprocessing?

2. Sampler performs splitting
    - ⊖: Makes sampler logic more complicated
    - ⊕: reduces chance of accidentally showing the model training data.
    - ⊖: sampler comes after encoder and is ignorant of what encoder does,
         how could it deal with all the possibilities?
    - Real solution would only be to encode **after** sampling
        - this makes a lot of things easier
        - But problematic optimization wise, because we would like to cache the whole dataset!
        - Distinction: regular encoder, irregular encoder.
        - regular encoder: acts instance wise, returns same shape number of items
        - But this type of encoding cannot be cached!

3. Use multiple encoders
    - pre-encoder: Applied before batching
    - post-encoder: Applied after batching
    - ⊖: Kinda complicates things a bit.
    - ⊖: Kinda difficult to explain why we need it.
    - ⊖: What we need anyway to cover all cases?
    - ⊕⊕: Covers all cases!
    - ⊕⊕: Easy to deal with!


.. code-block:: python

    data = preprocess(data)
    data = preencode(data)
    dataloader = DataLoader(data, sampler=sampler, collate_fn=collate_fn)

    for batch in dataloader:
        inputs, targets = batch
        outputs = model(*inputs)
        prediction = decode(outputs)
        result = loss(targets, outputs)







.. code-block:: python

    @jit.script
    def prep_batch(batch: tuple[Tensor, Tensor, Tensor], observation_horizon: int):
        T, X, Y = batch
        targets = Y[..., observation_horizon:].clone()
        Y[..., observation_horizon:] = float("nan")  # mask future
        X[..., observation_horizon:, :] = float("nan")  # mask future
        inputs = torch.cat([X, Y.unsqueeze(-1)], dim=-1)
        return T, inputs, targets


    def get_all_preds(model, dataloader):
        Y, Ŷ = [], []
        for batch in tqdm(dataloader, leave=False):
            with torch.no_grad():
                model.zero_grad()
                times, inputs, targets = prep_batch(batch, OBS_HORIZON)
                outputs, _ = model(times, inputs)
                predics = outputs[:, OBS_HORIZON:, -1]
                loss = LOSS(predics, targets)
                Y.append(targets)
                Ŷ.append(predics)

        return torch.cat(Y, dim=0), torch.cat(Ŷ, dim=0)
