Vanilla Training Loop
---------------------

Typical experimental flow, given a fixed hyperparameter combination


.. code-block:: python

    from tsdm.datasets import DATASETS
    from tsdm.encoders import ENCODERS
    from tsdm.models   import MODELS
    from tsdm.losses import LOSSES
    from tsdm. optimizers import OPTIMIZERS

    # current HP configuration
    HP = json.read("ID-XXXXXXX.json")

    dataset_cls = DATASETS[HP['dataset']]
    dataset     = dataset(HP['dataset_cfg'])

    model_cls = MODELS[HP['model']]
    model     = model_cls(HP['model_cfg'] + dataset.info)

    encoder_cls = ENCODERS[HP['encoder']]
    encoder     = encoder_cls(HP['encoder_cfg'] + dataset.info + model.info)

    optimizer_cls = OPTIMIZERS[HP['optimizer']]
    optimizer     = optimizer_cls[HP['optimizer_cfg']]

    loss_cls = LOSSES(HP['loss'])
    loss     = loss_cls(HP['loss_cfg'])

    dataloader_cls = HP['dataloader']
    dataloader     = dataloader_cls(HP['dataloader_cfg'])

    optimizer_cls = HP['optimizer']
    optimizer     = HP['optimizer_cfg']

    trainer_cls   = HP['trainer']
    trainer       = HP['trainer_cfg']

    metrics = HP['metrics']

    logger = default_logger
    logger.register(model=model, optimizer=optimizer, loss=loss, metrics=metrics)

    model.init_params()  # random initialization of model

    for batch in dataloader:
        x, y = batch
        x  = encoder.encode(x)
        yhat = model(x)
        yhat = encoder.decode(yhat)
        r = loss(y, yhat)
        r.backward()
        optimizer.step()

        logger.log(loss, model, optimizer, metrics, ....)

        if trainer.stopping_criteria(model, optimizer, dataloader, logger.history):
            break
    else:
        warning(F"No convergence in {dataloader.epochs} epochs!!")

    return results
