# Aegis Model Builder
Quickly create simple Keras models such as autoencoders and stacks of dense layers, etc.

Uses [Google python-fire](https://github.com/google/python-fire)

## Usage examples
Build an autoencoder model, display summary, save the model, then quit:
```
docker-compose run --rm builder dense_ae [1280] --latent-size 32 - compile - summary - save "models/ae.h5" - end
```

## TODO
* CLI without "end" command (for some reason, it assumes that more commands should follow w/o end)
* VAE
* document more usage
* LSTM
