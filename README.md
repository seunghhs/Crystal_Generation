# Crystal Generation with Desired Composition

This code is to generate crystals with desired composition by constraining and exploring the latent space based on  the CDVAE (Crystal Diffusion Variational Autoencoder) model. 



## Prepare before generation

- download cdvae ([https://github.com/txie-93/cdvae](https://github.com/txie-93/cdvae))
- change_params_dir.py
: change the path of the pre-trained model.

```python
python change_params_dir.py
```

- Modify generation/target_atoms.txt
: target composition and target numbers
    - example: ‘target_atoms.txt’
  
    ```python
    {'Cd': 2, 'S': 2}
    ```
  

## Generation

```python
python generation.py --model_path [generation folder]  --seed [seed]
```