# Mask User Tokens

Repository to reproduce results my blog [Mask Your User Tokens](https://yonigottesman.github.io/2024/05/13/mask-user-tokens.html).


Finetune gemma2b on Universal-NER with and without masking user tokens.

```bash
python train.py gemma2b_config.yaml --mask-user-tokens
```

```bash
python train.py gemma2b_config.yaml --no-mask-user-tokens
```


