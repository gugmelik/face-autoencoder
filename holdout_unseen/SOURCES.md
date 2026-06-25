# Holdout (unseen-identity) validation images

Two real-person portraits used only for qualitative validation plots — their
reconstructions are logged to wandb each epoch to gauge generalisation to
identities the model never trains on. They are NOT used for training.

Both are **CC0 1.0** (public-domain dedication) from Wikimedia Commons, free to
use and redistribute without attribution; credited here for provenance.

| File | Source | License |
|------|--------|---------|
| `person1_carlin_ross.jpg` | [Wikimedia Commons — Carlin Ross headshot](https://commons.wikimedia.org/wiki/File:Carlin_Ross_headshot.jpg) | CC0 1.0 |
| `person2_adonis_kapsalis.jpg` | [Wikimedia Commons — Adonis Kapsalis Headshot](https://commons.wikimedia.org/wiki/File:Adonis_Kapsalis_Headshot.jpg) | CC0 1.0 |

To use other faces, drop more images here (people not in your training roots);
`load_holdout` in train.py picks up to 8 automatically.
