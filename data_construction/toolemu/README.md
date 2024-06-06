# Sandbox Environment
We construct executable trajectories by building a sandbox environment that uses an LM to emulate the environment observations.

Our codebase is adapted from [ToolEmu](https://github.com/ryoungj/toolemu) but tailored to use privacy-sensitive seeds and vignettes to construct trajectories that create scenarios that LM agents may unintentionally leak information.

If you find this sandbox environment useful, please consider also citing the ToolEmu paper:
```bibtex
@inproceedings{ruan2024toolemu,
  title={Identifying the Risks of LM Agents with an LM-Emulated Sandbox},
  author={Ruan, Yangjun and Dong, Honghua and Wang, Andrew and Pitis, Silviu and Zhou, Yongchao and Ba, Jimmy and Dubois, Yann and Maddison, Chris J and Hashimoto, Tatsunori},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}
```