# What Did You Think Would Happen? Explaining Agent Behaviour Through Intended Outcomes

This repository is the official implementation of "What Did You Think Would Happen? Explaining Agent Behaviour Through Intended Outcomes".

Note that this is a reimplementation of the original codebase used to produce the results presented in the publication. While efforts have been made to produce equivalent results, there might be some discrepencies present.

This implementation is inspired by stable-baselines.

## Reference

Please cite the paper below if you find this paper helpful or if you use this code in your research:

```
@misc{yau2020did,
      title={What Did You Think Would Happen? Explaining Agent Behaviour Through Intended Outcomes}, 
      author={Herman Yau and Chris Russell and Simon Hadfield},
      year={2020},
      eprint={2011.05064},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```

## Requirements

To install requirements, create a new anaconda environment from the provided environment.yml file:

```bash
conda env create -f environment.yml
```

## Training
For more refine control of hyper-parameters, see class __init__ arguments.

```bash
python3 <training-script> --policy <policy-to-use>  --ckpt <checkpoint-path> --seed <random-seed> --load <load-model-if-needed> <--train>
```

## Acknowledgements
This work was partially supported by the UK Engineering and Physical Sciences Research Council (EPSRC), Omidyar Group and The Alan Turing Institute under grant agreements EP/S035761/1 and EP/N510129/1.
