# I-RAVEN Dataset

The original dataset is published in

[Stratified Rule-Aware Network for Abstract Visual Reasoning](https://arxiv.org/abs/2002.06838)  
Sheng Hu\*, Yuqing Ma\*, Xianglong Liu†, Yanlu Wei, Shihao Bai  
*Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)*, 2021  
(\* equal contribution, † corresponding author)

Code is based on the forked repo [husheng12345/SRAN](https://github.com/husheng12345/SRAN)

## Remarks by cwhy

I really like this dataset.

I isolated the I-Raven dataset for convinience of testing.

I also updated the code to support python3 (tested on python3.7/3.9).

There are [mach-nix](https://github.com/DavHau/mach-nix) configurations in `env`

## Run instruction

Install via `pip` or `nix`

Run example
```bash
python main.py --num-samples 10 --save-dir dataset
```
to generated the dataset.

Modify `dataset/view_data_point.py` and run to view any data points generated
