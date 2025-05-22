# 修改內容
## 環境建設
- python 3.9會報很多奇怪的錯誤，搞了很久(2~3小時)，不外乎是or, and的邏輯運算子操作問題，但是換到3.7後又神奇的沒有問題了
- 3.9會有一個Error processing row: too many dimensions 'str'的問題
- conda create -n my_env python=3.7
- torch只留torch-geometric>=2.0.0，其他都載不了
## 對chemtils.py的改動
主要兩個改動
```c
ligand_bond_features = torch.stack(ligand_bond_features) if ligand_bond_features else torch.empty((0, 11))
```
```c
ligand_edge_idx = torch.Tensor(ligand_edge_idx).long().T if ligand_edge_idx else torch.empty((2, 0)).long()
```
確保不會stack empty tensor
- Chem.MolFromSmiles回傳空值的情況修改

```c
Error evaluating model: mat1 and mat2 shapes cannot be multiplied (1x153 and 155x512)
```
- 這個問題是elif features == 153:沒有增加atom_eff_charge, atomic_mass，維度少2

還有另外的問題: RuntimeError: mat1 and mat2 shapes cannot be multiplied (6x166 and 164x153)

## 對pretrain_model.py的改動
- 修改total loss是Int無法backward的問題
- 修改total loss原地操作的問題

## 對trainutils_v2的改動
- 修改mse計算時兩個維度不同的問題，已確保相同



