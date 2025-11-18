import argparse

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


def read_database(args):
    database_path = f"{args.data_path}/monomer.xlsx"
    df_A = pd.read_excel(database_path, sheet_name="A")
    df_B = pd.read_excel(database_path, sheet_name="B")
    df_C = pd.read_excel(database_path, sheet_name="C")
    df_D = pd.read_excel(database_path, sheet_name="D")
    df_r = pd.read_excel(database_path, sheet_name="ratio")

    return df_A, df_B, df_C, df_D, df_r


def react(amine_smi, db_smi):
    rxn_1 = AllChem.ReactionFromSmarts('[NH2:1][C;!$(C=O):6].[C:2]=[C:3][C:4]=[O:5]>>[*]-[N:1]([C;!$(C=O):6])-[C:2][C:3][C:4]=[O:5]')
    rxn_2 = AllChem.ReactionFromSmarts('[C:1]=[C:2][C:3]=[O:4]>>[*]-[C:1][C:2][C:3]=[O:4]')
    reactants = (Chem.MolFromSmiles(amine_smi), Chem.MolFromSmiles(db_smi))
    products = rxn_1.RunReactants(reactants)
    products = rxn_2.RunReactants((products[0][0],))
    return Chem.MolToSmiles(products[0][0])


def retrieve_idx(args, df_raw):
    df_database = read_database(args)
    for col, df in zip(df_raw.columns[1:5], df_database[:4]):
        map_dict = df.set_index('idx')["smiles"].to_dict()
        df_raw[col] = df_raw[col].map(map_dict)
    df_raw_merge = df_raw.merge(df_database[-1], left_on="r", right_on="idx", how="left")
    df_raw_merge.drop(['idx', 'r'], axis=1, inplace=True)

    if args.label is True:
        df_raw_merge = df_raw_merge.reindex(columns=['ID', 'A', 'B', 'C', 'D', 'rB', 'rC', 'rD', 'value1', 'value2'])
    else:
        df_raw_merge = df_raw_merge.reindex(columns=['ID', 'A', 'B', 'C', 'D', 'rB', 'rC', 'rD'])
    return df_raw_merge


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--data_path", type=str, default='../datasets/')
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--label", action="store_true", help="Include label columns")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    df_raw = pd.read_csv(f"{args.data_path}/{args.dataset}/{args.dataset}_raw.csv")
    df = retrieve_idx(args, df_raw)
    df["B"] = df.apply(lambda x: react(x["B"], x["A"]), axis=1)
    df["C"] = df.apply(lambda x: react(x["C"], x["A"]), axis=1)
    df["D"] = df.apply(lambda x: react(x["D"], x["A"]), axis=1)
    df["rB"] = df["rB"] / 10
    df["rC"] = df["rC"] / 10
    df["rD"] = df["rD"] / 10
    df.drop(['A'], axis=1, inplace=True)
    df.to_csv(f"{args.data_path}/{args.dataset}/{args.dataset}.csv", index=False)
