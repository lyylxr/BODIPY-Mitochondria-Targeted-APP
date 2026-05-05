# app.py
import os, tempfile, shutil, pathlib
import pandas as pd
import joblib
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
import feature_rdkit as ft

# -------------------- 封装三段脚本 --------------------
def step1_smiles_to_features(smiles: str) -> pathlib.Path:
    """
    输入：单个 SMILES
    输出：step1 生成的 Excel 路径（newSMILES_features.xlsx）
    逻辑完全等价你第一段代码，只是用临时文件
    """
    # 临时目录
    tmpdir = pathlib.Path(tempfile.mkdtemp())
    new_smiles_xlsx   = tmpdir / "newSMILES.xlsx"
    dict_file         = "Data-target.xlsx"          # 原始字典文件
    output_folder     = tmpdir / "svg_tmp"          # 临时 svg 目录
    step1_out         = tmpdir / "newSMILES_features.xlsx"

    # 1. 建字典
    df_dict = pd.read_excel(dict_file, sheet_name="Sheet1")
    feature_idx_map = ft.create_unified_fingerprint_dict(df_dict["SMILES"], str(output_folder))
    if output_folder.exists():
        shutil.rmtree(output_folder)

    # 2. 构造仅有一行的“newSMILES.xlsx”
    df_new = pd.DataFrame([{"Number": 1, "SMILES": smiles}])
    df_new.to_excel(new_smiles_xlsx, index=False)

    # 3. 给这一行生成特征
    df_new = pd.read_excel(new_smiles_xlsx, sheet_name="Sheet1")
    fingerprints, descriptors_list = [], []
    for smi in df_new["SMILES"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fingerprints.append([0] * len(feature_idx_map))
            descriptors_list.append({k: None for k in ft.extract_descriptors(mol).keys()})
            continue
        fp = ft.generate_fingerprint(smi, feature_idx_map)
        fingerprints.append(fp if fp is not None else [0] * len(feature_idx_map))
        descriptors_list.append(ft.extract_descriptors(mol))

    # 4. 拼接
    df_new = df_new[["Number", "SMILES"]]
    feature_columns = [f"Feature_{i}" for i in range(len(feature_idx_map))]
    fingerprint_df = pd.DataFrame(fingerprints, columns=feature_columns)
    descriptors_df = pd.DataFrame(descriptors_list)
    df_final = pd.concat([df_new, descriptors_df, fingerprint_df], axis=1)
    df_final.to_excel(step1_out, index=False)
    return step1_out


def step2_align_columns(step1_file: pathlib.Path) -> pathlib.Path:
    """
    等价第二段代码：用 compound_features_choose.xlsx 对齐列
    返回对齐后的 Excel 路径
    """
    tmpdir = step1_file.parent
    aligned_file = tmpdir / "newSMILES_features_choose.xlsx"

    df_A = pd.read_excel("compound_features_choose.xlsx")
    df_B = pd.read_excel(step1_file)

    columns_to_keep = df_A.columns[2:]
    columns_to_drop = [c for c in df_B.columns[2:] if c not in columns_to_keep]
    df_B = df_B.drop(columns=columns_to_drop)
    df_B.to_excel(aligned_file, index=False)
    return aligned_file


def step3_predict(aligned_file: pathlib.Path) -> pd.DataFrame:
    """
    等价第三段代码：加载模型+scaler，返回带预测结果的新 DataFrame
    """
    gbm = joblib.load("LightGBM_model.pkl")
    scaler = joblib.load("LightGBM_scaler.pkl")

    df = pd.read_excel(aligned_file)
    X = df.iloc[:, 2:]
    X_scaled = scaler.transform(X)
    proba = gbm.predict(X_scaled)
    label = (proba > 0.5).astype(int)

    df["Predicted_Label"] = label
    df["Predicted_Probability"] = proba
    return df


# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Mitochondrial Targeting Predictor", layout="centered")

# 美化标题布局 - 两行左对齐
st.markdown(
    """
    <div style="display: flex; align-items: flex-start; margin-bottom: 1rem;">
        <span style="font-size: 2.5rem; margin-right: 15px;">🎯</span>
        <div>
            <div style="font-size: 1.8rem; font-weight: 700; line-height: 1.1;">
                Organic small molecule
            </div>
            <div style="font-size: 1.8rem; font-weight: 700; line-height: 1.1;">
                mitochondrial targeting predictor
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("Input SMILES (limited to BODIPY derivatives)")

smiles = st.text_input("SMILES：", placeholder="eg:CCOc1ccccc1")

if st.button("Forecast"):
    if not smiles.strip():
        st.warning("Please enter a valid SMILES!")
        st.stop()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("RDkit is unable to parse this SMILES. Please check!")
        st.stop()

    with st.spinner("Feature extraction is underway → Alignment → Prediction..."):
        # 三步曲
        step1_path = step1_smiles_to_features(smiles)
        step2_path = step2_align_columns(step1_path)
        result_df  = step3_predict(step2_path)

    # 取结果
    label = result_df["Predicted_Label"].iloc[0]
    proba = result_df["Predicted_Probability"].iloc[0]

    col1, col2 = st.columns(2)
    col1.metric("Prediction label (Is it targeting mitochondria?)", "Yes" if label else "No")
    col2.metric("Target probability", f"{proba:.3f}")

    # 分子图
    img = Draw.MolToImage(mol, size=(350, 350))
    st.image(img, caption="Molecular structure")

    # 下载
    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download the prediction results", data=csv, file_name="pred_result.csv", mime="text/csv")

    # 可选：清理临时目录
    shutil.rmtree(step1_path.parent)
