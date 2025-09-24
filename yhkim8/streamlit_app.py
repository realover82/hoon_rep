import streamlit as st
import os
import re
import math
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as tvm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import io
import pandas as pd

# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(layout="wide")
st.title("다이얼 게이지 자동 분석 애플리케이션 📊")

# 세션 상태 초기화
if 'analysis_results' not in st.session_state:
    st.session_state['analysis_results'] = None
if 'model_summary' not in st.session_state:
    st.session_state['model_summary'] = None
if 'finetune_preview' not in st.session_state:
    st.session_state['finetune_preview'] = None
if 'metrics_calculated' not in st.session_state:
    st.session_state['metrics_calculated'] = False
if 'model1' not in st.session_state:
    st.session_state['model1'] = None
if 'model2' not in st.session_state:
    st.session_state['model2'] = None

# 임시 파일 업로드 디렉토리
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)
UPLOAD_DIR_TEST = os.path.join(UPLOAD_DIR, "test")
os.makedirs(UPLOAD_DIR_TEST, exist_ok=True)

# 모델 및 데이터 저장 디렉토리
DATA_DIR = "gage_data"
os.makedirs(DATA_DIR, exist_ok=True)
CKPT_DIR_ZERO = os.path.join(DATA_DIR, "checkpoints_zerohead_A")
os.makedirs(CKPT_DIR_ZERO, exist_ok=True)

# TPU 가속은 웹 환경에서 사용 불가. CPU/GPU로 대체
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.info(f"사용 중인 장치: {device}")

# =========================================================
# 모델 및 유틸리티 함수 정의
# =========================================================

# tfm 변수를 전역으로 정의
tfm = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Lambda(lambda x: x.expand(3, -1, -1)),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class AngleHead(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = tvm.resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)

    def forward(self, x):
        y = self.backbone(x)
        return y / (y.norm(dim=1, keepdim=True) + 1e-8)

def parse_mm_prefix(fp):
    name = os.path.basename(fp)
    m = re.match(r"^\D*?(\d{1,2})", name)
    if not m:
        return None
    v = int(m.group(1))
    if 0 <= v <= 99:
        return v / 100.0
    return None

TWO_PI = 2.0 * math.pi
def wrap_angle(x):
    return (x + TWO_PI) % TWO_PI
    
# =========================================================
# Streamlit UI
# =========================================================
st.sidebar.header("⚙️ 모델 설정")

model_choice = st.sidebar.selectbox("모델 선택", ("ResNet-18 (AngleHead)", "YOLOv5 (예정)"))
if model_choice == "YOLOv5 (예정)":
    st.sidebar.warning("YOLOv5는 현재 더미 모델로 구현되어 있으며, 실제 기능은 없습니다.")

st.sidebar.text("")

# =========================================================
# 파일 업로드 섹션
# =========================================================
st.header("📂 데이터 업로드")
st.markdown("성능 테스트에 사용할 다이얼 게이지 이미지를 업로드하세요. 파일명은 `0-sample.png` 또는 `00-sample.png` 형식으로 `mm` 값이 포함되어야 합니다.")

uploaded_test_files = st.file_uploader("테스트용 이미지 업로드", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="test_uploader")
if uploaded_test_files:
    for file in glob.glob(os.path.join(UPLOAD_DIR_TEST, "*")):
        os.remove(file)
    for uploaded_file in uploaded_test_files:
        file_path = os.path.join(UPLOAD_DIR_TEST, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.success(f"{len(uploaded_test_files)}개의 테스트용 이미지 업로드 완료.")

st.subheader("모델 1 업로드")
uploaded_model1_pth = st.file_uploader("Model 1 .pth 파일 업로드", type=["pth"], key="model1_uploader")

st.subheader("모델 2 업로드")
uploaded_model2_pth = st.file_uploader("Model 2 .pth 파일 업로드", type=["pth"], key="model2_uploader")
    
# =========================================================
# 기능 버튼
# =========================================================
st.header("🚀 성능 테스트 실행")

if st.button("두 모델 분석 및 비교 시작"):
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    test_files = []
    for ext in image_extensions:
        test_files.extend(glob.glob(os.path.join(UPLOAD_DIR_TEST, ext)))

    if not test_files:
        st.warning("분석을 시작하려면 테스트용 이미지를 먼저 업로드해주세요.")
        st.stop()

    if not uploaded_model1_pth or not uploaded_model2_pth:
        st.warning("분석을 시작하려면 두 모델의 .pth 파일을 모두 업로드해야 합니다.")
        st.stop()

    st.info("두 모델을 로딩 중...")
    try:
        model1 = AngleHead(pretrained=False).to(device)
        model2 = AngleHead(pretrained=False).to(device)
        
        state_dict_m1 = torch.load(io.BytesIO(uploaded_model1_pth.read()), map_location=device)
        model1.load_state_dict(state_dict_m1, strict=False)
        model1.eval()

        state_dict_m2 = torch.load(io.BytesIO(uploaded_model2_pth.read()), map_location=device)
        model2.load_state_dict(state_dict_m2, strict=False)
        model2.eval()
        
        st.success("두 모델 로딩 완료!")

    except Exception as e:
        st.error(f"모델 로딩 중 오류 발생: {e}")
        st.stop()
    
    st.header("📊 개별 파일 분석 결과")
    results = []

    for fp in test_files:
        mm_from_name = parse_mm_prefix(fp)
        if mm_from_name is None:
            st.warning(f"파일명에서 mm 값을 파싱할 수 없습니다: {os.path.basename(fp)}. 이 파일은 분석에서 제외됩니다.")
            continue

        with torch.no_grad():
            x = tfm(Image.open(fp).convert("L")).unsqueeze(0).to(device)
            
            y1 = model1(x)[0].cpu().numpy()
            psi_pred1_rad = math.atan2(float(-y1[0]), float(y1[1]))
            psi_pred1_wrap = wrap_angle(psi_pred1_rad)
            predicted_mm_model1 = (psi_pred1_wrap / TWO_PI) * 100

            y2 = model2(x)[0].cpu().numpy()
            psi_pred2_rad = math.atan2(float(-y2[0]), float(y2[1]))
            psi_pred2_wrap = wrap_angle(psi_pred2_rad)
            predicted_mm_model2 = (psi_pred2_wrap / TWO_PI) * 100

        # 수학적 풀이 과정 출력
        st.text(f"--- 파일: {os.path.basename(fp)} ---")
        st.text(f"실제 값: {mm_from_name * 100:.2f} mm")
        st.text("")
        st.text("--- 모델 1 수학적 풀이 ---")
        st.text(f"모델 출력 (x, y): ({y1[1]:.4f}, {y1[0]:.4f})")
        st.text(f"math.atan2(-y, x) 적용: math.atan2({-y1[0]:.4f}, {y1[1]:.4f}) = {psi_pred1_rad:.4f} rad")
        st.text(f"wrap_angle() 적용: ({psi_pred1_rad:.4f} + {TWO_PI:.4f}) % {TWO_PI:.4f} = {psi_pred1_wrap:.4f} rad")
        st.text(f"mm 값으로 변환: ({psi_pred1_wrap:.4f} / {TWO_PI:.4f}) * 100 = {predicted_mm_model1:.2f} mm")
        st.text("")
        st.text("--- 모델 2 수학적 풀이 ---")
        st.text(f"모델 출력 (x, y): ({y2[1]:.4f}, {y2[0]:.4f})")
        st.text(f"math.atan2(-y, x) 적용: math.atan2({-y2[0]:.4f}, {y2[1]:.4f}) = {psi_pred2_rad:.4f} rad")
        st.text(f"wrap_angle() 적용: ({psi_pred2_rad:.4f} + {TWO_PI:.4f}) % {TWO_PI:.4f} = {psi_pred2_wrap:.4f} rad")
        st.text(f"mm 값으로 변환: ({psi_pred2_wrap:.4f} / {TWO_PI:.4f}) * 100 = {predicted_mm_model2:.2f} mm")
        st.text("-----------------------------------")
        
        results.append({
            "filepath": os.path.basename(fp),
            "true_mm": mm_from_name,
            "predicted_mm_model1": predicted_mm_model1,
            "predicted_mm_model2": predicted_mm_model2,
        })
        
    st.session_state['analysis_results'] = results
    st.success("분석 완료!")

if st.session_state['analysis_results']:
    st.header(f"📋 분석 결과")
    results = st.session_state['analysis_results']
    results_df = pd.DataFrame(results)

    # DataFrame 표시
    st.subheader("개별 파일 예측 결과")
    st.dataframe(results_df.style.format({
        'true_mm': '{:.2f}',
        'predicted_mm_model1': '{:.2f}',
        'predicted_mm_model2': '{:.2f}'
    }))

    # 모델 1 성능 지표 계산 및 시각화
    st.subheader("모델 1 성능 지표")
    y_true_binary = [1 if r['true_mm'] > 0.5 else 0 for r in results]
    y_pred_binary_m1 = [1 if r['predicted_mm_model1'] > 50 else 0 for r in results]
    
    cm_m1 = confusion_matrix(y_true_binary, y_pred_binary_m1)
    if cm_m1.shape == (2, 2):
        tn_m1, fp_m1, fn_m1, tp_m1 = cm_m1.ravel()
    else:
        tn_m1, fp_m1, fn_m1, tp_m1 = 0, 0, 0, 0
    
    st.markdown("### 혼동 행렬 (모델 1)")
    st.markdown("| | 예측 음성 (Negative) | 예측 양성 (Positive) |")
    st.markdown("|---|---|---|")
    st.markdown(f"| **실제 음성 (Negative)** | {tn_m1} (TN) | {fp_m1} (FP) |")
    st.markdown(f"| **실제 양성 (Positive)** | {fn_m1} (FN) | {tp_m1} (TP) |")
    
    fig_cm_m1, ax_cm_m1 = plt.subplots()
    sns.heatmap(cm_m1, annot=True, fmt='d', cmap='Blues', ax=ax_cm_m1)
    ax_cm_m1.set_xlabel('Predicted')
    ax_cm_m1.set_ylabel('True')
    st.pyplot(fig_cm_m1)

    metrics_m1 = {
        "Accuracy": [accuracy_score(y_true_binary, y_pred_binary_m1)],
        "Precision": [precision_score(y_true_binary, y_pred_binary_m1, zero_division=0)],
        "Recall": [recall_score(y_true_binary, y_pred_binary_m1, zero_division=0)],
        "F1 Score": [f1_score(y_true_binary, y_pred_binary_m1, zero_division=0)],
    }
    metrics_df_m1 = pd.DataFrame(metrics_m1).T
    metrics_df_m1 = metrics_df_m1.rename(columns={0: 'Model 1'})
    st.dataframe(metrics_df_m1.style.format({0: '{:.4f}'}))

    # 모델 2 성능 지표 계산 및 시각화
    st.subheader("모델 2 성능 지표")
    y_pred_binary_m2 = [1 if r['predicted_mm_model2'] > 50 else 0 for r in results]
    
    cm_m2 = confusion_matrix(y_true_binary, y_pred_binary_m2)
    if cm_m2.shape == (2, 2):
        tn_m2, fp_m2, fn_m2, tp_m2 = cm_m2.ravel()
    else:
        tn_m2, fp_m2, fn_m2, tp_m2 = 0, 0, 0, 0

    st.markdown("### 혼동 행렬 (모델 2)")
    st.markdown("| | 예측 음성 (Negative) | 예측 양성 (Positive) |")
    st.markdown("|---|---|---|")
    st.markdown(f"| **실제 음성 (Negative)** | {tn_m2} (TN) | {fp_m2} (FP) |")
    st.markdown(f"| **실제 양성 (Positive)** | {fn_m2} (FN) | {tp_m2} (TP) |")

    fig_cm_m2, ax_cm_m2 = plt.subplots()
    sns.heatmap(cm_m2, annot=True, fmt='d', cmap='Blues', ax=ax_cm_m2)
    ax_cm_m2.set_xlabel('Predicted')
    ax_cm_m2.set_ylabel('True')
    st.pyplot(fig_cm_m2)

    metrics_m2 = {
        "Accuracy": [accuracy_score(y_true_binary, y_pred_binary_m2)],
        "Precision": [precision_score(y_true_binary, y_pred_binary_m2, zero_division=0)],
        "Recall": [recall_score(y_true_binary, y_pred_binary_m2, zero_division=0)],
        "F1 Score": [f1_score(y_true_binary, y_pred_binary_m2, zero_division=0)],
    }
    metrics_df_m2 = pd.DataFrame(metrics_m2).T
    metrics_df_m2 = metrics_df_m2.rename(columns={0: 'Model 2'})
    st.dataframe(metrics_df_m2.style.format({0: '{:.4f}'}))
    
    st.subheader("ROC 곡선 및 AUC")
    y_scores_m1 = [r['predicted_mm_model1'] / 100 for r in results]
    fpr_m1, tpr_m1, _ = roc_curve(y_true_binary, y_scores_m1)
    roc_auc_m1 = auc(fpr_m1, tpr_m1)

    y_scores_m2 = [r['predicted_mm_model2'] / 100 for r in results]
    fpr_m2, tpr_m2, _ = roc_curve(y_true_binary, y_scores_m2)
    roc_auc_m2 = auc(fpr_m2, tpr_m2)
    
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr_m1, y=tpr_m1, mode='lines', name=f'Model 1 ROC (AUC = {roc_auc_m1:.2f})'))
    fig_roc.add_trace(go.Scatter(x=fpr_m2, y=tpr_m2, mode='lines', name=f'Model 2 ROC (AUC = {roc_auc_m2:.2f})'))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(dash='dash')))
    fig_roc.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        showlegend=True
    )
    st.plotly_chart(fig_roc)

st.markdown("---")
st.subheader("분석 결과 다운로드")
if st.button("분석결과 엑셀 다운로드"):
    if st.session_state['analysis_results'] is None:
        st.warning("분석 결과를 먼저 생성해야 합니다.")
    else:
        results_df = pd.DataFrame(st.session_state['analysis_results'])
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name='Analysis_Results', index=False)
        buffer.seek(0)
        
        st.download_button(
            label="엑셀 파일 다운로드",
            data=buffer,
            file_name="analysis_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
