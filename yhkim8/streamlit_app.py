import streamlit as st
import os
import re
import math
import csv
import glob
import random
import time
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as tvm
# sklearn.metrics에서 specificity_score 임포트 제거
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import zipfile
import shutil
from torchsummary import summary
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
from contextlib import redirect_stdout
import pandas as pd

# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(layout="wide")
st.title("다이얼 게이지 자동 분석 애플리케이션 📊")

# 세션 상태 초기화
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'analysis_results' not in st.session_state:
    st.session_state['analysis_results'] = None
if 'model_summary' not in st.session_state:
    st.session_state['model_summary'] = None
if 'finetune_preview' not in st.session_state:
    st.session_state['finetune_preview'] = None
if 'metrics_calculated' not in st.session_state:
    st.session_state['metrics_calculated'] = False

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

# PDF 리포트 생성 함수
def create_pdf_report(filename, results, cm_fig_path, roc_fig_path):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>다이얼 게이지 분석 보고서</b>", styles['Heading1']))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("<b>1. 성능 지표</b>", styles['Heading2']))
    story.append(Spacer(1, 0.1 * inch))
    
    y_true_binary = [1 if r['true_mm'] > 0.5 else 0 for r in results]
    y_pred_binary = [1 if r['predicted_psi_rad_model1'] > math.pi else 0 for r in results]
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        specificity = 0

    story.append(Paragraph(f"Accuracy: {accuracy:.4f}", styles['Normal']))
    story.append(Paragraph(f"Precision: {precision:.4f}", styles['Normal']))
    story.append(Paragraph(f"Recall (Sensitivity): {recall:.4f}", styles['Normal']))
    story.append(Paragraph(f"F1 Score: {f1:.4f}", styles['Normal']))
    story.append(Paragraph(f"Specificity: {specificity:.4f}", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))
    
    story.append(Paragraph("<b>2. 혼동 행렬 (Confusion Matrix)</b>", styles['Heading2']))
    story.append(ReportLabImage(cm_fig_path, width=4*inch, height=4*inch))
    story.append(Spacer(1, 0.2 * inch))
    # 혼동 행렬 텍스트 설명 추가
    cm_text = f"""
    혼동 행렬은 모델의 예측 결과가 실제 정답과 얼마나 일치하는지 보여줍니다.
    - True Positive (TP): 실제 양성(Positive)인 것을 양성으로 올바르게 예측한 경우: {tp}
    - True Negative (TN): 실제 음성(Negative)인 것을 음성으로 올바르게 예측한 경우: {tn}
    - False Positive (FP): 실제 음성인 것을 양성으로 잘못 예측한 경우 (제1종 오류): {fp}
    - False Negative (FN): 실제 양성인 것을 음성으로 잘못 예측한 경우 (제2종 오류): {fn}
    """
    story.append(Paragraph(cm_text.replace('\n', '<br/>'), styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("<b>3. ROC 곡선 (ROC Curve)</b>", styles['Heading2']))
    story.append(ReportLabImage(roc_fig_path, width=4*inch, height=4*inch))
    # ROC AUC 텍스트 설명 추가
    roc_text = f"""
    ROC 곡선은 모델의 분류 성능을 시각화합니다. 곡선이 왼쪽 위 모서리에 가까울수록 성능이 우수합니다.
    - 곡선 아래 면적(AUC)은 {auc:.2f}입니다. AUC 값은 0과 1 사이이며, 1에 가까울수록 모델의 성능이 좋습니다.
    - 0.5에 가까운 AUC 값은 모델이 무작위 추측(Random Guess)과 다르지 않음을 의미합니다.
    """
    story.append(Paragraph(roc_text.replace('\n', '<br/>'), styles['Normal']))


    doc.build(story)
    
# =========================================================
# Streamlit UI
# =========================================================
st.sidebar.header("⚙️ 모델 설정")

model_choice = st.sidebar.selectbox("모델 선택", ("ResNet-18 (AngleHead)", "YOLOv5 (예정)"))
if model_choice == "YOLOv5 (예정)":
    st.sidebar.warning("YOLOv5는 현재 더미 모델로 구현되어 있으며, 실제 기능은 없습니다.")

load_mode = st.sidebar.radio("모델 가중치 로드", ("파인튜닝", "무작위 초기화"))

# 파인튜닝 레이어 설정 목록 버튼
st.sidebar.header("파인튜닝 설정")
if st.session_state['model'] is not None:
    layer_names = [name for name, param in st.session_state['model'].named_parameters()]
    finetune_layers_list = st.sidebar.multiselect(
        "파인튜닝할 레이어 선택",
        options=layer_names,
        default=['backbone.fc.weight', 'backbone.fc.bias']
    )
else:
    finetune_layers_list = []
    st.sidebar.warning("모델을 로드하면 레이어 목록이 표시됩니다.")

# 파인튜닝 설정 적용 버튼 (재학습 기능은 현재 없음)
if st.button("파인튜닝 설정 적용"):
    if st.session_state['model'] is None:
        st.warning("먼저 모델을 로드하거나 초기화해주세요.")
    else:
        try:
            for name, param in st.session_state['model'].named_parameters():
                if name in finetune_layers_list:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            st.success("파인튜닝 레이어 설정이 적용되었습니다.")
            st.info("이 설정은 '분석 시작' 버튼에 반영됩니다.")
        except Exception as e:
            st.error(f"파인튜닝 설정 적용 중 오류 발생: {e}")

st.sidebar.text("")

# =========================================================
# 파일 업로드 섹션 (수정됨)
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
# 기능 버튼 (수정됨)
# =========================================================
st.header("🚀 성능 테스트 실행")

# 모델 구조 보기 버튼
if st.button("모델 구조 보기"):
    st.info("모델 구조를 분석 중입니다...")
    try:
        if st.session_state['model'] is None:
            st.warning("모델이 로드되지 않았습니다.")
        else:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                summary(st.session_state['model'], (3, 224, 224), device=str(device))
            st.session_state['model_summary'] = buffer.getvalue()
            st.success("모델 구조 분석 완료!")
    except Exception as e:
        st.error(f"모델 구조 분석 중 오류 발생: {e}")

if st.session_state['model_summary']:
    st.subheader("모델 구조 상세")
    st.code(st.session_state['model_summary'])

# 파인튜닝 레이어 설정 미리보기 버튼
if st.button("파인튜닝 레이어 미리보기"):
    st.info("파인튜닝 레이어 설정을 미리보기 합니다...")
    if st.session_state['model']:
        preview_text = "파인튜닝에 사용될 레이어:\n"
        for name, param in st.session_state['model'].named_parameters():
            preview_text += f"- {name} ({'학습 가능' if param.requires_grad else '고정'})\n"
        st.session_state['finetune_preview'] = preview_text
        st.success("미리보기 완료!")
    else:
        st.warning("먼저 모델을 로드하거나 초기화해주세요.")

if st.session_state['finetune_preview']:
    st.subheader("파인튜닝 설정 미리보기")
    st.text(st.session_state['finetune_preview'])

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

    # 두 모델 초기화 및 가중치 로드
    st.info("두 모델을 로딩 중...")
    try:
        model1 = AngleHead(pretrained=False).to(device)
        model2 = AngleHead(pretrained=False).to(device)
        
        # 파일 업로드 데이터를 메모리에서 바로 사용
        model1.load_state_dict(torch.load(io.BytesIO(uploaded_model1_pth.read()), map_location=device))
        model2.load_state_dict(torch.load(io.BytesIO(uploaded_model2_pth.read()), map_location=device))
        
        model1.eval()
        model2.eval()
        st.success("두 모델 로딩 완료!")

    except Exception as e:
        st.error(f"모델 로딩 중 오류 발생: {e}")
        st.stop()
    
    # 텍스트 출력 헤더
    st.header("📊 개별 파일 분석 결과")
    results = []

    for fp in test_files:
        mm_from_name = parse_mm_prefix(fp)
        if mm_from_name is None:
            st.warning(f"파일명에서 mm 값을 파싱할 수 없습니다: {os.path.basename(fp)}. 이 파일은 분석에서 제외됩니다.")
            continue

        with torch.no_grad():
            x = tfm(Image.open(fp).convert("L")).unsqueeze(0).to(device)
            
            # 모델 1 예측
            y1 = model1(x)[0].cpu().numpy()
            # y좌표가 이미지 좌표계와 반대이므로 -를 붙여줌
            psi_pred1 = wrap_angle(math.atan2(float(-y1[0]), float(y1[1])))
            
            # 모델 2 예측
            y2 = model2(x)[0].cpu().numpy()
            # y좌표가 이미지 좌표계와 반대이므로 -를 붙여줌
            psi_pred2 = wrap_angle(math.atan2(float(-y2[0]), float(y2[1])))

        # 텍스트로 결과 출력
        st.text(f"--- 파일: {os.path.basename(fp)} ---")
        st.text(f"실제 값: {mm_from_name * 100:.2f} mm")
        st.text(f"모델 1 예측 값: {(psi_pred1 / TWO_PI) * 100:.2f} mm")
        st.text(f"모델 2 예측 값: {(psi_pred2 / TWO_PI) * 100:.2f} mm")
        st.text("-----------------------------------")
        
        results.append({
            "filepath": os.path.basename(fp),
            "true_mm": mm_from_name,
            "predicted_psi_rad_model1": psi_pred1,
            "predicted_psi_rad_model2": psi_pred2,
        })
        
    st.session_state['analysis_results'] = results
    st.success("분석 완료!")


if st.session_state['analysis_results']:
    st.header(f"📋 분석 결과")
    results = st.session_state['analysis_results']

    # 모델 1 성능 지표 계산 및 시각화
    st.subheader("모델 1 성능 지표")
    y_true_binary = [1 if r['true_mm'] > 0.5 else 0 for r in results]
    y_pred_binary_m1 = [1 if r['predicted_psi_rad_model1'] > math.pi else 0 for r in results]
    
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

    st.write("**Accuracy:**", accuracy_score(y_true_binary, y_pred_binary_m1))
    st.write("**Precision:**", precision_score(y_true_binary, y_pred_binary_m1, zero_division=0))
    st.write("**Recall:**", recall_score(y_true_binary, y_pred_binary_m1, zero_division=0))
    st.write("**F1 Score:**", f1_score(y_true_binary, y_pred_binary_m1, zero_division=0))

    # 모델 2 성능 지표 계산 및 시각화
    st.subheader("모델 2 성능 지표")
    y_pred_binary_m2 = [1 if r['predicted_psi_rad_model2'] > math.pi else 0 for r in results]
    
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

    st.write("**Accuracy:**", accuracy_score(y_true_binary, y_pred_binary_m2))
    st.write("**Precision:**", precision_score(y_true_binary, y_pred_binary_m2, zero_division=0))
    st.write("**Recall:**", recall_score(y_true_binary, y_pred_binary_m2, zero_division=0))
    st.write("**F1 Score:**", f1_score(y_true_binary, y_pred_binary_m2, zero_division=0))
    
    st.subheader("ROC 곡선 및 AUC")
    y_scores_m1 = [r['predicted_psi_rad_model1'] / TWO_PI for r in results]
    fpr_m1, tpr_m1, _ = roc_curve(y_true_binary, y_scores_m1)
    roc_auc_m1 = auc(fpr_m1, tpr_m1)

    y_scores_m2 = [r['predicted_psi_rad_model2'] / TWO_PI for r in results]
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
st.subheader("변형된 모델 저장")
# 이 부분은 단일 모델을 다루므로 원본 코드 유지
if st.button("변형된 모델 저장"):
    st.warning("이 기능은 단일 모델에만 적용됩니다. Model 1 또는 Model 2 중 하나를 수동으로 저장해야 합니다.")


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
