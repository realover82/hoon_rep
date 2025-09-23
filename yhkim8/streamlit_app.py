import streamlit as st
import pandas as pd
import io
import plotly.express as px
from datetime import datetime, timedelta
import logging

# 'openpyxl'과 'plotly' 라이브러리가 설치되어 있어야 XLSX 파일을 처리하고 차트를 생성할 수 있습니다.
# 설치 명령어: pip install openpyxl plotly

# 로그 설정 (Streamlit 콘솔에 로그 출력)
logging.basicConfig(level=logging.INFO)

# --- XLSX 파일 분석 및 표시 함수 (메인 화면에서 호출) ---
def display_excel_analysis_result(uploaded_file):
    """업로드된 XLSX 파일 내용을 읽고 Streamlit에 표시하는 함수"""
    try:
        # XLSX 파일 읽기
        df = pd.read_excel(uploaded_file)
        st.session_state['df_data'] = df  # 업로드된 파일을 세션 상태에 저장
        st.success(f"'{uploaded_file.name}' 파일이 성공적으로 업로드되었습니다.")
        st.markdown("---")
        st.subheader("업로드된 파일 내용 미리보기")
        st.dataframe(df)
        
    except Exception as e:
        st.error(f"파일을 처리하는 중 오류가 발생했습니다: {e}")

# --- 메인 애플리케이션 로직 ---
def main():
    st.set_page_config(layout="wide")
    st.title("통합 데이터 분석 및 조회 시스템")
    st.markdown("---")

    # 세션 상태 초기화
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""
    if 'search_results_df' not in st.session_state:
        st.session_state.search_results_df = pd.DataFrame()
    if 'show_chart' not in st.session_state:
        st.session_state.show_chart = False
    
    # 사이드바: 파일 업로드
    with st.sidebar:
        st.header("엑셀 파일 업로드")
        st.write("분석을 원하는 XLSX 파일을 업로드하세요.")
        uploaded_file = st.file_uploader("파일 선택", type=["xlsx"])
        if uploaded_file:
            st.session_state['uploaded_file'] = uploaded_file
        else:
            if 'uploaded_file' in st.session_state:
                del st.session_state['uploaded_file']
            if 'df_data' in st.session_state:
                del st.session_state['df_data']
    
    # 메인 화면
    if 'uploaded_file' in st.session_state:
        # 파일이 업로드되었을 때만 분석 결과를 표시
        uploaded_file_obj = st.session_state['uploaded_file']
        display_excel_analysis_result(uploaded_file_obj)

        # --------------------------------------------------------------------------------
        # 파일 내용 검색 섹션
        # --------------------------------------------------------------------------------
        st.markdown("---")
        st.header("파일 내용 검색")
        
        # 날짜 범위 검색
        st.subheader("날짜 범위 검색")
        date_start = st.date_input("시작일", key="date_start")
        date_end = st.date_input("종료일", key="date_end")
        
        if st.button("날짜 검색"):
            if 'df_data' in st.session_state and not st.session_state.df_data.empty:
                df_to_use = st.session_state.df_data.copy()
                if '효력시작일' in df_to_use.columns:
                    try:
                        df_to_use['효력시작일'] = pd.to_datetime(df_to_use['효력시작일'])
                        mask = (df_to_use['효력시작일'].dt.date >= date_start) & \
                               (df_to_use['효력시작일'].dt.date <= date_end)
                        filtered_df = df_to_use.loc[mask].copy()
                        st.session_state.search_results_df = filtered_df
                        st.session_state.search_query = f"날짜 범위 ({date_start} ~ {date_end})"
                    except Exception as e:
                        st.error(f"날짜 열 형식이 올바르지 않습니다: {e}")
                        st.session_state.search_results_df = pd.DataFrame()
                else:
                    st.warning("날짜 검색을 위해 '효력시작일' 열이 필요합니다.")
                    st.session_state.search_results_df = pd.DataFrame()
            else:
                st.info("먼저 파일을 업로드해주세요.")
        
        # 일반 텍스트 검색 (3가지 검색창)
        st.subheader("텍스트 검색 (복합 검색)")
        search_query_name = st.text_input("자재명으로 검색", key="search_input_name")
        search_query_code = st.text_input("자재코드로 검색", key="search_input_code")
        search_query_supplier = st.text_input("공급업체로 검색", key="search_input_supplier")
        
        if st.button("텍스트 검색"):
            if 'df_data' in st.session_state and not st.session_state.df_data.empty:
                df_to_use = st.session_state.df_data.copy()
                
                # 검색 쿼리들이 모두 비어있는 경우
                if not search_query_name and not search_query_code and not search_query_supplier:
                    st.session_state.search_results_df = pd.DataFrame()
                    st.info("검색어를 입력해주세요.")
                else:
                    combined_mask = pd.Series([True] * len(df_to_use))
                    
                    if search_query_name and '자재명' in df_to_use.columns:
                        mask_name = df_to_use['자재명'].astype(str).str.contains(search_query_name, case=False, na=False)
                        combined_mask &= mask_name
                    if search_query_code and '자재코드' in df_to_use.columns:
                        mask_code = df_to_use['자재코드'].astype(str).str.contains(search_query_code, case=False, na=False)
                        combined_mask &= mask_code
                    if search_query_supplier and '공급업체' in df_to_use.columns:
                        mask_supplier = df_to_use['공급업체'].astype(str).str.contains(search_query_supplier, case=False, na=False)
                        combined_mask &= mask_supplier

                    filtered_df = df_to_use[combined_mask].copy()
                    st.session_state.search_results_df = filtered_df
                    st.session_state.search_query = f"{search_query_name or ''} {search_query_code or ''} {search_query_supplier or ''}".strip()
                    
            
        # --------------------------------------------------------------------------------
        # 결과 및 차트 섹션
        # --------------------------------------------------------------------------------
        
        # 검색 결과 표시
        if not st.session_state.search_results_df.empty:
            st.success("검색 결과:")
            st.dataframe(st.session_state.search_results_df)
        elif 'search_query' in st.session_state and st.session_state.search_query:
            st.warning(f"'{st.session_state.search_query}'에 대한 검색 결과가 없습니다.")
        
        st.markdown("---")
        st.header("가격 변동 차트")
        st.write("검색된 자재의 가격 변동 경과일수를 보여주는 차트를 생성합니다.")
        
        if st.button("차트 보기"):
            st.session_state.show_chart = True

        if st.session_state.show_chart and not st.session_state.search_results_df.empty:
            filtered_df = st.session_state.search_results_df

            if '효력시작일' in filtered_df.columns:
                try:
                    filtered_df['효력시작일'] = pd.to_datetime(filtered_df['효력시작일'])
                    today = datetime.now()
                    filtered_df['경과일수'] = (today - filtered_df['효력시작일']).dt.days

                    if not filtered_df.empty:
                        # 차트 라벨 생성: 자재명(자재코드)(공급업체)
                        label_cols = ['자재명', '자재코드', '공급업체']
                        for col in label_cols:
                            if col not in filtered_df.columns:
                                filtered_df[col] = '' # 해당 열이 없으면 빈 문자열로 채움
                        
                        filtered_df['차트_라벨'] = filtered_df.apply(
                            lambda row: f"{row['자재명']} ({row['자재코드']}) ({row['공급업체']})", axis=1
                        )
                        
                        fig = px.bar(
                            filtered_df.sort_values(by='경과일수', ascending=False),
                            x='경과일수',
                            y='차트_라벨',
                            orientation='h',
                            title=f'"{st.session_state.search_query}" 가격 변경 경과 일수',
                            labels={'경과일수': '경과 일수', '차트_라벨': '자재 정보'},
                            text='경과일수',
                            color_discrete_sequence=['darkorange']
                        )
                        fig.update_layout(
                            yaxis={'autorange': 'reversed'},
                            title_font_size=20,
                            margin={'t': 50, 'b': 20},
                            xaxis_title_font_size=14,
                            yaxis_title_font_size=14
                        )
                        fig.update_traces(texttemplate='%{text} days', textposition='outside')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("차트를 생성할 데이터가 없습니다. 먼저 검색을 해주세요.")

                except Exception as e:
                    st.error(f"차트를 생성하는 중 오류가 발생했습니다: {e}")
            else:
                st.warning("차트를 생성하려면 '효력시작일' 열이 포함되어 있어야 합니다.")
        elif st.session_state.show_chart and st.session_state.search_results_df.empty:
            st.info("차트를 생성하려면 먼저 검색을 해주세요.")

if __name__ == "__main__":
    main()
