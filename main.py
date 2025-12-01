"""
CAS1100-01 Project - Students Performance Analysis
Student ID: 2015147529
Name: 김창원

필요 패키지:
    pip install pandas matplotlib seaborn scikit-learn statsmodels

사용 데이터:
    StudentsPerformance.csv  (Kaggle Students Performance 데이터셋)

제안서 대비 차별점:
    - average_score 외에 다음 파생 변수를 추가로 생성하여 분석 범위를 확장
        1) reading_writing_gap : 읽기/쓰기 점수 차이
        2) stem_bias           : 수학 점수가 언어(읽기+쓰기) 평균보다 얼마나 높은지
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import statsmodels.api as sm
from statsmodels.formula.api import ols

# =========================
# 전역 설정
# =========================

DATA_PATH = "StudentsPerformance.csv"
OUTPUT_DIR = Path("outputs")


# =========================
# 1. 데이터 로딩 및 전처리
# =========================

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    CSV를 읽고 기본적인 정리 및 파생변수들을 생성한다.
    - 컬럼명 정리 (공백 -> 언더스코어)
    - 평균 점수 average_score
    - 추가 파생변수:
        reading_writing_gap, stem_bias
    """
    df = pd.read_csv(path)

    # 컬럼명 정리 (공백 -> 언더스코어)
    df = df.rename(columns={
        "race/ethnicity": "race_ethnicity",
        "parental level of education": "parental_education",
        "test preparation course": "test_preparation",
        "math score": "math_score",
        "reading score": "reading_score",
        "writing score": "writing_score",
    })

    # 중복 행 제거
    df = df.drop_duplicates().reset_index(drop=True)

    # 평균 점수 파생 변수
    df["average_score"] = df[["math_score",
                              "reading_score",
                              "writing_score"]].mean(axis=1)

    # 제안서에는 없던 추가 파생변수 2개
    df["reading_writing_gap"] = df["reading_score"] - df["writing_score"]
    df["stem_bias"] = df["math_score"] - (
        df["reading_score"] + df["writing_score"]
    ) / 2.0

    return df


def encode_features(df: pd.DataFrame):
    """
    범주형 변수 원-핫 인코딩 후, 설명변수(X)와 타깃(y)을 반환한다.
    학업 성취도(average_score)를 타깃으로 사용하며
    성적 점수(수학/읽기/쓰기)는 설명변수에서 제외한다.
    """
    df_clean = df.dropna().copy()

    categorical_cols = [
        "gender",
        "race_ethnicity",
        "parental_education",
        "lunch",
        "test_preparation",
    ]

    df_encoded = pd.get_dummies(df_clean, columns=categorical_cols,
                                drop_first=True)

    # 외부 요인만으로 average_score를 설명하기 위해
    # 개별 점수 컬럼은 설명변수에서 제거
    drop_cols = [
        "math_score",
        "reading_score",
        "writing_score",
        "average_score",
        # 파생변수들도 설명변수에는 사용하지 않고
        # 기술 통계/시각화용으로만 활용
        "reading_writing_gap",
        "stem_bias",
    ]
    feature_cols = [c for c in df_encoded.columns if c not in drop_cols]

    X = df_encoded[feature_cols]
    y = df_encoded["average_score"]

    return df_clean, X, y, feature_cols


# =========================
# 2. 기술 통계 및 시각화
# =========================

def descriptive_stats(df: pd.DataFrame) -> None:
    """
    기술 통계, 그룹별 평균, 박스플롯을 생성하여 파일로 저장한다.
    - score_summary.csv
    - group_means.txt
    - box_lunch.png, box_test_prep.png, box_parental_education.png
    """
    # 전체 점수 요약 통계
    summary = df[["math_score",
                  "reading_score",
                  "writing_score",
                  "average_score"]].describe()
    summary.to_csv(OUTPUT_DIR / "score_summary.csv", encoding="utf-8-sig")

    # 그룹별 평균 (부모 학력, 점심 유형, 시험 준비, 성별)
    group_cols = [
        "parental_education",
        "lunch",
        "test_preparation",
        "gender",
    ]
    with open(OUTPUT_DIR / "group_means.txt", "w", encoding="utf-8") as f:
        for col in group_cols:
            f.write(f"\n==== {col} 별 average_score 평균 ====\n")
            means = (df.groupby(col)["average_score"]
                     .mean()
                     .sort_values(ascending=False))
            f.write(str(means))
            f.write("\n")

    # 점심 유형별 평균 점수 박스플롯
    plt.figure()
    sns.boxplot(data=df, x="lunch", y="average_score")
    plt.title("점심 유형별 평균 점수 분포")
    plt.xlabel("lunch")
    plt.ylabel("average_score")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "box_lunch.png")
    plt.close()

    # 시험 준비 여부별 평균 점수 박스플롯
    plt.figure()
    sns.boxplot(data=df, x="test_preparation", y="average_score")
    plt.title("시험 준비 여부별 평균 점수 분포")
    plt.xlabel("test_preparation")
    plt.ylabel("average_score")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "box_test_prep.png")
    plt.close()

    # 부모 학력별 평균 점수 박스플롯
    plt.figure()
    sns.boxplot(
        data=df,
        x="parental_education",
        y="average_score",
        order=[
            "some high school",
            "high school",
            "some college",
            "associate's degree",
            "bachelor's degree",
            "master's degree",
        ],
    )
    plt.xticks(rotation=30, ha="right")
    plt.title("부모 학력 수준별 평균 점수 분포")
    plt.xlabel("parental_education")
    plt.ylabel("average_score")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "box_parental_education.png")
    plt.close()


def correlation_analysis(df: pd.DataFrame) -> None:
    """
    점수 간 상관관계 히트맵을 생성한다.
    - corr_scores.csv
    - corr_scores.png
    """
    numeric_cols = ["math_score",
                    "reading_score",
                    "writing_score",
                    "average_score"]
    corr = df[numeric_cols].corr()
    corr.to_csv(OUTPUT_DIR / "corr_scores.csv", encoding="utf-8-sig")

    plt.figure()
    sns.heatmap(corr, annot=True, vmin=-1, vmax=1, square=True)
    plt.title("점수 간 상관관계 히트맵")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "corr_scores.png")
    plt.close()


# =========================
# 3. 다중회귀분석 (OLS)
# =========================

def run_ols(X, y, feature_cols):
    """
    OLS 다중회귀분석을 수행하고 결과 요약을 파일로 저장한다.
    - ols_summary.txt
    """
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()

    with open(OUTPUT_DIR / "ols_summary.txt", "w", encoding="utf-8") as f:
        f.write(model.summary().as_text())

    print("\n[OLS 회귀분석 요약]")
    print(model.summary().tables[1])

    return model


# =========================
# 4. Random Forest 회귀
# =========================

def run_random_forest(X, y, feature_cols):
    """
    Random Forest Regressor를 사용해 변수 중요도를 계산한다.
    - rf_feature_importances.csv
    - rf_feature_importances.png
    - rf_metrics.txt (Train/Test R^2)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 약간의 규제를 줘서 테스트 성능을 안정화
    rf = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        max_depth=4,
        min_samples_leaf=20,
    )
    rf.fit(X_train, y_train)

    r2_train = rf.score(X_train, y_train)
    r2_test = rf.score(X_test, y_test)

    importances = pd.Series(
        rf.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    importances.to_csv(
        OUTPUT_DIR / "rf_feature_importances.csv", encoding="utf-8-sig"
    )

    plt.figure()
    importances.head(15).sort_values().plot(kind="barh")
    plt.title("Random Forest Feature Importance (상위 15개)")
    plt.xlabel("중요도")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rf_feature_importances.png")
    plt.close()

    with open(OUTPUT_DIR / "rf_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Train R^2: {r2_train:.4f}\n")
        f.write(f"Test  R^2: {r2_test:.4f}\n")

    print("\n[Random Forest 성능]")
    print(f"Train R^2: {r2_train:.4f}")
    print(f"Test  R^2: {r2_test:.4f}")
    print("\n[Random Forest 상위 중요 변수]")
    print(importances.head(10))

    return rf, importances


# =========================
# 5. ANOVA 분석
# =========================

def run_anova(df: pd.DataFrame):
    """
    성별, 인종, 부모 학력, 점심, 시험 준비가 average_score에
    미치는 영향을 ANOVA로 검정한다.
    - anova_table.csv
    """
    formula = (
        "average_score ~ C(gender) + C(race_ethnicity) + "
        "C(parental_education) + C(lunch) + C(test_preparation)"
    )
    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    anova_table.to_csv(
        OUTPUT_DIR / "anova_table.csv", encoding="utf-8-sig"
    )

    print("\n[ANOVA 결과]")
    print(anova_table)

    return anova_table


# =========================
# main
# - 데이터 로딩 → 기술 통계/시각화 → 인코딩
#   → OLS 회귀 → Random Forest → ANOVA 순서로 실행
# =========================

def main():
    # CSV 파일 존재 여부 체크 (오류 처리)
    if not Path(DATA_PATH).exists():
        print(f"[ERROR] 데이터 파일을 찾을 수 없습니다: {DATA_PATH}")
        print("같은 폴더에 StudentsPerformance.csv가 있는지 확인하세요.")
        sys.exit(1)

    print("데이터 로딩 중...")
    df = load_data(DATA_PATH)

    print("기술 통계 및 시각화 생성 중...")
    descriptive_stats(df)
    correlation_analysis(df)

    print("범주형 변수 인코딩 및 회귀/모델링 준비 중...")
    df_clean, X, y, feature_cols = encode_features(df)

    print("OLS 다중회귀분석 수행 중...")
    run_ols(X, y, feature_cols)

    print("Random Forest 회귀 및 Feature Importance 계산 중...")
    run_random_forest(X, y, feature_cols)

    print("ANOVA 분석 수행 중...")
    run_anova(df_clean)

    print("\n분석이 완료되었습니다. 'outputs' 폴더의 결과 파일들을 확인하세요.")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    main()
