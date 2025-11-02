from __future__ import annotations

from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd

LOCATION_KEYWORDS: tuple[str, ...] = (
    "路",
    "道",
    "街",
    "巷",
    "路口",
    "交叉口",
    "大道",
    "公路",
    "口",
)
AREA_KEYWORDS: tuple[str, ...] = (
    "新城",
    "临城",
    "千岛",
    "翁山",
    "海天",
    "海宇",
    "定沈",
    "滨海",
    "港岛",
    "体育",
    "长升",
    "金岛",
    "桃湾",
)

LOCATION_MAPPING: dict[str, str] = {
    "新城千岛路": "千岛路",
    "千岛路海天大道": "千岛路海天大道口",
    "海天大道千岛路": "千岛路海天大道口",
    "新城翁山路": "翁山路",
    "翁山路金岛路": "翁山路金岛路口",
    "海天大道临长路": "海天大道临长路口",
    "定沈路卫生医院门口": "定沈路医院段",
    "翁山路海城路西口": "翁山路海城路口",
    "海宇道路口": "海宇道",
    "海天大道路口": "海天大道",
    "定沈路交叉路口": "定沈路",
    "千岛路路口": "千岛路",
    "体育路路口": "体育路",
    "金岛路路口": "金岛路",
}

SEVERITY_MAP: dict[str, int] = {"财损": 1, "伤人": 2, "亡人": 4}


def _extract_road_info(location: str | float | None) -> str:
    if pd.isna(location):
        return "未知路段"
    text = str(location)
    for keyword in LOCATION_KEYWORDS + AREA_KEYWORDS:
        if keyword in text:
            words = text.replace("，", " ").replace(",", " ").split()
            for word in words:
                if keyword in word:
                    return word
            return text
    return text[:20] if len(text) > 20 else text


def prepare_hotspot_dataset(accident_records: pd.DataFrame) -> pd.DataFrame:
    df = accident_records.copy()
    required_defaults: dict[str, str] = {
        "道路类型": "未知道路类型",
        "路口路段类型": "未知路段",
        "事故具体地点": "未知路段",
        "事故类型": "财损",
        "所在街道": "未知街道",
    }
    for column, default_value in required_defaults.items():
        if column not in df.columns:
            df[column] = default_value
        else:
            df[column] = df[column].fillna(default_value)

    if "severity" not in df.columns:
        df["severity"] = df["事故类型"].map(SEVERITY_MAP).fillna(1).astype(int)

    df["事故时间"] = pd.to_datetime(df["事故时间"], errors="coerce")
    df = df.dropna(subset=["事故时间"]).sort_values("事故时间").reset_index(drop=True)
    df["standardized_location"] = (
        df["事故具体地点"].apply(_extract_road_info).replace(LOCATION_MAPPING)
    )
    return df


def analyze_hotspot_frequency(df: pd.DataFrame, time_window: str = "7D") -> pd.DataFrame:
    recent_cutoff = df["事故时间"].max() - pd.Timedelta(time_window)

    overall_stats = df.groupby("standardized_location").agg(
        accident_count=("事故时间", "count"),
        last_accident=("事故时间", "max"),
        main_accident_type=("事故类型", _mode_fallback),
        main_road_type=("道路类型", _mode_fallback),
        main_intersection_type=("路口路段类型", _mode_fallback),
        total_severity=("severity", "sum"),
    )

    recent_stats = (
        df[df["事故时间"] >= recent_cutoff]
        .groupby("standardized_location")
        .agg(
            recent_count=("事故时间", "count"),
            recent_accident_type=("事故类型", _mode_fallback),
            recent_severity=("severity", "sum"),
        )
    )

    result = (
        overall_stats.merge(recent_stats, left_index=True, right_index=True, how="left")
        .fillna({"recent_count": 0, "recent_severity": 0})
        .fillna("")
    )
    result["recent_count"] = result["recent_count"].astype(int)
    result["trend_ratio"] = result["recent_count"] / result["accident_count"]
    result["days_since_last"] = (
        df["事故时间"].max() - result["last_accident"]
    ).dt.days.astype(int)
    result["avg_severity"] = result["total_severity"] / result["accident_count"]
    return result.sort_values(["recent_count", "accident_count"], ascending=False)


def calculate_hotspot_risk_score(hotspot_df: pd.DataFrame) -> pd.DataFrame:
    df = hotspot_df.copy()
    if df.empty:
        return df

    df["frequency_score"] = (df["accident_count"] / df["accident_count"].max() * 40).clip(
        0, 40
    )
    df["trend_score"] = (df["trend_ratio"] * 30).clip(0, 30)
    severity_map = {"财损": 5, "伤人": 15, "亡人": 20}
    df["severity_score"] = df["main_accident_type"].map(severity_map).fillna(5)
    df["urgency_score"] = ((30 - df["days_since_last"]) / 30 * 10).clip(0, 10)
    df["risk_score"] = (
        df["frequency_score"]
        + df["trend_score"]
        + df["severity_score"]
        + df["urgency_score"]
    )
    conditions = [
        df["risk_score"] >= 70,
        df["risk_score"] >= 50,
        df["risk_score"] >= 30,
    ]
    choices = ["高风险", "中风险", "低风险"]
    df["risk_level"] = np.select(conditions, choices, default="一般风险")
    return df.sort_values("risk_score", ascending=False)


def generate_hotspot_strategies(
    hotspot_df: pd.DataFrame, time_period: str = "本周"
) -> list[dict[str, str | float]]:
    strategies: list[dict[str, str | float]] = []
    for location_name, location_data in hotspot_df.iterrows():
        accident_count = float(location_data["accident_count"])
        recent_count = float(location_data.get("recent_count", 0))
        accident_type = str(location_data.get("main_accident_type", "财损"))
        intersection_type = str(location_data.get("main_intersection_type", "普通路段"))
        trend_ratio = float(location_data.get("trend_ratio", 0))
        risk_level = str(location_data.get("risk_level", "一般风险"))

        base_info = f"{time_period}对【{location_name}】"
        data_support = (
            f"（近期{int(recent_count)}起，累计{int(accident_count)}起，{accident_type}为主）"
        )

        strategy_parts: list[str] = []
        if "信号灯" in intersection_type:
            if accident_type == "财损":
                strategy_parts.extend(["加强闯红灯查处", "优化信号配时", "整治不按规定让行"])
            else:
                strategy_parts.extend(["完善人行过街设施", "加强非机动车管理", "设置警示标志"])
        elif "普通路段" in intersection_type:
            strategy_parts.extend(["加强巡逻管控", "整治违法停车", "设置限速标志"])
        else:
            strategy_parts.extend(["分析事故成因", "制定综合整治方案"])

        if risk_level == "高风险":
            strategy_parts.extend(["列为重点整治路段", "开展专项整治行动"])
        elif risk_level == "中风险":
            strategy_parts.append("加强日常监管")

        if trend_ratio > 0.4:
            strategy_parts.append("近期重点监控")

        strategy_text = (
            base_info + "，" + "，".join(strategy_parts) + data_support
            if strategy_parts
            else base_info + "加强交通安全管理" + data_support
        )

        strategies.append(
            {
                "location": location_name,
                "strategy": strategy_text,
                "risk_level": risk_level,
                "accident_count": accident_count,
                "recent_count": recent_count,
            }
        )
    return strategies


def serialise_datetime_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column in result.columns and pd.api.types.is_datetime64_any_dtype(result[column]):
            result[column] = result[column].dt.strftime("%Y-%m-%d %H:%M:%S")
    return result


def _mode_fallback(series: pd.Series) -> str:
    if series.empty:
        return ""
    mode = series.mode()
    return str(mode.iloc[0]) if not mode.empty else str(series.iloc[0])

