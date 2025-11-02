from __future__ import annotations

import re
from typing import Iterable, Mapping

import pandas as pd
import streamlit as st

COLUMN_ALIASES: Mapping[str, str] = {
    '事故发生时间': '事故时间',
    '发生时间': '事故时间',
    '时间': '事故时间',
    '街道': '所在街道',
    '所属街道': '所在街道',
    '所属辖区': '所在区县',
    '辖区街道': '所在街道',
    '事故发生地点': '事故地点',
    '事故地址': '事故地点',
    '事故位置': '事故地点',
    '事故具体地址': '事故具体地点',
    '案件类型': '事故类型',
    '事故类别': '事故类型',
    '事故性质': '事故类型',
    '事故类型1': '事故类型',
}

ACCIDENT_TYPE_NORMALIZATION: Mapping[str, str] = {
    '财产损失': '财损',
    '财产损失事故': '财损',
    '一般程序': '伤人',
    '一般程序事故': '伤人',
    '伤人事故': '伤人',
    '造成人员受伤': '伤人',
    '造成人员死亡': '亡人',
    '死亡事故': '亡人',
    '亡人事故': '亡人',
    '亡人死亡': '亡人',
    '号': '财损',
}

REGION_FROM_LOCATION_PATTERN = re.compile(r'([一-龥]{2,8}(街道|新区|开发区|镇|区))')

REGION_NORMALIZATION: Mapping[str, str] = {
    '临城中队': '临城街道',
    '临城新区': '临城街道',
    '临城': '临城街道',
    '新城': '临城街道',
    '千岛中队': '千岛街道',
    '千岛新区': '千岛街道',
    '千岛': '千岛街道',
    '沈家门中队': '沈家门街道',
    '沈家门': '沈家门街道',
    '普陀城区': '沈家门街道',
    '普陀': '沈家门街道',
}


def _clean_text(series: pd.Series) -> pd.Series:
    """Strip whitespace and normalise obvious null placeholders."""
    cleaned = series.astype(str).str.strip()
    null_tokens = {'', 'nan', 'NaN', 'None', 'NULL', '<NA>', '无', '—'}
    return cleaned.mask(cleaned.isin(null_tokens))


def _maybe_seek_start(file_obj) -> None:
    if hasattr(file_obj, "seek"):
        try:
            file_obj.seek(0)
        except Exception:  # pragma: no cover - guard against non file-likes
            pass


def _prepare_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise a single sheet from the事故数据 workbook."""
    if df is None or df.empty:
        return pd.DataFrame()

    sheet = df.copy()
    # Normalise column names first
    sheet.columns = [str(col).strip() for col in sheet.columns]
    # If 栏目 still not recognised, attempt to locate header row inside the data
    if '事故时间' not in sheet.columns and '事故发生时间' not in sheet.columns:
        header_row = None
        for idx, row in sheet.iterrows():
            values = [str(cell).strip() for cell in row.tolist()]
            if '事故时间' in values or '事故发生时间' in values or '报警时间' in values:
                header_row = idx
                break
        if header_row is not None:
            sheet.columns = [str(x).strip() for x in sheet.iloc[header_row].tolist()]
            sheet = sheet.iloc[header_row + 1 :].reset_index(drop=True)
            sheet.columns = [str(col).strip() for col in sheet.columns]

    # Apply aliases after potential header relocation
    sheet = sheet.rename(columns={src: dst for src, dst in COLUMN_ALIASES.items() if src in sheet.columns})

    return sheet


def _coalesce_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    result = pd.Series(pd.NA, index=df.index, dtype="object")
    for col in columns:
        if col in df.columns:
            candidate = _clean_text(df[col])
            result = result.fillna(candidate)
    return result


def _infer_region_from_location(location: str) -> str | None:
    if pd.isna(location):
        return None
    text = str(location).strip()
    if not text:
        return None
    match = REGION_FROM_LOCATION_PATTERN.search(text)
    if match:
        return match.group(1)
    return None


def _normalise_region_series(series: pd.Series) -> pd.Series:
    return series.map(lambda val: REGION_NORMALIZATION.get(val, val) if pd.notna(val) else val)


def load_accident_records(accident_file, *, require_location: bool = False) -> pd.DataFrame:
    """
    Load accident records from the updated Excel template.

    The function supports workbooks with a single sheet (e.g. sample/事故处理/事故2021-2022.xlsx)
    as well as legacy multi-sheet formats where the header row might sit within the data.
    """
    _maybe_seek_start(accident_file)
    sheets = pd.read_excel(accident_file, sheet_name=None)
    if isinstance(sheets, dict):
        frames = [frame for frame in ( _prepare_sheet(df) for df in sheets.values() ) if not frame.empty]
    else:  # pragma: no cover - pandas only returns dict when sheet_name=None, but keep guard
        frames = [_prepare_sheet(sheets)]

    if not frames:
        raise ValueError("未在上传的事故数据中检测到有效的事故记录，请确认文件内容。")

    accident_df = pd.concat(frames, ignore_index=True)

    # Normalise columns of interest
    if '事故时间' not in accident_df.columns and '报警时间' in accident_df.columns:
        accident_df['事故时间'] = accident_df['报警时间']

    if '事故时间' not in accident_df.columns:
        raise ValueError("事故数据缺少“事故时间”字段，请确认模板是否为最新版本。")

    accident_df['事故时间'] = pd.to_datetime(accident_df['事故时间'], errors='coerce')

    # Location harmonisation (used for both region inference and hotspot analysis)
    location_columns_available = [col for col in ['事故具体地点', '事故地点'] if col in accident_df.columns]
    location_series = _coalesce_columns(accident_df, ['事故具体地点', '事故地点'])

    # Region handling
    region = _coalesce_columns(accident_df, ['所在街道', '所属街道', '所在区县', '辖区中队'])
    # Infer region from location fields when still missing
    if region.isna().any():
        inferred = location_series.map(_infer_region_from_location)
        region = region.fillna(inferred)
        region = region.fillna(_clean_text(location_series))

    region_clean = _clean_text(region)
    accident_df['所在街道'] = _normalise_region_series(region_clean)

    # Accident type normalisation
    accident_type = _coalesce_columns(accident_df, ['事故类型', '事故类别', '事故性质'])
    accident_type = accident_type.replace(ACCIDENT_TYPE_NORMALIZATION)
    accident_type = _clean_text(accident_type).replace(ACCIDENT_TYPE_NORMALIZATION)
    accident_df['事故类型'] = accident_type.fillna('财损')

    # Location column harmonisation
    if require_location and not location_columns_available and location_series.isna().all():
        raise ValueError("事故数据缺少“事故具体地点”字段，请确认模板是否与 sample/事故处理 中示例一致。")
    accident_df['事故具体地点'] = _clean_text(location_series)

    # Drop records with missing core fields
    subset = ['事故时间', '所在街道', '事故类型']
    if require_location:
        subset.append('事故具体地点')
    accident_df = accident_df.dropna(subset=subset)

    # Severity score
    severity_map = {'财损': 1, '伤人': 2, '亡人': 4}
    accident_df['severity'] = accident_df['事故类型'].map(severity_map).fillna(1).astype(int)

    accident_df = accident_df.sort_values('事故时间').reset_index(drop=True)

    return accident_df


@st.cache_data(show_spinner=False)
def load_and_clean_data(accident_file, strategy_file):
    accident_records = load_accident_records(accident_file)

    accident_data = accident_records.rename(
        columns={'事故时间': 'date_time', '所在街道': 'region', '事故类型': 'category'}
    )

    _maybe_seek_start(strategy_file)
    strategy_df = pd.read_excel(strategy_file)
    strategy_df = strategy_df.rename(columns=lambda col: str(col).strip())
    if '发布时间' not in strategy_df.columns:
        raise ValueError("策略数据缺少“发布时间”字段，请确认文件格式。")

    strategy_df['发布时间'] = pd.to_datetime(strategy_df['发布时间'], errors='coerce')
    if '交通策略类型' not in strategy_df.columns:
        raise ValueError("策略数据缺少“交通策略类型”字段，请确认文件格式。")

    strategy_df['交通策略类型'] = _clean_text(strategy_df['交通策略类型'])
    strategy_df = strategy_df.dropna(subset=['发布时间', '交通策略类型'])

    accident_data = accident_data[['date_time', 'region', 'category', 'severity']]
    strategy_df = strategy_df[['发布时间', '交通策略类型']].rename(
        columns={'发布时间': 'date_time', '交通策略类型': 'strategy_type'}
    )

    return accident_data, strategy_df


@st.cache_data(show_spinner=False)
def aggregate_daily_data(accident_data: pd.DataFrame, strategy_data: pd.DataFrame) -> pd.DataFrame:
    accident_data = accident_data.copy()
    strategy_data = strategy_data.copy()

    accident_data['date'] = accident_data['date_time'].dt.date
    daily_accidents = accident_data.groupby('date').agg(
        accident_count=('date_time', 'count'),
        severity=('severity', 'sum')
    )
    daily_accidents.index = pd.to_datetime(daily_accidents.index)

    strategy_data['date'] = strategy_data['date_time'].dt.date
    daily_strategies = strategy_data.groupby('date')['strategy_type'].apply(list)
    daily_strategies.index = pd.to_datetime(daily_strategies.index)

    combined = daily_accidents.join(daily_strategies, how='left')
    combined['strategy_type'] = combined['strategy_type'].apply(lambda x: x if isinstance(x, list) else [])
    combined = combined.asfreq('D')
    combined[['accident_count', 'severity']] = combined[['accident_count', 'severity']].fillna(0)
    combined['strategy_type'] = combined['strategy_type'].apply(lambda x: x if isinstance(x, list) else [])
    return combined


@st.cache_data(show_spinner=False)
def aggregate_daily_data_by_region(accident_data: pd.DataFrame, strategy_data: pd.DataFrame) -> pd.DataFrame:
    df = accident_data.copy()
    df['date'] = df['date_time'].dt.date
    g = df.groupby(['region', 'date']).agg(
        accident_count=('date_time', 'count'),
        severity=('severity', 'sum')
    )
    g.index = g.index.set_levels([g.index.levels[0], pd.to_datetime(g.index.levels[1])])
    g = g.sort_index()

    s = strategy_data.copy()
    s['date'] = s['date_time'].dt.date
    daily_strategies = s.groupby('date')['strategy_type'].apply(list)
    daily_strategies.index = pd.to_datetime(daily_strategies.index)

    regions = g.index.get_level_values(0).unique()
    dates = pd.date_range(g.index.get_level_values(1).min(), g.index.get_level_values(1).max(), freq='D')
    full_index = pd.MultiIndex.from_product([regions, dates], names=['region', 'date'])
    g = g.reindex(full_index).fillna(0)

    strat_map = daily_strategies.to_dict()
    g = g.assign(strategy_type=[strat_map.get(d, []) for d in g.index.get_level_values('date')])
    return g
