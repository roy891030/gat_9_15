#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV 欄位自動檢測和修復工具

這個工具會：
1. 讀取 CSV 檔案
2. 自動檢測欄位名稱
3. 生成適合的 COLMAPS 配置
4. 修改 build_artifacts.py
"""

import pandas as pd
import sys

def detect_columns(csv_path):
    """檢測 CSV 欄位"""
    print("=" * 60)
    print("CSV 欄位自動檢測")
    print("=" * 60)

    # 讀取 CSV
    try:
        df = pd.read_csv(csv_path, nrows=5)
    except Exception as e:
        print(f"❌ 讀取 CSV 失敗: {e}")
        return None

    columns = df.columns.tolist()

    print(f"\n檔案: {csv_path}")
    print(f"總欄位數: {len(columns)}")
    print("\n所有欄位：")
    for i, col in enumerate(columns, 1):
        print(f"  {i:2d}. {col}")

    # 嘗試匹配關鍵欄位
    print("\n" + "=" * 60)
    print("關鍵欄位匹配")
    print("=" * 60)

    date_col = find_column(columns, ['年月日', 'date', 'Date', '日期', 'trading_date', 'trade_date'])
    code_col = find_column(columns, ['證券代碼', 'code', 'Code', '代碼', 'stock_code', 'symbol', 'ticker'])
    close_col = find_column(columns, ['收盤價(元)', 'close', 'Close', '收盤價', 'adj_close'])
    open_col = find_column(columns, ['開盤價(元)', 'open', 'Open', '開盤價'])
    high_col = find_column(columns, ['最高價(元)', 'high', 'High', '最高價'])
    low_col = find_column(columns, ['最低價(元)', 'low', 'Low', '最低價'])
    volume_col = find_column(columns, ['成交量(千股)', 'volume', 'Volume', '成交量', 'vol'])
    industry_col = find_column(columns, ['TEJ產業_名稱', 'industry', 'Industry', '產業', 'sector'])

    print(f"\n日期欄位: {date_col or '❌ 未找到'}")
    print(f"代碼欄位: {code_col or '❌ 未找到'}")
    print(f"收盤價欄位: {close_col or '❌ 未找到'}")
    print(f"開盤價欄位: {open_col or '（可選）'}")
    print(f"最高價欄位: {high_col or '（可選）'}")
    print(f"最低價欄位: {low_col or '（可選）'}")
    print(f"成交量欄位: {volume_col or '（可選）'}")
    print(f"產業欄位: {industry_col or '（可選）'}")

    if not (date_col and code_col and close_col):
        print("\n❌ 缺少關鍵欄位！")
        print("\n請手動指定欄位名稱：")
        print("  date_col = input('日期欄位名稱: ')")
        print("  code_col = input('代碼欄位名稱: ')")
        print("  close_col = input('收盤價欄位名稱: ')")
        return None

    # 生成 COLMAP
    colmap = {
        "date": date_col,
        "code": code_col,
        "close": close_col,
    }

    if open_col:
        colmap["open"] = open_col
    if high_col:
        colmap["high"] = high_col
    if low_col:
        colmap["low"] = low_col
    if volume_col:
        colmap["volk"] = volume_col
    if industry_col:
        colmap["industry"] = industry_col

    print("\n" + "=" * 60)
    print("生成的 COLMAP:")
    print("=" * 60)
    print("{")
    for key, val in colmap.items():
        print(f'    "{key}": "{val}",')
    print("}")

    return colmap

def find_column(columns, candidates):
    """在欄位列表中查找匹配的欄位"""
    for candidate in candidates:
        for col in columns:
            if candidate.lower() in col.lower():
                return col
    return None

def generate_colmap_code(colmap):
    """生成 Python 代碼"""
    code = "# 自動生成的 COLMAP\nCOLMAPS = [\n    {\n"
    for key, val in colmap.items():
        code += f'        "{key}": "{val}",\n'
    code += "    }\n]\n"
    return code

if __name__ == "__main__":
    csv_path = "unique_2019q3to2025q3.csv"

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

    colmap = detect_columns(csv_path)

    if colmap:
        print("\n" + "=" * 60)
        print("下一步：")
        print("=" * 60)
        print("\n1. 複製上面的 COLMAP")
        print("2. 修改 build_artifacts.py 的 COLMAPS（第 83 行）")
        print("3. 或者運行：")
        print(f"   python {__file__} {csv_path} > colmap.txt")
        print("=" * 60)
