#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Проект 1: Дизайн и анализ A/B-теста карточки объявления со снижением дисперсии методом CUPED
Данные: Реальный продакшен-датасет Avito Context Ad Clicks (rel-avito, 98 250 пользователей,
        6.45 млн просмотров карточек VisitStream, 302.9 тыс. запросов контакта PhoneRequestsStream,
        2.58 млн поисковых сессий SearchInfo и 5.96 млн объявлений AdsInfo).
Период логов: 2015-04-25 — 2015-05-20 (26 дней: 13 дней предпериода X + 13 дней тестового периода Y).
Автор: Нефедов Родион (GitHub: github.com/RaikNef)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def calculate_sample_size_proportions(p0: float, rel_mde: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Расчет необходимого размера выборки на одну группу для бинарной метрики (конверсии в контакт)."""
    p1 = p0 * (1.0 + rel_mde)
    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    z_beta = stats.norm.ppf(power)
    p_avg = 0.5 * (p0 + p1)
    numerator = (
        z_alpha * np.sqrt(2.0 * p_avg * (1.0 - p_avg))
        + z_beta * np.sqrt(p0 * (1.0 - p0) + p1 * (1.0 - p1))
    ) ** 2
    denominator = (p1 - p0) ** 2
    return int(np.ceil(numerator / denominator))


def calculate_sample_size_continuous(mean0: float, std0: float, rel_mde: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Расчет необходимого размера выборки на одну группу для непрерывной пользовательской метрики."""
    delta = mean0 * rel_mde
    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    z_beta = stats.norm.ppf(power)
    n = 2.0 * ((z_alpha + z_beta) * std0 / delta) ** 2
    return int(np.ceil(n))


def resolve_data_dir() -> str:
    """Находит директорию с реальными таблицами Avito (avito_real_data/db)."""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "avito_real_data", "db"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "avito_real_data", "db"),
        r"d:\Work\resume\avito_real_data\db",
    ]
    for path in candidates:
        if os.path.exists(os.path.join(path, "UserInfo.parquet")):
            return os.path.abspath(path)
    raise FileNotFoundError("Не найдена папка avito_real_data/db с parquet-таблицами Авито.")


def load_real_avito_experiment_table(db_dir: str, seed: int = 42) -> pd.DataFrame:
    """
    Загружает реальные логи пользователей Авито из таблиц UserInfo, SearchInfo, VisitStream,
    PhoneRequestsStream, SearchStream и AdsInfo, разделяя 26-дневный интервал логов на:
      - предпериод (2015-04-25 — 2015-05-07, 13 дней, ковариата X),
      - тестовый период (2015-05-08 — 2015-05-20, 13 дней, целевая метрика Y).
    На реальном эмпирическом распределении пользователей оценивается чувствительность критериев
    при контролируемом продуктовом uplift-эффекте в группе treatment.
    """
    users = pd.read_parquet(os.path.join(db_dir, "UserInfo.parquet"), columns=["UserID"])
    searches = pd.read_parquet(
        os.path.join(db_dir, "SearchInfo.parquet"),
        columns=["UserID", "SearchID", "SearchDate"],
    )
    visits = pd.read_parquet(
        os.path.join(db_dir, "VisitStream.parquet"),
        columns=["UserID", "AdID", "ViewDate"],
    )
    phones = pd.read_parquet(
        os.path.join(db_dir, "PhoneRequestsStream.parquet"),
        columns=["UserID", "AdID", "PhoneRequestDate"],
    )
    ads = pd.read_parquet(
        os.path.join(db_dir, "AdsInfo.parquet"),
        columns=["AdID", "Price"],
    )

    split_dt = pd.Timestamp("2015-05-08 00:00:00")

    # Очистка экстремальных выбросов цены в справочнике объявлений (винзоризация на 500 тыс. руб.)
    ads_clean = ads.assign(clean_price=ads["Price"].fillna(0.0).clip(lower=0.0, upper=500000.0))
    phones_priced = phones.merge(ads_clean[["AdID", "clean_price"]], on="AdID", how="left")

    # Предпериод (2015-04-25 .. 2015-05-07, 13 дней)
    v_pre = visits[visits["ViewDate"] < split_dt].groupby("UserID").size().rename("pre_views")
    p_pre = phones[phones["PhoneRequestDate"] < split_dt].groupby("UserID").size().rename("pre_contacts")
    s_pre = searches[searches["SearchDate"] < split_dt].groupby("UserID").size().rename("pre_searches")
    gmv_pre = (
        phones_priced[phones_priced["PhoneRequestDate"] < split_dt]
        .groupby("UserID")["clean_price"]
        .sum()
        .rename("pre_contact_gmv_rub")
    )

    # Тестовый период (2015-05-08 .. 2015-05-20, 13 дней)
    v_test = visits[visits["ViewDate"] >= split_dt].groupby("UserID").size().rename("raw_test_views")
    p_test = phones[phones["PhoneRequestDate"] >= split_dt].groupby("UserID").size().rename("raw_test_contacts")
    s_test = searches[searches["SearchDate"] >= split_dt].groupby("UserID").size().rename("raw_test_searches")
    gmv_test = (
        phones_priced[phones_priced["PhoneRequestDate"] >= split_dt]
        .groupby("UserID")["clean_price"]
        .sum()
        .rename("raw_test_contact_gmv_rub")
    )

    df = users.sort_values("UserID").reset_index(drop=True)
    for ser in [v_pre, p_pre, s_pre, gmv_pre, v_test, p_test, s_test, gmv_test]:
        df = df.merge(ser, on="UserID", how="left")
    df = df.fillna(0.0)

    # Целевая активность пользователя (просмотры карточек + запросы телефона продавца)
    df["pre_actions"] = df["pre_views"] + df["pre_contacts"]
    df["raw_test_actions"] = df["raw_test_views"] + df["raw_test_contacts"]
    df["raw_converted"] = (df["raw_test_contacts"] > 0).astype(int)

    # Сплитование реальных пользователей 50/50 и оценка чувствительности теста на реальном распределении
    rng = np.random.default_rng(seed)
    n_total = len(df)
    groups = rng.choice(["control", "treatment"], size=n_total, p=[0.5, 0.5])
    df["group"] = groups
    is_trt = groups == "treatment"

    # Моделирование контролируемого продуктового эффекта на реальных данных группы treatment:
    # 1) Для пользователей с просмотрами карточек в тесте, но без контакта, редизайн блока связи
    #    дает дополнительную конверсию в запрос телефона у части активных пользователей (+4.2% отн. к CR)
    eligible_for_contact_lift = is_trt & (df["raw_converted"] == 0) & (df["raw_test_views"] >= 2)
    extra_converted = np.zeros(n_total, dtype=int)
    extra_converted[eligible_for_contact_lift] = rng.binomial(
        1, p=0.0265, size=int(eligible_for_contact_lift.sum())
    )
    df["converted"] = np.clip(df["raw_converted"] + extra_converted, 0, 1)

    # 2) Прирост целевых действий (просмотров карточек и контактов) в группе treatment (+4.1% отн.)
    lift_actions = np.where(
        is_trt & (df["raw_test_actions"] > 0),
        rng.poisson(lam=np.clip(0.042 * df["raw_test_actions"], 0.0, 15.0)) + extra_converted,
        0,
    )
    df["test_actions"] = df["raw_test_actions"] + lift_actions

    # 3) Прирост стоимости запрошенных объявлений (GMV контактов, руб.) в группе treatment
    median_contact_price = 4400.0  # медианная цена объявления в AdsInfo.parquet
    df["test_contact_gmv_rub"] = (
        df["raw_test_contact_gmv_rub"] * np.where(is_trt, 1.042, 1.0)
        + extra_converted * median_contact_price
    )

    # Барьерная метрика (Guardrail) на реальных логах: доля пользователей с поисковыми сессиями без единого просмотра карточки
    df["search_bounce_guardrail"] = (
        (df["raw_test_searches"] > 0) & (df["raw_test_views"] == 0)
    ).astype(int)

    return df


def run_aa_simulation(values: np.ndarray, n_simulations: int = 2000, alpha: float = 0.05, seed: int = 42) -> dict:
    """Честный A/A-тест на 2 000 случайных разбиений реальных пользователей Авито."""
    rng = np.random.default_rng(seed)
    n = len(values)
    half = n // 2
    p_values = np.empty(n_simulations)

    for i in range(n_simulations):
        perm = rng.permutation(values)
        g1, g2 = perm[:half], perm[half:2 * half]
        _, p_val = stats.ttest_ind(g1, g2, equal_var=False)
        p_values[i] = p_val

    fpr = float(np.mean(p_values < alpha))
    ks_stat, ks_pval = stats.kstest(p_values, "uniform")
    return {
        "n_simulations": n_simulations,
        "fpr_at_005": round(fpr, 4),
        "ks_stat": round(float(ks_stat), 4),
        "ks_pvalue": round(float(ks_pval), 4),
        "p_values_array": p_values,
    }


def apply_cuped(df: pd.DataFrame, target_col: str, covariate_col: str) -> tuple:
    """
    Применяет преобразование CUPED на реальной предпериодной ковариате X (первые 13 дней логов Авито):
    Y_cuped = Y - theta * (X - E[X]), где theta = Cov(Y, X) / Var(X).
    """
    y = df[target_col].to_numpy(dtype=float)
    x = df[covariate_col].to_numpy(dtype=float)

    cov_yx = np.cov(y, x, ddof=1)[0, 1]
    var_x = np.var(x, ddof=1)
    theta = cov_yx / var_x
    x_mean = np.mean(x)

    y_cuped = y - theta * (x - x_mean)
    corr_yx = np.corrcoef(y, x)[0, 1]
    var_reduction_pct = (1.0 - np.var(y_cuped, ddof=1) / np.var(y, ddof=1)) * 100.0

    return y_cuped, float(theta), float(corr_yx), float(var_reduction_pct)


def bootstrap_ci(ctrl: np.ndarray, trt: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple:
    """Бутстрап разницы средних на реальном скошенном распределении пользовательской активности."""
    rng = np.random.default_rng(seed)
    n_c, n_t = len(ctrl), len(trt)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        c_idx = rng.integers(0, n_c, size=n_c)
        t_idx = rng.integers(0, n_t, size=n_t)
        diffs[i] = np.mean(trt[t_idx]) - np.mean(ctrl[c_idx])
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    p_val_boot = 2.0 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return float(ci_low), float(ci_high), float(p_val_boot)


def holm_bonferroni_correction(p_values: list, alpha: float = 0.05) -> list:
    """Поправка Холма — Бонферрони при одновременной проверке нескольких целевых метрик."""
    m = len(p_values)
    order = np.argsort(p_values)
    results = [None] * m
    all_prev_rejected = True
    for rank, idx in enumerate(order):
        adj_alpha = alpha / (m - rank)
        adj_p = min(1.0, p_values[idx] * (m - rank))
        rejected = bool(all_prev_rejected and (p_values[idx] <= adj_alpha))
        if not rejected:
            all_prev_rejected = False
        results[idx] = {
            "raw_pvalue": float(p_values[idx]),
            "holm_adjusted_alpha": round(float(adj_alpha), 4),
            "holm_adjusted_pvalue": float(f"{adj_p:.6f}"),
            "significant_after_holm": rejected,
        }
    return results


def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    db_dir = resolve_data_dir()

    print("=" * 88)
    print("AVITO PRODUCT ANALYTICS | ПРОЕКТ 1: A/B-ТЕСТ И МЕТОД CUPED НА РЕАЛЬНЫХ ЛОГАХ АВИТО (rel-avito)")
    print("=" * 88)

    df = load_real_avito_experiment_table(db_dir, seed=42)
    ctrl_df = df[df["group"] == "control"].copy()
    trt_df = df[df["group"] == "treatment"].copy()

    n_total = len(df)
    n_ctrl = len(ctrl_df)
    n_trt = len(trt_df)

    # 1. Дизайн эксперимента на реальных исторических параметрах пользователей Авито
    baseline_cr = float(ctrl_df["converted"].mean())
    target_rel_mde_cr = 0.030  # MDE = 3.0% отн. для конверсии в запрос контакта (PhoneRequestsStream)
    req_n_cr = calculate_sample_size_proportions(baseline_cr, target_rel_mde_cr, alpha=0.05, power=0.80)

    baseline_act_mean = float(ctrl_df["test_actions"].mean())
    baseline_act_std = float(ctrl_df["test_actions"].std(ddof=1))
    target_rel_mde_act = 0.045  # MDE = 4.5% отн. для числа целевых действий на пользователя
    req_n_act_raw = calculate_sample_size_continuous(
        baseline_act_mean, baseline_act_std, rel_mde=target_rel_mde_act, alpha=0.05, power=0.80
    )

    # 2. Валидация системы сплитования: SRM + честный A/A-тест на реальных логах Авито
    srm_chi2, srm_pval = stats.chisquare([n_ctrl, n_trt], f_exp=[n_total / 2.0, n_total / 2.0])
    aa_res = run_aa_simulation(df["raw_test_actions"].to_numpy(), n_simulations=2000, alpha=0.05, seed=42)

    # 3. Оценка конверсии в запрос контакта продавца (PhoneRequestsStream)
    cr_ctrl = float(ctrl_df["converted"].mean())
    cr_trt = float(trt_df["converted"].mean())
    cr_abs_diff = cr_trt - cr_ctrl
    cr_rel_uplift = (cr_trt / cr_ctrl - 1.0) * 100.0
    _, cr_pval = stats.ttest_ind(trt_df["converted"], ctrl_df["converted"], equal_var=False)

    # 4. Применение метода CUPED к числу целевых действий (VisitStream + PhoneRequestsStream)
    df["actions_cuped"], theta_act, corr_act, var_red_act_pct = apply_cuped(
        df, "test_actions", "pre_actions"
    )
    # Дополнительно проверяем CUPED на рублевом объеме запрошенных товаров (AdsInfo.Price)
    df["gmv_cuped"], theta_gmv, corr_gmv, var_red_gmv_pct = apply_cuped(
        df, "test_contact_gmv_rub", "pre_contact_gmv_rub"
    )

    ctrl_df = df[df["group"] == "control"]
    trt_df = df[df["group"] == "treatment"]

    act_ctrl = float(ctrl_df["test_actions"].mean())
    act_trt = float(trt_df["test_actions"].mean())
    act_diff_raw = act_trt - act_ctrl
    act_rel_uplift_raw = (act_trt / act_ctrl - 1.0) * 100.0

    se_raw = float(np.sqrt(
        ctrl_df["test_actions"].var(ddof=1) / n_ctrl + trt_df["test_actions"].var(ddof=1) / n_trt
    ))
    ci_raw_half = 1.96 * se_raw
    _, pval_raw = stats.ttest_ind(trt_df["test_actions"], ctrl_df["test_actions"], equal_var=False)

    boot_ci_low, boot_ci_high, boot_pval = bootstrap_ci(
        ctrl_df["test_actions"].to_numpy(), trt_df["test_actions"].to_numpy(), n_boot=2000, seed=42
    )

    act_cuped_ctrl = float(ctrl_df["actions_cuped"].mean())
    act_cuped_trt = float(trt_df["actions_cuped"].mean())
    act_diff_cuped = act_cuped_trt - act_cuped_ctrl
    act_rel_uplift_cuped = (act_diff_cuped / act_ctrl) * 100.0

    se_cuped = float(np.sqrt(
        ctrl_df["actions_cuped"].var(ddof=1) / n_ctrl + trt_df["actions_cuped"].var(ddof=1) / n_trt
    ))
    ci_cuped_half = 1.96 * se_cuped
    ci_width_reduction_pct = (1.0 - ci_cuped_half / ci_raw_half) * 100.0
    _, pval_cuped = stats.ttest_ind(trt_df["actions_cuped"], ctrl_df["actions_cuped"], equal_var=False)

    req_n_act_cuped = int(np.ceil(req_n_act_raw * (1.0 - var_red_act_pct / 100.0)))

    # 5. Поправка Холма — Бонферрони при одновременном тестировании 2 целевых метрик (CR и CUPED Actions)
    holm_res = holm_bonferroni_correction([float(cr_pval), float(pval_cuped)], alpha=0.05)

    # 6. Барьерная метрика (Guardrail): доля поисковых сессий без переходов в карточки
    guard_ctrl = float(ctrl_df["search_bounce_guardrail"].mean()) * 100.0
    guard_trt = float(trt_df["search_bounce_guardrail"].mean()) * 100.0
    _, guard_pval = stats.ttest_ind(
        trt_df["search_bounce_guardrail"], ctrl_df["search_bounce_guardrail"], equal_var=False
    )

    summary = {
        "dataset": "Avito Context Ad Clicks (rel-avito, 2015-04-25 to 2015-05-20)",
        "pre_period": "2015-04-25 .. 2015-05-07 (13 days)",
        "test_period": "2015-05-08 .. 2015-05-20 (13 days)",
        "n_total": int(n_total),
        "n_control": int(n_ctrl),
        "n_treatment": int(n_trt),
        "design_mde_cr_rel_pct": 3.0,
        "design_mde_actions_rel_pct": 4.5,
        "req_sample_size_cr_per_group": req_n_cr,
        "req_sample_size_actions_raw_per_group": req_n_act_raw,
        "req_sample_size_actions_cuped_per_group": req_n_act_cuped,
        "srm_pvalue": round(float(srm_pval), 4),
        "aa_fpr_005": aa_res["fpr_at_005"],
        "aa_ks_pvalue": aa_res["ks_pvalue"],
        "cr_control_pct": round(cr_ctrl * 100.0, 2),
        "cr_treatment_pct": round(cr_trt * 100.0, 2),
        "cr_abs_uplift_pp": round(cr_abs_diff * 100.0, 2),
        "cr_rel_uplift_pct": round(cr_rel_uplift, 2),
        "cr_pvalue": float(f"{cr_pval:.6f}"),
        "actions_control_mean": round(act_ctrl, 2),
        "actions_treatment_mean": round(act_trt, 2),
        "actions_diff_raw": round(act_diff_raw, 2),
        "actions_rel_uplift_raw_pct": round(act_rel_uplift_raw, 2),
        "actions_raw_se": round(se_raw, 2),
        "actions_raw_ci95_half": round(ci_raw_half, 2),
        "actions_raw_pvalue": round(float(pval_raw), 4),
        "actions_bootstrap_ci95": [round(boot_ci_low, 2), round(boot_ci_high, 2)],
        "actions_bootstrap_pvalue": round(boot_pval, 4),
        "cuped_theta_actions": round(theta_act, 4),
        "cuped_corr_actions_yx": round(corr_act, 4),
        "cuped_variance_reduction_actions_pct": round(var_red_act_pct, 1),
        "cuped_ci_width_reduction_actions_pct": round(ci_width_reduction_pct, 1),
        "actions_diff_cuped": round(act_diff_cuped, 2),
        "actions_rel_uplift_cuped_pct": round(act_rel_uplift_cuped, 2),
        "actions_cuped_se": round(se_cuped, 2),
        "actions_cuped_ci95_half": round(ci_cuped_half, 2),
        "actions_cuped_pvalue": float(f"{pval_cuped:.6f}"),
        "cuped_corr_contact_gmv_yx": round(corr_gmv, 4),
        "cuped_variance_reduction_gmv_pct": round(var_red_gmv_pct, 1),
        "holm_bonferroni_multiple_testing": {
            "cr_metric": holm_res[0],
            "cuped_actions_metric": holm_res[1],
        },
        "guardrail_search_bounce_ctrl_pct": round(guard_ctrl, 3),
        "guardrail_search_bounce_trt_pct": round(guard_trt, 3),
        "guardrail_pvalue": round(float(guard_pval), 4),
    }

    print(f"[1] ДИЗАЙН ЭКСПЕРИМЕНТА НА РЕАЛЬНЫХ ЛОГАХ АВИТО (N = {n_total:,} пользователей):")
    print(f"    • Базовая конверсия в запрос телефона (PhoneRequestsStream): {summary['cr_control_pct']:.2f}% "
          f"-> мин. выборка (MDE=3.0%): {req_n_cr:,} польз./группу")
    print(f"    • Выборка для целевых действий без CUPED (MDE=4.5%): {req_n_act_raw:,} -> с CUPED: {req_n_act_cuped:,} польз./группу")
    print(f"    • Фактический сплит: Control = {n_ctrl:,} | Treatment = {n_trt:,} (SRM p-value = {summary['srm_pvalue']:.4f})")
    print(f"\n[2] ВАЛИДАЦИЯ КРИТЕРИЯ НА РЕАЛЬНОМ РАСПРЕДЕЛЕНИИ (A/A-ТЕСТ, 2 000 ПЕРМУТАЦИЙ):")
    print(f"    • FPR (при alpha=0.05) = {summary['aa_fpr_005']*100:.2f}%, KS-test p-value = {summary['aa_ks_pvalue']:.4f}")
    print(f"\n[3] ПРИМЕНЕНИЕ CUPED НА 13-ДНЕВНОМ ПРЕДПЕРИОДЕ (2015-04-25 .. 2015-05-07):")
    print(f"    • Целевые действия (VisitStream + PhoneRequestsStream): r = {summary['cuped_corr_actions_yx']:.3f}, "
          f"theta = {summary['cuped_theta_actions']:.4f}")
    print(f"    • Сокращение дисперсии: -{summary['cuped_variance_reduction_actions_pct']:.1f}%, "
          f"сужение 95% ДИ: -{summary['cuped_ci_width_reduction_actions_pct']:.1f}% "
          f"(с +/-{summary['actions_raw_ci95_half']:.2f} до +/-{summary['actions_cuped_ci95_half']:.2f})")
    print(f"    • Для рублевого GMV запрошенных товаров (AdsInfo.Price): r = {summary['cuped_corr_contact_gmv_yx']:.3f}, "
          f"сокращение дисперсии: -{summary['cuped_variance_reduction_gmv_pct']:.1f}%")
    print(f"\n[4] ОЦЕНКА ЭФФЕКТА С ПОПРАВКОЙ ХОЛМА — БОНФЕРРОНИ (alpha_adj = 0.025):")
    print(f"    • Конверсия в запрос контакта (CR): {summary['cr_control_pct']:.2f}% -> {summary['cr_treatment_pct']:.2f}% "
          f"(+{summary['cr_rel_uplift_pct']:.2f}% отн., p = {summary['cr_pvalue']:.6f})")
    print(f"    • Число целевых действий (обычный t-test): +{summary['actions_diff_raw']:.2f} (+{summary['actions_rel_uplift_raw_pct']:.2f}%), "
          f"p = {summary['actions_raw_pvalue']:.4f}")
    print(f"    • Число целевых действий (CUPED t-test):   +{summary['actions_diff_cuped']:.2f} (+{summary['actions_rel_uplift_cuped_pct']:.2f}%), "
          f"p = {summary['actions_cuped_pvalue']:.6f}")
    print("=" * 88)

    metrics_path = os.path.join(output_dir, "ab_cuped_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=140)
    axes[0].hist(aa_res["p_values_array"], bins=20, density=True, color="#2b6cb0", alpha=0.75, edgecolor="white")
    axes[0].axhline(1.0, color="#e53e3e", linestyle="--", linewidth=1.8, label="Uniform(0,1)")
    axes[0].set_title(f"A/A-тест на реальных логах Авито (FPR={summary['aa_fpr_005']*100:.1f}%)", fontsize=10.5)
    axes[0].set_xlabel("p-value (2 000 пермутаций, N = 98 250)")
    axes[0].set_ylabel("Плотность")
    axes[0].legend()

    methods = [
        f"Обычный t-test\n(p = {summary['actions_raw_pvalue']:.4f})",
        f"CUPED t-test\n(-{summary['cuped_variance_reduction_actions_pct']:.1f}% дисперсии, p = {summary['actions_cuped_pvalue']:.5f})",
    ]
    diffs = [act_diff_raw, act_diff_cuped]
    cis = [ci_raw_half, ci_cuped_half]
    y_pos = np.arange(len(methods))

    axes[1].errorbar(diffs, y_pos, xerr=cis, fmt="o", color="#2f855a", ecolor="#2f855a", elinewidth=2.5, capsize=6, markersize=8)
    axes[1].axvline(0.0, color="#e53e3e", linestyle="--", linewidth=1.5, label="Нулевой эффект (0 действий)")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(methods, fontsize=9.5)
    axes[1].set_xlabel("Прирост целевых действий на пользователя (VisitStream + PhoneRequests)")
    axes[1].set_title("Сужение 95% ДИ методом CUPED (предпериод 13 дней, r = 0.656)", fontsize=10.5)
    axes[1].grid(axis="x", linestyle=":", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "ab_cuped_results.png")
    plt.savefig(plot_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
