import numpy as np

weights = {
    "prevalence": 0.4,
    "age60": 0.25,
    "gender_diff": 0.2,
    "urban_rural_diff": 0.15
}

def zscore(value, mean, std):
    std = std if std != 0 else 1e-6
    return (value - mean) / std

def calculate_risk(data_row, stats):
    z_pre = zscore(data_row["prevalence"], stats["pre_mean"], stats["pre_std"])
    z_age = zscore(data_row["age60"], stats["age_mean"], stats["age_std"])
    z_gender = zscore(data_row["gender_diff"], stats["gender_mean"], stats["gender_std"])
    z_urban = zscore(data_row["urban_rural_diff"], stats["urban_mean"], stats["urban_std"])

    risk = (
        z_pre * weights["prevalence"] +
        z_age * weights["age60"] +
        z_gender * weights["gender_diff"] +
        z_urban * weights["urban_rural_diff"]
    )

    z_values = {
        "总体患病率": z_pre,
        "≥60岁比例": z_age,
        "性别差异": z_gender,
        "城乡差异": z_urban
    }
    return risk, z_values

def risk_level(risk_index):
    if risk_index < -0.5:
        return "🟢 低风险 | 健康水平良好，常规监测"
    elif risk_index < 0.5:
        return "🟡 中风险 | 危险因素可控，重点人群干预"
    else:
        return "🔴 高风险 | 整体风险偏高，强化全域防控"

def simulate_policy(z_values, pre_change=0, age_change=0, gender_change=0, urban_change=0, stats=None):
    if stats is None:
        raise ValueError("stats字典必须提供")

    delta_pre = pre_change / stats["pre_std"] if stats["pre_std"] != 0 else pre_change
    delta_age = age_change / stats["age_std"] if stats["age_std"] != 0 else age_change
    delta_gender = gender_change / stats["gender_std"] if stats["gender_std"] != 0 else gender_change
    delta_urban = urban_change / stats["urban_std"] if stats["urban_std"] != 0 else urban_change

    sim_pre = z_values["总体患病率"] + delta_pre
    sim_age = z_values["≥60岁比例"] + delta_age
    sim_gender = z_values["性别差异"] + delta_gender
    sim_urban = z_values["城乡差异"] + delta_urban

    sim_risk = (
        sim_pre * weights["prevalence"] +
        sim_age * weights["age60"] +
        sim_gender * weights["gender_diff"] +
        sim_urban * weights["urban_rural_diff"]
    )
    return sim_risk

def max_impact(z_values):
    impact = {
        "总体患病率": abs(z_values["总体患病率"] * weights["prevalence"]),
        "≥60岁比例": abs(z_values["≥60岁比例"] * weights["age60"]),
        "性别差异": abs(z_values["性别差异"] * weights["gender_diff"]),
        "城乡差异": abs(z_values["城乡差异"] * weights["urban_rural_diff"])
    }
    max_key = max(impact, key=impact.get)
    return max_key, impact[max_key]