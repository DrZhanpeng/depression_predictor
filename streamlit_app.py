from gettext import install
import streamlit as st

st.title("🎈 My new Streamlit app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)


import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the model
model = joblib.load('XGBoost1.bin')

# Define feature options
性别_options = {
    1: 'Male (1)',
    2: 'Female (2)'
}

婚姻状态_options = {
    1: 'Married and cohabiting (1)',
    2: 'Married but separated/divorced/widowed/separated/unmarried (2)'
}

教育水平_options = {
    1: 'Illiteracy (1)',
    2: 'not having completed primary school/graduated from a private school/graduated from primary school  (2)',
    3: 'junior high school/senior high school/technical secondary school/junior college (3)',
    4: 'bachelor/master/doctoral (4)'
}

胃部疾病或消化系统疾病_options = {
    1: 'No (1)',
    2: 'Yes (2)'
}

与记忆有关的疾病_options = {
    1: 'No (1)',
    2: 'Yes (2)'
}

关节炎或风湿病_options = {
    1: 'No (1)',
    2: 'Yes (2)'
}

是否受疫情影响未能看病_options = {
    1: 'No (1)',
    2: 'Yes (2)'
}

是否摔倒过_options = {
    1: 'No (1)',
    2: 'Yes (2)'
}

是否疼痛而难受_options = {
    1: 'No (1)',
    2: 'Yes (2)'
}

是否干过十天以上农活_options = {
    1: 'No (1)',
    2: 'Yes (2)'
}

家户成员是否从事农业活动或销售自产农产品_options = {
    1: 'No (1)',
    2: 'Yes (2)'
}

是否已办理退休手续_options = {
    1: 'No (1)',
    2: 'Yes (2)'
}

饮酒频率_options = {
    1: 'Drinking alcohol more than once a month (1)',
    2: 'Drinking alcohol less than once a month  (2)',
    3: 'Drinking nothing (3)'
}

对生活是否满意_options = {
    1: 'Completely satisfied (1)',
    2: 'highly satisfied  (2)',
    3: 'basically satisfied (3)',
    4: 'partially satisfied (4)',
    5: 'completely unsatisfied (5)',
}

住房内有无洗澡设施_options = {
    1: 'Unified hot water supply (1)',
    2: 'Household self-installed water heaters  (2)',
    3: 'No bathing facilities in the residence (3)'
}

烹饪主要燃料_options = {
    1: 'Clean fuel (1)',
    2: 'non-clean fuel  (2)'
}

健康状况_options = {
    1: 'Very bad (1)',
    2: 'not good  (2)',
    3: 'average (3)',
    4: 'good (4)',
    5: 'very good (5)'
}

自身健康状况的预期_options = {
    1: 'Almost impossible (1)',
    2: 'unlikely  (2)',
    3: 'possible (3)',
    4: 'very likely (4)',
    5: 'almost certain (5)'
}

子女的关系满意程度_options = {
    1: 'Completely satisfied (1)',
    2: 'highly satisfied  (2)',
    3: 'basically satisfied (3)',
    4: 'partially satisfied (4)',
    5: 'completely dissatisfied (5)',
    6: 'no children now (6)'
}


# Define feature names
feature_names = [
    "Sex", "Educational level", "Marital status", "Stomach diseases or digestive system diseases", "Diseases related to memory",
    "Arthritis or rheumatismArthritis or rheumatism", "Have you been unable to see a doctor due to the impact of the epidemic", "Have you ever fallen down", "Is it painful and uncomfortable",
    "Average sleep time per night", "Frequency of alcohol consumption", "Activities of daily living", "Mini-mental State Examination Scale", "Are you satisfied with your life", "Have you ever done farm work for more than ten days", "Whether the household members are engaged in agricultural activities or sell self-produced agricultural products", "Are there any bathing facilities in the house", "Main fuel for cooking", "Health condition", "Expectations of one's own health condition", "The degree of satisfaction with the relationship among children", "Have you completed the retirement procedures"
]

# Streamlit user interface
st.title("Depression Predictor")

# age: numerical input
平均每晚睡着时间 = st.number_input("Average sleep time per night:", min_value=0, max_value=15, value=8)

日常生活活动能力 = st.number_input("Activities of daily living:", min_value=0, max_value=36, value=24)

简易智力状态检查量表 = st.number_input("Mini-mental State Examination Scale (MMSE):", min_value=0.00, max_value=24.00, value=15.75)
# sex: categorical selection
性别 = st.selectbox("Sex (2=Female, 1=Male):", options=[1, 2], format_func=lambda x: 'Female (2)' if x == 2 else 'Male (1)')

胃部疾病或消化系统疾病 = st.selectbox("Stomach diseases or digestive system diseases (0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

与记忆有关的疾病 = st.selectbox("Diseases related to memory (0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

关节炎或风湿病 = st.selectbox("Arthritis or rheumatism (0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

是否受疫情影响未能看病 = st.selectbox("Have you been unable to see a doctor due to the impact of the epidemic (0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

是否摔倒过 = st.selectbox("Have you ever fallen down (0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

是否疼痛而难受 = st.selectbox("Is it painful and uncomfortable (0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

是否干过十天以上农活 = st.selectbox("Have you ever done farm work for more than ten days (0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

家户成员是否从事农业活动或销售自产农产品 = st.selectbox("Whether the household members are engaged in agricultural activities or sell self-produced agricultural products (0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

是否已办理退休手续 = st.selectbox("Have you completed the retirement procedures (0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

烹饪主要燃料 = st.selectbox("Main fuel for cooking (0=No, 1=Yes):", options=[1, 2], format_func=lambda x: 'Clean fuel (1)' if x == 1 else 'non-clean fuel (2)')


# cp: categorical selection
教育水平 = st.selectbox("Educational level:", options=list(教育水平_options.keys()), format_func=lambda x: 教育水平_options[x])

婚姻状态 = st.selectbox("Marital status:", options=list(婚姻状态_options.keys()), format_func=lambda x: 婚姻状态_options[x])

饮酒频率 = st.selectbox("Frequency of alcohol consumption:", options=list(饮酒频率_options.keys()), format_func=lambda x: 饮酒频率_options[x])

对生活是否满意 = st.selectbox("Are you satisfied with your life:", options=list(对生活是否满意_options.keys()), format_func=lambda x: 对生活是否满意_options[x])

住房内有无洗澡设施 = st.selectbox("Are there any bathing facilities in the house:", options=list(住房内有无洗澡设施_options.keys()), format_func=lambda x: 住房内有无洗澡设施_options[x])

健康状况 = st.selectbox("Health condition:", options=list(健康状况_options.keys()), format_func=lambda x: 健康状况_options[x])

自身健康状况的预期 = st.selectbox("Expectations of one's own health condition:", options=list(自身健康状况的预期_options.keys()), format_func=lambda x: 自身健康状况的预期_options[x])

子女的关系满意程度 = st.selectbox("The degree of satisfaction with the relationship among children:", options=list(子女的关系满意程度_options.keys()), format_func=lambda x: 子女的关系满意程度_options[x])



# Process inputs and make predictions
feature_values = [性别, 教育水平, 婚姻状态, 胃部疾病或消化系统疾病, 与记忆有关的疾病, 关节炎或风湿病, 是否受疫情影响未能看病, 是否摔倒过, 是否疼痛而难受, 平均每晚睡着时间, 饮酒频率, 日常生活活动能力, 简易智力状态检查量表, 对生活是否满意, 是否干过十天以上农活, 家户成员是否从事农业活动或销售自产农产品, 住房内有无洗澡设施, 烹饪主要燃料,健康状况, 自身健康状况的预期, 子女的关系满意程度, 是否已办理退休手续]
features = np.array([feature_values])

if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of depression. "
            f"The model predicts that your probability of having depression is {probability:.1f}%. "
            "While this is just an estimate, it suggests that you may be at significant risk. "
            "I recommend that you consult a Psychologist as soon as possible for further evaluation and "
            "to ensure you receive an accurate diagnosis and necessary treatment."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of depression. "
            f"The model predicts that your probability of not having depression is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is still very important. "
            "I recommend regular check-ups to monitor your mental health, "
            "and to seek medical advice promptly if you experience any symptoms."
        )

    st.write(advice)

    try:
        # 首先尝试统一API
        explainer = shap.Explainer(model, feature_perturbation="tree_path_dependent")
        st.success("SHAP解释器创建成功（使用统一API）")
        shap_values = explainer(pd.DataFrame([feature_values], columns=feature_names))
    except Exception as e:
        st.warning(f"创建SHAP解释器时遇到问题: {str(e)}")
        st.warning("尝试备用方案...")
        
        # 备用方案：手动处理base_score
        base_value = 0.5  # 默认值
        
        # 尝试从模型中获取base_score
        if hasattr(model, 'base_score'):
            base_value = model.base_score
        elif hasattr(model, 'get_booster'):
            try:
                booster = model.get_booster()
                base_score_str = booster.attributes().get('base_score', '0.5')
                # 处理方括号
                if base_score_str.startswith('[') and base_score_str.endswith(']'):
                    base_score_str = base_score_str[1:-1]
                base_value = float(base_score_str)
            except Exception as e2:
                st.warning(f"获取base_score失败: {str(e2)}，使用默认值0.5")
                base_value = 0.5
        
        # 使用TreeExplainer并设置base_value
        try:
            explainer = shap.TreeExplainer(model, base_value=base_value)
            st.success("SHAP解释器创建成功（使用备用方案）")
            shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
        except Exception as e3:
            st.error(f"无法创建SHAP解释器: {str(e3)}")
            st.stop()
    
    # 生成并保存SHAP力图
    fig, ax = plt.subplots()
    shap.force_plot(
        explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value,
        shap_values[0] if isinstance(shap_values, list) else shap_values.values[0],
        pd.DataFrame([feature_values], columns=feature_names),
        matplotlib=True,
        show=False,
        figsize=(12, 4),
        text_rotation=15
    )
    
    # 直接在Streamlit中显示图表，而不是保存到文件
    st.pyplot(fig)







