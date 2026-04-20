import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st

from meal_model import MealRecommendationSystem, RecipeRecommender, UserProfile

# ============================================================================
# UNIFIED CATEGORICAL VALUES
# ============================================================================

ALLERGIES = ["Lactose", "Gluten", "Nuts", "None"]
FITNESS_GOALS = ["Lose weight", "Gain weight", "Improve health", "Maintain weight", "Build muscle"]
ACTIVITY_LEVELS = ["Sedentary", "Light", "Moderate", "High"]
DIETARY_PREFERENCES = ["High protein", "Vegan", "Low carb", "Keto"]
HEALTH_CONDITIONS = ["High Blood Pressure", "Heart Disease", "Diabetes", "None"]

CALORIES_RANGE = (1500, 2500)
PROTEIN_RANGE = (40, 200)
CARBS_RANGE = (130, 300)
FATS_RANGE = (30, 100)

BASE_DIR = Path(__file__).resolve().parent
RECIPES_PATH = BASE_DIR / "recips.json"
BEST_MODEL_PATH = BASE_DIR / "best_nutrition_model.joblib"
OLD_MODEL_PATH = BASE_DIR / "nutrition_model.joblib"


def get_model_path() -> str:
    if BEST_MODEL_PATH.exists():
        return str(BEST_MODEL_PATH)
    return str(OLD_MODEL_PATH)


def normalize_text(value: Any) -> str:
    return str(value).strip().lower()


def recipe_search(recipes: List[Dict[str, Any]], query: str, top_n: int = 5) -> List[Dict[str, Any]]:
    query_tokens = [t for t in normalize_text(query).replace("?", " ").replace(",", " ").split() if t]
    if not query_tokens:
        return recipes[:top_n]

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for recipe in recipes:
        text = " ".join([
            str(recipe.get("name", "")),
            " ".join(recipe.get("ingredients", [])),
            " ".join(recipe.get("diet", [])),
            " ".join(recipe.get("diseases", [])),
            " ".join(recipe.get("categories", [])),
        ]).lower()

        score = 0
        for token in query_tokens:
            if token in text:
                score += 1

        if ("diabetes" in query_tokens or "سكر" in query or "سكري" in query) and "diabetes" in recipe.get("diseases", []):
            score += 2
        if ("protein" in query_tokens or "بروتين" in query) and float(recipe.get("protein", 0)) >= 18:
            score += 2
        if ("keto" in query_tokens) and "keto" in recipe.get("diet", []):
            score += 2
        if ("vegan" in query_tokens or "نباتي" in query) and "vegan" in recipe.get("diet", []):
            score += 2
        if (("low" in query_tokens and "carb" in query_tokens) or "قليل الكارب" in query or "low carb" in query.lower()):
            if float(recipe.get("carbs", 999)) <= 35:
                score += 2

        if score > 0:
            scored.append((score, recipe))

    if not scored:
        return sorted(recipes, key=lambda r: float(r.get("protein", 0)), reverse=True)[:top_n]

    scored.sort(key=lambda x: (x[0], float(x[1].get("protein", 0))), reverse=True)
    return [recipe for _, recipe in scored[:top_n]]


def local_chat_answer(recipes: List[Dict[str, Any]], question: str, user_profile: Optional[Dict[str, Any]] = None) -> str:
    matches = recipe_search(recipes, question, top_n=5)
    q = normalize_text(question)

    if not matches:
        return "مش لاقي نتائج مناسبة في الداتا الحالية."

    if "compare" in q or "قارن" in q:
        if len(matches) >= 2:
            a, b = matches[0], matches[1]
            return (
                f"**مقارنة سريعة**\n\n"
                f"**{a['name']}**: {a['calories']} kcal, {a['protein']}g protein, {a['carbs']}g carbs, {a['fats']}g fats\n\n"
                f"**{b['name']}**: {b['calories']} kcal, {b['protein']}g protein, {b['carbs']}g carbs, {b['fats']}g fats"
            )

    if "highest protein" in q or "اعلى بروتين" in q or "أعلى بروتين" in q:
        best = max(matches, key=lambda x: float(x.get("protein", 0)))
        return f"أعلى وجبة بروتين من الداتا هي **{best['name']}** وفيها **{best['protein']}g protein**."

    if "lowest calories" in q or "اقل سعرات" in q or "أقل سعرات" in q:
        best = min(matches, key=lambda x: float(x.get("calories", 0)))
        return f"أقل وجبة سعرات من الداتا هي **{best['name']}** وفيها **{best['calories']} kcal**."

    lines = [
        f"- **{item['name']}** — {item['calories']} kcal | {item['protein']}g protein | {item['carbs']}g carbs | {item['fats']}g fats"
        for item in matches
    ]
    return "دي أفضل النتائج من `recips.json`:\n\n" + "\n".join(lines)


def openrouter_answer(recipes: List[Dict[str, Any]], question: str, user_profile: Optional[Dict[str, Any]] = None) -> str:
    api_key = st.secrets.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return local_chat_answer(recipes, question, user_profile)

    matches = recipe_search(recipes, question, top_n=8)
    context = {
        "user_profile": user_profile or {},
        "matched_foods_from_dataset": [
            {
                "name": r.get("name"),
                "calories": r.get("calories"),
                "protein": r.get("protein"),
                "carbs": r.get("carbs"),
                "fats": r.get("fats"),
                "diet": r.get("diet", []),
                "diseases": r.get("diseases", []),
                "allergy": r.get("allergy", []),
                "ingredients": r.get("ingredients", [])[:10],
            }
            for r in matches
        ],
        "user_question": question,
    }

    system_prompt = (
        "You are a nutrition assistant for a meal recommendation project. "
        "Answer ONLY using the provided local recipe dataset context. "
        "Do not invent meals, values, or medical claims. "
        "If the question is broad, suggest the best available meals from the dataset."
    )

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "openai/gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
                ],
            },
            timeout=60,
        )
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
    except Exception:
        pass

    return local_chat_answer(recipes, question, user_profile)


@st.cache_resource
def load_system() -> MealRecommendationSystem:
    model_path = get_model_path()
    return MealRecommendationSystem(model_path=model_path, recipes_path=str(RECIPES_PATH))


@st.cache_resource
def load_recipes_only() -> List[Dict[str, Any]]:
    recommender = RecipeRecommender(str(RECIPES_PATH))
    return recommender.recipes


def build_user_profile(
    age: int,
    gender: str,
    height_cm: float,
    weight_kg: float,
    activity_level: str,
    fitness_goal: str,
    dietary_preference: str,
    allergies: List[str],
    health_conditions: List[str],
    meals_per_day: int,
    max_calories: Optional[float],
    max_protein: Optional[float],
    max_carbs: Optional[float],
    max_fats: Optional[float],
) -> UserProfile:
    return UserProfile(
        age=int(age),
        gender=gender,
        height_cm=float(height_cm),
        weight_kg=float(weight_kg),
        activity_level=activity_level,
        fitness_goal=fitness_goal,
        dietary_preference=dietary_preference,
        allergies=allergies,
        health_conditions=health_conditions,
        meals_per_day=int(meals_per_day),
        max_calories=max_calories,
        max_protein=max_protein,
        max_carbs=max_carbs,
        max_fats=max_fats,
    )


st.set_page_config(page_title="Healthy Meal Recommender", page_icon="🥗", layout="wide")
st.title("🥗 Healthy Meal Recommendation System")

system = load_system()
recipes = load_recipes_only()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "latest_user_profile_dict" not in st.session_state:
    st.session_state.latest_user_profile_dict = {}

tab1, tab2, tab3 = st.tabs(["🎯 Recommendations", "🤖 Chatbot", "📊 Model Evaluation"])

with tab1:
    st.subheader("User Profile")

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", min_value=10, max_value=100, value=25, step=1)
        gender = st.selectbox("Gender", ["male", "female"])
        height = st.number_input("Height (cm)", min_value=120.0, max_value=230.0, value=170.0, step=0.5)
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=70.0, step=0.5)

    with c2:
        activity_level = st.selectbox("Activity Level", ACTIVITY_LEVELS)
        goal = st.selectbox("Fitness Goal", FITNESS_GOALS)
        diet_type = st.selectbox("Dietary Preference", DIETARY_PREFERENCES)

    with c3:
        meals_per_day = st.selectbox("Meals Per Day", [2, 3], index=1)
        allergies = st.multiselect("Allergies", ALLERGIES, default=["None"])
        health_conditions = st.multiselect("Health Conditions", HEALTH_CONDITIONS, default=["None"])

    with st.expander("Nutrition Limits", expanded=True):
        st.caption("These values are daily upper bounds and are also applied inside the model before meal scoring.")
        lc1, lc2, lc3, lc4 = st.columns(4)
        with lc1:
            max_calories = st.slider("Calories", min_value=CALORIES_RANGE[0], max_value=CALORIES_RANGE[1], value=CALORIES_RANGE[1], step=10)
        with lc2:
            max_protein = st.slider("Proteins", min_value=PROTEIN_RANGE[0], max_value=PROTEIN_RANGE[1], value=PROTEIN_RANGE[1], step=5)
        with lc3:
            max_carbs = st.slider("Carbs", min_value=CARBS_RANGE[0], max_value=CARBS_RANGE[1], value=CARBS_RANGE[1], step=5)
        with lc4:
            max_fats = st.slider("Fats", min_value=FATS_RANGE[0], max_value=FATS_RANGE[1], value=FATS_RANGE[1], step=5)

    user = build_user_profile(
        age=age,
        gender=gender,
        height_cm=height,
        weight_kg=weight,
        activity_level=activity_level,
        fitness_goal=goal,
        dietary_preference=diet_type,
        allergies=allergies,
        health_conditions=health_conditions,
        meals_per_day=meals_per_day,
        max_calories=float(max_calories),
        max_protein=float(max_protein),
        max_carbs=float(max_carbs),
        max_fats=float(max_fats),
    )

    st.session_state.latest_user_profile_dict = {
        "age": age,
        "gender": gender,
        "height_cm": height,
        "weight_kg": weight,
        "activity_level": activity_level,
        "fitness_goal": goal,
        "dietary_preference": diet_type,
        "allergies": allergies,
        "health_conditions": health_conditions,
        "meals_per_day": meals_per_day,
    }

    if st.button("Get Recommendations", type="primary", use_container_width=True):
        result = system.recommend(user, top_k=10)

        st.subheader("Your Nutrition Targets")
        st.caption("Predicted by the ML model based on user profile")

        daily_targets = result["daily_targets"]
        predicted_daily_targets = result.get("predicted_daily_targets", daily_targets)
        meal_target = result["meal_target"]

        with st.container(border=True):
            st.markdown("### Daily Targets")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Calories", f"{daily_targets['calories']:.0f} kcal", delta=f"Predicted {predicted_daily_targets['calories']:.0f}")
            c2.metric("Protein", f"{daily_targets['protein']:.0f} g", delta=f"Predicted {predicted_daily_targets['protein']:.0f}")
            c3.metric("Carbs", f"{daily_targets['carbs']:.0f} g", delta=f"Predicted {predicted_daily_targets['carbs']:.0f}")
            c4.metric("Fats", f"{daily_targets['fats']:.0f} g", delta=f"Predicted {predicted_daily_targets['fats']:.0f}")
            st.caption("Displayed daily targets already respect your selected nutrition limits.")

        with st.container(border=True):
            st.markdown("### Meal Target")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Calories", f"{meal_target['calories']:.0f} kcal")
            c2.metric("Protein", f"{meal_target['protein']:.0f} g")
            c3.metric("Carbs", f"{meal_target['carbs']:.0f} g")
            c4.metric("Fats", f"{meal_target['fats']:.0f} g")
            st.caption(f"Calculated by dividing daily targets by {meals_per_day} meals")

        st.subheader("Top Recommended Meals")
        for idx, meal in enumerate(result["recommendations"], start=1):
            with st.container(border=True):
                left, right = st.columns([1, 2], gap="large")
                with left:
                    if meal.get("image"):
                        st.image(meal["image"], width="stretch")
                    st.caption(f"Score: {meal.get('score', 0)}")
                with right:
                    st.markdown(f"### {idx}. {meal['name']}")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Calories", f"{meal.get('calories', 0)}")
                    m2.metric("Protein", f"{meal.get('protein', 0)} g")
                    m3.metric("Carbs", f"{meal.get('carbs', 0)} g")
                    m4.metric("Fats", f"{meal.get('fats', 0)} g")

                    e1, e2, e3 = st.columns(3)
                    e1.metric("Price", f"{meal.get('price', 0):.0f} EGP")
                    e2.metric("Discount", f"{meal.get('discount', 0):.0f}%")
                    e3.metric("Time", f"{meal.get('time', 0):.0f} min")

                    with st.expander("Ingredients"):
                        for ing in meal.get("ingredients", []):
                            st.write(f"- {ing}")

        st.success("✅ Recommendations generated based on your profile!")

with tab2:
    st.subheader("Chat with your recipe data")
    st.caption("The chatbot always searches directly in `recips.json`, with no dependency on recommendations.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask about healthy meals, diabetes, high protein, low carb...")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        answer = openrouter_answer(
            recipes=recipes,
            question=prompt,
            user_profile=st.session_state.latest_user_profile_dict,
        )

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)


EVAL_RESULTS_PATH = BASE_DIR / "model_evaluation_results.json"
EVAL_SCRIPT_PATH = BASE_DIR / "evaluate_models.py"


def _load_eval_results():
    if not EVAL_RESULTS_PATH.exists():
        return None
    with open(EVAL_RESULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


with tab3:
    st.subheader("📊 Model Evaluation & Comparison")
    st.caption("5-fold cross-validation on nutrition_dataset.csv")

    if st.button("🔄 Run Evaluation Now", type="primary"):
        with st.spinner("Training & evaluating models (this may take a minute) ..."):
            result = subprocess.run(
                ["python", str(EVAL_SCRIPT_PATH)],
                capture_output=True, text=True, cwd=str(BASE_DIR),
            )
            if result.returncode == 0:
                st.success("✅ Evaluation complete! Results refreshed.")
                st.rerun()
            else:
                st.error(f"Evaluation failed:\n```\n{result.stderr}\n```")

    eval_data = _load_eval_results()

    if eval_data is None:
        st.info("No evaluation results found. Click **Run Evaluation Now** to generate them.")
    else:
        best_name = eval_data.get("best_model", "")
        models = eval_data.get("models", [])

        st.markdown(f"### 🏆 Best Model: **{best_name.replace('_', ' ').title()}**")

        for model_info in models:
            name = model_info["model_name"]
            is_best = name == best_name
            label = f"{'🏆 ' if is_best else ''}{name.replace('_', ' ').title()}"
            avg_r2 = model_info["average_r2"]

            with st.container(border=True):
                st.markdown(f"#### {label}  —  Average R² = `{avg_r2}`")

                c1, c2, c3, c4 = st.columns(4)
                targets = model_info["targets"]

                for col, (target_name, vals) in zip([c1, c2, c3, c4], targets.items()):
                    with col:
                        st.metric(target_name.capitalize(), f"R² {vals['R2']}")
                        st.caption(f"MAE {vals['MAE']}  |  RMSE {vals['RMSE']}")

        st.markdown("### 📋 Comparison Table")

        table_rows = []
        for model_info in models:
            name = model_info["model_name"].replace("_", " ").title()
            for target_name, vals in model_info["targets"].items():
                table_rows.append({
                    "Model": name,
                    "Target": target_name.capitalize(),
                    "MAE": vals["MAE"],
                    "RMSE": vals["RMSE"],
                    "R²": vals["R2"],
                })

        import pandas as pd
        df_compare = pd.DataFrame(table_rows)
        st.dataframe(df_compare, use_container_width=True, hide_index=True)

        st.markdown("### 📊 R² Score Comparison")

        chart_data = []
        for model_info in models:
            name = model_info["model_name"].replace("_", " ").title()
            for target_name, vals in model_info["targets"].items():
                chart_data.append({
                    "Model": name,
                    "Target": target_name.capitalize(),
                    "R²": vals["R2"],
                })

        df_chart = pd.DataFrame(chart_data)
        pivot = df_chart.pivot(index="Target", columns="Model", values="R²")
        st.bar_chart(pivot)
