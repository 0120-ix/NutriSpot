import json
import os
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Literal

from meal_model import (
    UserProfile,
    MealRecommendationSystem,
    ACTIVITY_LEVELS,
    FITNESS_GOALS,
    DIETARY_PREFERENCES,
    ALLERGIES,
    HEALTH_CONDITIONS
)
from chatbot_engine import FoodChatbot

app = FastAPI(
    title="Nutrition & Meal Recommendation API",
    description="API for predicting nutrition targets, recommending meals, and chatting with an AI assistant.",
    version="1.0.0"
)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants and paths
MODEL_PATH = "nutrition_model.joblib"
RECIPES_PATH = "recips.json"
DATASET_PATH = "nutrition_dataset.csv"

# Global instances initialized on startup
system: Optional[MealRecommendationSystem] = None
chatbot: Optional[FoodChatbot] = None

# --- Pydantic Models for Input Validation ---

class UserProfileInput(BaseModel):
    age: int = Field(..., description="Age in years")
    gender: Literal["Male", "Female"] = Field(..., description="Gender (Male / Female)")
    height_cm: float = Field(..., description="Height in cm")
    weight_kg: float = Field(..., description="Weight in kg")
    activity_level: Literal["Sedentary", "Light", "Moderate", "High"] = Field(..., description="Activity Level")
    fitness_goal: Literal["Lose weight", "Gain weight", "Improve health", "Maintain weight", "Build muscle"] = Field(..., description="Fitness Goal")
    dietary_preference: Literal["High protein", "Vegan", "Low carb", "Keto"] = Field(..., description="Dietary Preference")
    allergies: List[str] = Field(default=["None"], description="List of allergies, e.g., ['None'], ['Lactose'], ['Nuts']")
    health_conditions: List[str] = Field(default=["None"], description="List of conditions, e.g., ['None'], ['Diabetes']")
    meals_per_day: Literal[2, 3] = Field(3, description="Meals per day (2 or 3)")
    notes: str = Field("", description="Additional notes")
    max_calories: Optional[float] = None
    max_protein: Optional[float] = None
    max_carbs: Optional[float] = None
    max_fats: Optional[float] = None

    @model_validator(mode='before')
    @classmethod
    def clean_inputs(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for k in ['gender', 'activity_level', 'fitness_goal', 'dietary_preference']:
                if k in data and isinstance(data[k], str):
                    data[k] = data[k].capitalize()
        return data

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to the chatbot")
    user_profile: Optional[UserProfileInput] = None
    recommendations: Optional[List[Dict[str, Any]]] = Field(None, description="Previous meal recommendations context")

# --- App Startup ---

@app.on_event("startup")
def load_models():
    global system, chatbot
    
    # Initialize recommendation system
    if os.path.exists(MODEL_PATH) and os.path.exists(RECIPES_PATH):
        try:
            system = MealRecommendationSystem(model_path=MODEL_PATH, recipes_path=RECIPES_PATH)
            
            # Load raw recipes for the chatbot
            with open(RECIPES_PATH, "r", encoding="utf-8") as f:
                recipes_data = json.load(f)
            
            chatbot = FoodChatbot(foods_data=recipes_data, recommendation_engine=system)
            print("Models loaded successfully.")
        except Exception as e:
            print(f"Error loading models on startup: {e}")
    else:
        print("Warning: Model or Recipes file not found. Ensure models are trained first.")

# --- Endpoints ---

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Nutrition Prediction & Chat API is running. Check /docs for documentation."}

@app.post("/api/recommend-meals")
def get_recommendations(profile: UserProfileInput):
    """
    Generates predicted nutritional targets and recommends daily meals based on the user's profile.
    """
    if not system:
        raise HTTPException(status_code=503, detail="Recommendation system not initialized. Ensure models are trained and present.")
    
    try:
        # Convert pydantic model to the internal dataclass
        user = UserProfile(**profile.model_dump())
        
        # Get recommendations
        result = system.recommend(user=user, top_k=10)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
def chat_with_bot(req: ChatRequest):
    """
    Interact with the AI chatbot, which has context of recipes, the user's profile, and their recommended meals.
    """
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not initialized.")
    
    try:
        profile_dict = req.user_profile.model_dump() if req.user_profile else None
        
        response_text = chatbot.respond(
            user_msg=req.message,
            user_profile=profile_dict,
            recommendations=req.recommendations
        )
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))