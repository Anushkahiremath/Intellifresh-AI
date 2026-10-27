"""Google Gemini recommendation service with an authoritative local fallback.

100% free tool using Google Gemini API (gemini-2.5-flash / gemini-2.0-flash / gemini-1.5-flash).
"""
from __future__ import annotations

import json
import os
from dotenv import load_dotenv

load_dotenv()


def recommendation_for(
    result: dict,
    temperature: float,
    humidity: float,
    api_key: str | None = None,
) -> tuple[str, str]:
    fallback = _local_recommendation(result, temperature, humidity)
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        return fallback, "IntelliFresh Scientific Rules (Add free Gemini key for AI advice)"

    try:
        from google import genai
        client = genai.Client(api_key=key)
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        prompt = (
            "You are a post-harvest produce quality assistant. Give a concise response with two labelled parts:\n"
            "Reason: explain how the detected fruit condition, temperature, humidity, and estimated shelf life affect freshness and bacterial/mold risk.\n"
            "Recommendation: give practical handling, storage, and consumption steps (e.g. ideal temperature, ethylene isolation, culinary use or disposal). Use no more than 85 words.\n"
            f"Produce Data: Fruit: {result.get('fruit', result.get('produce_name'))}, Quality: {result.get('status')}, "
            f"Freshness: {result.get('freshness_score')}%, Shelf Life: {result.get('shelf_life_days')} days, "
            f"Temperature: {temperature}°C, Relative Humidity: {humidity}%"
        )

        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        return response.text.strip(), f"Google Gemini ({model})"
    except Exception:
        # Try fallback model if modern flash had an issue
        try:
            from google import genai
            client = genai.Client(api_key=key)
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt,
            )
            return response.text.strip(), "Google Gemini (gemini-1.5-flash)"
        except Exception:
            return fallback, "Local Fallback (Free Expert Rule Engine)"


def _local_recommendation(result: dict, temperature: float, humidity: float) -> str:
    days = result.get("shelf_life_days")
    status = str(result.get("status", "")).lower()
    fruit = str(result.get("fruit", result.get("produce_name", "Produce"))).title()

    if "stale" in status or "rotten" in status:
        return (
            f"Reason: Structural breakdown and high microbial spore activity detected for {fruit}. "
            f"At {temperature:.1f}°C and {humidity:.1f}% RH, pathogen proliferation accelerates rapidly.\n"
            f"Recommendation: Unfit for direct consumption. Immediately isolate from healthy produce "
            f"to prevent airborne spore contamination and discard or divert to compost."
        )

    urgency = "Consume or process within 1-2 days. " if days is not None and days < 3 else "Prime condition. "
    environment = "Maintain refrigeration at 2°C–6°C" if temperature > 12 else "Current cool storage is effective"
    if humidity > 88:
        environment += " with improved airflow to prevent moisture droplets"
    elif humidity < 55:
        environment += " with breathable wrapping to prevent water loss and shriveling"
    return f"Reason: {urgency}Cellular turgidity is intact.\nRecommendation: {environment}. Wash thoroughly before consumption."
