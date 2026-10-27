"""Gemini-powered comprehensive food safety, shelf-life, and biological risk assessment service.

Provides in-depth post-harvest physiology insights, clinical toxicity warnings,
spoilage reactions, and customized storage handling guidelines.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Structured botanical and toxicological knowledge base for local fallback
PRODUCE_KNOWLEDGE = {
    "apple": {
        "scientific": "Malus domestica",
        "stale_pathogens": "Penicillium expansum (blue mold), Botrytis cinerea, Alternaria alternata",
        "toxin": "Patulin (a dangerous cytotoxic mycotoxin that penetrates deep into apple pulp)",
        "reactions": "Ingesting patulin-contaminated apple causes acute nausea, mucosal ulceration, severe gastrointestinal irritation, vomiting, and potential immunological/renal stress.",
        "storage": "Store at 0°C to 4°C with 90-95% relative humidity. Isolate from ethylene-sensitive vegetables.",
        "shelf_fresh": "14 to 28 days (refrigerated) / 5 to 7 days (room temp)",
        "shelf_stale": "0 days — Discard immediately if soft rot, mold patches, or fermentation smell is present."
    },
    "banana": {
        "scientific": "Musa acuminata",
        "stale_pathogens": "Colletotrichum musae (anthracnose), Fusarium oxysporum, Lasiodiplodia theobromae",
        "toxin": "Fusaric acid and secondary fungal metabolites; high biogenic amines (tyramine/histamine)",
        "reactions": "Consuming over-rotted bananas with pulp liquefaction can trigger histaminic headaches, abdominal colic, diarrhea, and allergen exacerbations.",
        "storage": "Store unpeeled at 13°C to 15°C with 85-90% RH. Never refrigerate below 11°C to avoid chilling injury.",
        "shelf_fresh": "5 to 7 days at room temperature / 2 to 3 days once fully ripe",
        "shelf_stale": "0 days if mold penetrates rind or pulp smells fermented. Brown-speckled peels are safe for baking if pulp is firm."
    },
    "orange": {
        "scientific": "Citrus sinensis",
        "stale_pathogens": "Penicillium digitatum (green mold), Penicillium italicum (blue mold), Geotrichum citri-aurantii",
        "toxin": "Citrus fungal mycotoxins, volatile terpenoid spoilage compounds",
        "reactions": "Inhalation of mold dust during peeling can provoke asthma attacks and allergic rhinitis; ingestion causes bitter-spoilage gastroenteritis, acid reflux, and mucosal inflammation.",
        "storage": "Refrigerate at 4°C to 7°C with 85-90% RH. Ensure adequate dry air circulation to prevent mold bloom.",
        "shelf_fresh": "14 to 21 days (refrigerated) / 7 to 10 days (ambient)",
        "shelf_stale": "0 days — Discard immediately upon appearance of white/blue/green fungal powder."
    },
    "tomato": {
        "scientific": "Solanum lycopersicum",
        "stale_pathogens": "Alternaria solani, Rhizopus stolonifer, Erwinia carotovora, Salmonella enterica serovars",
        "toxin": "Alternariol (AOH), alternariol monomethyl ether (AME), bacterial endotoxins",
        "reactions": "Cracked, watery, or moldy tomatoes harbor invasive bacterial enterotoxins capable of severe food poisoning (Salmonellosis), profuse vomiting, fever, and enteritis.",
        "storage": "Keep stem-up at 12°C to 15°C with 85% RH. Avoid prolonged refrigeration under 10°C to preserve flavor volatiles.",
        "shelf_fresh": "5 to 9 days at room temperature / 10 to 14 days in mild cool cellar",
        "shelf_stale": "0 days — Leakage, bacterial slip-skin, or surface fuzz warrants complete disposal."
    },
    "capsicum": {
        "scientific": "Capsicum annuum",
        "stale_pathogens": "Botrytis cinerea, Alternaria alternata, Erwinia soft rot",
        "toxin": "Fungal mycotoxins and bacterial cell-wall maceration enzymes",
        "reactions": "Consuming slimy or blackened bell peppers can trigger toxic dyspepsia, bacterial colitis, and intestinal distress.",
        "storage": "Store in crisper drawer at 7°C to 9°C, 90-95% RH. Keep dry; moisture droplets catalyze rapid decay.",
        "shelf_fresh": "10 to 14 days (refrigerated)",
        "shelf_stale": "0 days — Internal seed cavity mold is toxic even if exterior skin appears semi-firm."
    },
    "bitter gourd": {
        "scientific": "Momordica charantia",
        "stale_pathogens": "Fusarium pallidoroseum, Colletotrichum gloeosporioides, soft-rot pectinolytic bacteria",
        "toxin": "Degraded momordicines, toxic glycosides from rotten pith, bacterial endotoxins",
        "reactions": "Excessively decayed bitter gourd exhibits degraded bitter cucurbitacins causing toxic squash syndrome: harsh stomach cramps, mucosal nausea, and toxic diarrhea.",
        "storage": "Wrap in perforated paper and store at 8°C to 10°C. Protect from moisture buildup.",
        "shelf_fresh": "4 to 6 days under refrigeration",
        "shelf_stale": "0 days if yellowish-orange breakdown or inner spongy mushiness is evident."
    },
    "default": {
        "scientific": "Horticultural Produce",
        "stale_pathogens": "Botrytis cinerea, Rhizopus, Alternaria, and pectinolytic Pseudomonas spp.",
        "toxin": "General mycotoxins, histamine, and microbial metabolic byproducts",
        "reactions": "Ingestion of spoiled produce introduces fungal spores and bacterial colonies, frequently leading to acute bacterial gastroenteritis, stomach cramps, nausea, vomiting, and systemic foodborne illness.",
        "storage": "Maintain controlled cold storage between 4°C and 8°C with moderated 85% relative humidity.",
        "shelf_fresh": "5 to 10 days depending on cold chain continuity",
        "shelf_stale": "0 days — Unfit for human consumption; send to compost."
    }
}


def _match_knowledge(produce_name: str) -> dict:
    name_clean = str(produce_name).lower().replace("_", " ").strip()
    for key, data in PRODUCE_KNOWLEDGE.items():
        if key in name_clean:
            return data
    return PRODUCE_KNOWLEDGE["default"]


def generate_gemini_assessment(
    produce_name: str,
    status: str,
    confidence: float,
    freshness_score: float,
    shelf_life_days: float | None,
    temperature: float,
    humidity: float,
    gemini_api_key: str | None = None,
) -> dict:
    """Generate an authoritative post-harvest quality, shelf-life, and biological toxicity report.

    Attempts Google Gemini API (gemini-2.5-flash / gemini-2.0-flash / gemini-1.5-flash),
    falling back to a scientifically rigorous botanical agro-safety engine (100% free).
    """
    api_key = gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    is_fresh = str(status).strip().lower() == "fresh"
    kb = _match_knowledge(produce_name)

    # 1. Try Google Gemini Client
    if api_key:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)

            prompt = f"""You are a Senior Post-Harvest Physiologist and Food Safety Toxicologist.
Analyze the following quality inspection result for a major project final-year evaluation report:

- Produce: {produce_name} ({kb['scientific']})
- Quality Classification: {status} (Freshness Score: {freshness_score:.1f}/100, Model Confidence: {confidence:.1f}%)
- Estimated Shelf Life: {shelf_life_days if shelf_life_days is not None else 'Calculated based on conditions'} days
- Ambient Storage Environment: Temperature {temperature:.1f}°C, Relative Humidity {humidity:.1f}%

Provide a comprehensive, authoritative evaluation in strict valid JSON format with EXACTLY these keys:
{{
  "verdict": "short verdict string (e.g. CERTIFIED FRESH & SAFE / RAPID CONSUMPTION ADVISED / HAZARDOUS - UNFIT FOR CONSUMPTION)",
  "usable_days_summary": "precise statement on remaining safe usable days under current vs optimal conditions",
  "safe_to_use": "YES / CONDITIONALLY / STRICTLY NO",
  "culinary_action": "concrete guidance (e.g., safe for raw consumption, trim & cook, vinegar processing, or immediate discard)",
  "biological_reactions_if_consumed": "detailed 2-3 paragraph medical and biochemical explanation of biological reactions, pathogenic colonization, toxins (e.g. {kb['toxin']}), gastrointestinal symptoms, vomiting/diarrhea, and risks to vulnerable groups if consumed in its current/stale state",
  "spoilage_pathogens": "list of primary fungi/bacteria active in this produce spoilage",
  "storage_protocol": "optimal temperature, humidity, airflow, and ethylene handling advice",
  "risk_severity": "LOW / MODERATE / HIGH / CRITICAL"
}}
Return ONLY raw JSON, without markdown formatting or backticks if possible."""

            # Try modern Gemini models
            for model_id in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
                try:
                    response = client.models.generate_content(
                        model=model_id,
                        contents=prompt,
                    )
                    text = response.text.strip()
                    # Clean markdown codeblocks if present
                    if text.startswith("```"):
                        text = re.sub(r"^```(?:json)?\s*", "", text)
                        text = re.sub(r"\s*```$", "", text)
                    parsed = json.loads(text)
                    parsed["source"] = f"Google Gemini ({model_id})"
                    return _enrich_report(parsed, produce_name, is_fresh, kb, shelf_life_days, temperature, humidity)
                except Exception:
                    continue
        except Exception:
            pass

    # 2. Scientific Botanical & Toxicological Rule Engine (100% Free, Instant, Offline Guaranteed Fallback)
    return _generate_scientific_fallback(produce_name, status, is_fresh, kb, freshness_score, shelf_life_days, temperature, humidity)


def _generate_scientific_fallback(
    produce_name: str,
    status: str,
    is_fresh: bool,
    kb: dict,
    freshness_score: float,
    shelf_life_days: float | None,
    temperature: float,
    humidity: float,
) -> dict:
    days = shelf_life_days if shelf_life_days is not None else (7.0 if is_fresh else 0.5)

    if is_fresh:
        verdict = "CERTIFIED FRESH — SAFE FOR COMMERCIAL & DIRECT CONSUMPTION"
        safe_to_use = "YES — PRIME CONDITION"
        risk_severity = "LOW"
        culinary_action = (
            f"Recommended for direct table consumption, fresh salad preparation, or commercial export packaging. "
            f"Rinse under potable water before ingestion. Cellular turgidity and nutrient profile are intact."
        )
        usable_days = (
            f"Estimated {days:.1f} days of high commercial quality remaining under current storage conditions "
            f"({temperature:.1f}°C, {humidity:.1f}% RH). Extending refrigeration to optimal range can extend life by 30-40%."
        )
        reactions = (
            f"The produce is structurally sound with intact cuticle barrier and no active microbial colonization. "
            f"Normal digestion expected; rich in dietary antioxidants, active dietary fiber, and essential micronutrients. "
            f"No toxic biological reactions or microbial pathogenic concerns detected under standard food-handling standards."
        )
    else:
        verdict = "HAZARDOUS / STALE — UNFIT FOR FRESH CONSUMPTION"
        safe_to_use = "STRICTLY NO — DO NOT CONSUME"
        risk_severity = "CRITICAL" if freshness_score < 30 else "HIGH"
        culinary_action = (
            f"Immediate segregation required to prevent airborne spore propagation to adjacent healthy stock. "
            f"DO NOT consume raw. Not recommended for cooking as heat does not destroy heat-stable mycotoxins like patulin. "
            f"Divert to municipal organic composting or bio-gas digestion."
        )
        usable_days = (
            f"0 days for safe consumption. Maximum {max(0.2, days):.1f} days before full liquefaction and structural collapse. "
            f"Current storage conditions ({temperature:.1f}°C, {humidity:.1f}% RH) accelerate pathogen germination."
        )
        reactions = (
            f"CRITICAL HEALTH HAZARD: Ingestion of deteriorated {produce_name} poses immediate gastrointestinal and toxicological risks.\n\n"
            f"1. **Toxic Biochemical Synthesis**: Spoilage is actively driven by {kb['stale_pathogens']}. "
            f"These organisms secrete hazardous secondary metabolites, notably {kb['toxin']}.\n\n"
            f"2. **Clinical Physiological Reactions**: Upon ingestion, fungal mycotoxins and bacterial endotoxins disrupt the gastric and intestinal epithelial mucosa. "
            f"Within 2 to 6 hours, patients may experience severe acute nausea, persistent vomiting, abdominal cramps, and watery or hemorrhagic diarrhea.\n\n"
            f"3. **Systemic & Immunological Vulnerability**: Inhaling volatile spore dust or ingesting necrotic tissue introduces mycotoxic stress on hepatic and renal filtration systems. "
            f"High-risk individuals (children, elderly, pregnant women, and immunocompromised patients) face elevated risk of severe electrolyte depletion and systemic intoxication."
        )

    parsed = {
        "verdict": verdict,
        "usable_days_summary": usable_days,
        "safe_to_use": safe_to_use,
        "culinary_action": culinary_action,
        "biological_reactions_if_consumed": reactions,
        "spoilage_pathogens": kb["stale_pathogens"],
        "storage_protocol": kb["storage"],
        "risk_severity": risk_severity,
        "source": "IntelliFresh Expert Agro-Safety Engine",
    }
    return _enrich_report(parsed, produce_name, is_fresh, kb, shelf_life_days, temperature, humidity)


def _enrich_report(report: dict, produce_name: str, is_fresh: bool, kb: dict, shelf_life: float | None, temp: float, hum: float) -> dict:
    report["produce_scientific"] = kb.get("scientific", "Horticultural Specimen")
    report["inspected_temperature"] = f"{temp:.1f}°C"
    report["inspected_humidity"] = f"{hum:.1f}%"
    report["safe_shelf_estimate"] = report.get("usable_days_summary", "")
    return report
