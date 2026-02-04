import re

description = "The scene is dimly lit. Snow covers much of the ground. Thick snow is piled on the right side. This snow is against a white door frame and an adjacent wall. A dark, ridged object is in the upper left. A striped mat lies in the foreground. The mat shows red, green, and beige stripes. Melting snow and ice cover parts of the mat. Dark debris is scattered across the snow and mat.\nWeather: 13.78 °F, Clouds"
gemini_lower = description.lower()
detected_classes = ['cat']

print(f"Description: {description}")
print(f"Detected classes: {detected_classes}")

# Old logic simulation
print("\n--- Old Logic ---")
gemini_confirmed_old = False
for cls in detected_classes:
    if cls.lower() in gemini_lower:
        gemini_confirmed_old = True
        print(f"MATCH: '{cls}' found in description (Substring match)")
        break
print(f"Confirmed: {gemini_confirmed_old}")

# New logic simulation
print("\n--- New Logic ---")
gemini_confirmed_new = False
for cls in detected_classes:
    pattern = r'\b' + re.escape(cls.lower()) + r'\b'
    if re.search(pattern, gemini_lower):
        gemini_confirmed_new = True
        print(f"MATCH: '{cls}' found in description (Word match)")
        break
    else:
        print(f"NO MATCH: '{cls}' NOT found as whole word")

print(f"Confirmed: {gemini_confirmed_new}")

# Test negation logic
print("\n--- Negation Logic Test ---")
negation_phrases = ["don't see", "do not see", "no ", "cannot see", "can't see", "is not visible", "not visible"]
has_negation_old = any(phrase in gemini_lower for phrase in negation_phrases)
print(f"Old Negation (substring 'no '): {has_negation_old}")

negation_patterns = [
    r"\bdon'?t see\b", 
    r"\bdo not see\b", 
    r"\bno\b", 
    r"\bcannot see\b", 
    r"\bcan'?t see\b", 
    r"\bis not visible\b", 
    r"\bnot visible\b"
]
has_negation_new = any(re.search(p, gemini_lower) for p in negation_patterns)
print(f"New Negation (regex '\\bno\\b'): {has_negation_new}")
