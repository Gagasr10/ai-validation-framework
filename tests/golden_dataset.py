"""
golden_dataset.py

Single source of truth for the gold-standard recipe dataset.
Imported by test_accuracy.py, test_llm_judge.py, and conftest.py fixtures.
"""

GOLD_DATA: list[tuple[list[str], str]] = [
    (["eggs", "flour", "milk"],               "Pancakes"),
    (["potato", "salt", "oil"],               "French fries"),
    (["chicken fillet", "salt", "pepper"],    "Chicken fillet"),
    (["tomato", "cheese", "basil"],           "Caprese salad"),
    (["pasta", "tomato sauce", "garlic"],     "Pasta with tomato sauce"),
    (["rice", "chicken", "broccoli"],         "Chicken and rice bowl"),
    (["flour", "sugar", "butter", "eggs"],    "Cake"),
    (["tuna", "mayonnaise", "corn"],          "Tuna salad"),
    (["spinach", "eggs", "feta cheese"],      "Spinach omelette"),
    (["shrimp", "garlic", "lemon", "butter"], "Garlic butter shrimp"),
]
