import csv
from models.venue import Venue

def is_duplicate_venue(venue_name: str, seen_names: set) -> bool:
    return venue_name in seen_names

def is_complete_venue(venue: dict, required_keys: list) -> bool:
    return all(key in venue for key in required_keys)

def save_venues_to_csv(venues: list, filename: str):
    if not venues:
        print("No venues to save.")
        return

    # Use the field names defined in the Venue model
    fieldnames = list(Venue.__annotations__.keys())  # Correct way to get field names

    # Ensure we’re writing the data according to the model's fields
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(venues)
    print(f"Saved {len(venues)} venues to '{filename}'.")
