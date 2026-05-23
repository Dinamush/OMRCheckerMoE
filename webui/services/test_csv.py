"""Server-side test CSV generation for the Generate Test CSV page."""

from __future__ import annotations

import csv
from pathlib import Path

FIRST_NAMES = [
    "Aaron", "Abigail", "Adam", "Adrian", "Aisha", "Alex", "Alicia", "Aliyah",
    "Amanda", "Amber", "Amelia", "Andre", "Andrew", "Angela", "Ann", "Anthony",
    "Ashley", "Ayesha", "Barbara", "Benjamin", "Brandon", "Brianna", "Caleb",
    "Cameron", "Carlos", "Carmen", "Chantel", "Chelsea", "Christian",
    "Christine", "Christopher", "Cindy", "Claire", "Clayton", "Cody", "Crystal",
    "Damian", "Daniel", "Danielle", "David", "Diana", "Dominic", "Dylan",
    "Eduardo", "Elena", "Elizabeth", "Emily", "Emma", "Eric", "Ethan", "Faith",
    "Fatima", "Fernando", "Frank", "Gabriel", "George", "Grace", "Hannah",
    "Hector", "Henry", "Imani", "Isaiah", "Jacob", "Jade", "James", "Janet",
    "Jasmine", "Jason", "Jennifer", "Jessica", "Joel", "Jonathan", "Jordan",
    "Jose", "Joshua", "Juan", "Julian", "Kayla", "Kevin", "Kiran", "Kyle",
    "Laura", "Lauren", "Leah", "Leonardo", "Leslie", "Liam", "Lisa", "Logan",
    "Lucas", "Luis", "Madison", "Marcus", "Maria", "Mason", "Matthew", "Maya",
    "Mia", "Michael", "Michelle", "Miguel", "Mira", "Nathan", "Natalie",
    "Nicholas", "Nicole", "Noah", "Nora", "Olivia", "Omar", "Patrick", "Paul",
    "Peter", "Rachel", "Rebecca", "Richard", "Robert", "Ryan", "Samantha",
    "Samuel", "Sandra", "Sara", "Sarah", "Shawn", "Sofia", "Stephanie",
    "Steven", "Susan", "Taylor", "Thomas", "Timothy", "Tyler", "Victoria",
    "Vincent", "Whitney", "William", "Yasmine", "Zachary", "Zoe", "Zara",
]

LAST_NAMES = [
    "Adams", "Alexander", "Allen", "Anderson", "Andrews", "Archer", "Armstrong",
    "Atkins", "Austin", "Bailey", "Baker", "Banks", "Barnes", "Bell", "Bennett",
    "Bishop", "Black", "Blake", "Boyd", "Brooks", "Brown", "Bryan", "Burke",
    "Burns", "Butler", "Campbell", "Carr", "Carter", "Chambers", "Chapman",
    "Charles", "Clarke", "Coleman", "Collins", "Cook", "Cooper", "Cox",
    "Crawford", "Cruz", "Davis", "Dean", "Dixon", "Edwards", "Ellis", "Evans",
    "Ferguson", "Fisher", "Fleming", "Fletcher", "Ford", "Foster", "Fox",
    "Francis", "Fraser", "Freeman", "Garcia", "Gibson", "Gill", "Gordon",
    "Graham", "Grant", "Gray", "Green", "Griffin", "Hall", "Hamilton",
    "Harris", "Harrison", "Hart", "Harvey", "Hayes", "Henderson", "Henry",
    "Hill", "Holmes", "Howard", "Hughes", "Hunter", "Jackson", "James",
    "Jenkins", "Johnson", "Jones", "Joseph", "Kelly", "Kennedy", "King",
    "Knight", "Lambert", "Lawrence", "Lewis", "Long", "Lopez", "Martin",
    "Martinez", "Mason", "Mathews", "Mitchell", "Moore", "Morgan", "Morris",
    "Morrison", "Murray", "Nelson", "Newton", "Nichols", "Noel", "Oliver",
    "Owens", "Palmer", "Parker", "Patterson", "Payne", "Perry", "Peters",
    "Phillips", "Pierre", "Porter", "Powell", "Price", "Ramkissoon", "Reid",
    "Richards", "Richardson", "Roberts", "Robinson", "Rogers", "Ross",
    "Russell", "Sanchez", "Sanders", "Scott", "Shaw", "Singh", "Smith",
    "Spencer", "Stewart", "Sullivan", "Taylor", "Thomas", "Thompson", "Torres",
    "Turner", "Walker", "Ward", "Watson", "White", "Williams", "Wilson", "Wood",
]

NAME_COMBINATIONS = len(FIRST_NAMES) * len(LAST_NAMES)
NAME_STEP = 9973


def hash_string(value: str) -> int:
    """Return the same FNV-1a style unsigned 32-bit hash used by the JS preview."""
    value_hash = 2166136261
    for char in value:
        value_hash ^= ord(char)
        value_hash = (value_hash * 16777619) & 0xFFFFFFFF
    return value_hash


def realistic_name(row_index: int, seed: int) -> str:
    """Return a deterministic, random-looking name for a one-based row index."""
    zero_index = row_index - 1
    combo = (seed + (zero_index * NAME_STEP)) % NAME_COMBINATIONS
    first = FIRST_NAMES[combo % len(FIRST_NAMES)]
    last = LAST_NAMES[(combo // len(FIRST_NAMES)) % len(LAST_NAMES)]
    if zero_index < NAME_COMBINATIONS:
        return f"{first} {last}"
    return f"{first} {last} {(zero_index // NAME_COMBINATIONS) + 1}"


def row_values(
    row_index: int,
    school_name: str,
    exam_name: str,
    candidate_start: int,
    name_style: str,
    name_seed: int,
    include_output_file: bool = False,
) -> list[str]:
    """Return one CSV row using the same deterministic naming as the preview.

    The optional ``output_file`` column is omitted by default because the
    prefill PDF mode does not use it; only the ZIP-of-PNGs mode honours it.
    """
    if name_style == "random":
        name = realistic_name(row_index, name_seed)
    else:
        name = f"Student {row_index}"
    candidate_number = f"{candidate_start + row_index - 1:010d}"
    row = [name, school_name, exam_name, candidate_number]
    if include_output_file:
        slug = "_".join(part for part in "".join(
            char.lower() if char.isalnum() else " " for char in name
        ).split())
        row.append(f"{slug}.png")
    return row


def write_test_csv(
    *,
    dst_path: Path,
    count: int,
    school_name: str,
    exam_name: str,
    candidate_start: str,
    name_style: str,
    include_output_file: bool = False,
) -> dict[str, int]:
    """Write a generated student CSV to *dst_path* without buffering all rows."""
    candidate_start_int = int(candidate_start)
    name_seed = hash_string(f"{school_name}|{exam_name}|{candidate_start}")
    headers = [
        "student_name",
        "school_name",
        "exam_name",
        "candidate_number",
    ]
    if include_output_file:
        headers.append("output_file")
    with dst_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row_index in range(1, count + 1):
            writer.writerow(row_values(
                row_index,
                school_name,
                exam_name,
                candidate_start_int,
                name_style,
                name_seed,
                include_output_file,
            ))
    return {"count": count, "size_bytes": dst_path.stat().st_size}
