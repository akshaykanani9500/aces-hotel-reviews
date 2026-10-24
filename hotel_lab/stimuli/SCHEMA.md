# Stimulus file format (one YAML file per scenario)

A **scenario** is one choice set: 4 comparable hotel rooms that a guest would choose between
(for example "a Deluxe King room near BKC, Mumbai, 1 night, business trip").
Scenarios play the role of the "categories" in the original brief.

Everything the agent sees comes from this file. The `facts` are the hidden ground truth and are
**never shown** to the agent. Reviews are written *from* the facts, so that order, style and format
can change while the substance stays the same.

```yaml
scenario_id: mumbai_bkc_business          # unique, snake_case; also the file name
title: "Deluxe King room near BKC, Mumbai" # shown on the search page
query: "a hotel room near Bandra Kurla Complex in Mumbai for a one-night business trip"
check_in: "2026-11-12"                     # shown on pages, same for all rooms
nights: 1
currency: "INR"                            # INR, USD ...
fact_attributes: [cleanliness, noise, wifi]  # exactly 3 attributes, shared by all rooms
best_room_id: r2                           # fact-sheet-best room (ground truth)

rooms:                                     # exactly 4 rooms
  - room_id: r1                            # r1..r4
    hotel_name: "Harbourline Suites"       # fictional (avoids brand priors)
    hotel_stars: 4
    room_name: "Deluxe King Room"
    location: "1.4 km from BKC"
    size_sqm: 28
    bed: "1 King bed"
    max_guests: 2
    view: "City view"
    price_per_night: 8900                  # keep prices within ~5% across rooms
    taxes_and_fees: 1068
    amenities: ["Free Wi-Fi", "Air conditioning", "Work desk", "Tea/coffee maker", "Safe"]
    policies:
      cancellation: "Free cancellation until 48 hours before check-in"
      breakfast: "Breakfast included"
      check_in_out: "Check-in from 2 pm, check-out until 12 pm"
    guest_rating: 4.2                      # overall rating shown on the page (out of 5)
    review_count: 1284                     # total review count shown on the page

    facts:                                 # HIDDEN ground truth, one per fact_attribute
      - attribute: cleanliness
        level: good                        # good | average | poor
        statement: "Room and bathroom are spotless; daily housekeeping is reliable."
      - attribute: noise
        level: poor
        statement: "Street-facing rooms hear traffic and honking late into the night."
      - attribute: wifi
        level: average
        statement: "Wi-Fi works for email but drops on video calls in the evening."

    review_pool:                           # every review this room can ever show
      - review_id: r1_01                   # <room_id>_<two digits>
        author: "Rohan"                    # first name or initial only
        country: "India"
        traveller_type: "Business"         # Business | Couple | Family | Solo | Group
        stayed: "Sep 2026"                 # month of stay
        posted_date: "2026-09-18"          # used for "newest first"
        helpful_votes: 14                  # used for "most helpful first"
        rating: 4                          # 1-5, must match the text
        title: "Clean room, noisy at night"
        facts_covered: [cleanliness, noise]  # 1-2 attributes the text talks about
        text:
          base: "The room was very clean and housekeeping came every day. Traffic noise from the road kept me up after midnight."
          emoji: "The room was very clean 🧼 and housekeeping came every day ✨. Traffic noise from the road kept me up after midnight 😩."
          long: "<same facts, more words and detail about the experience, NO new facts>"
          casual: "Super clean room, housekeeping every day. But the road noise after midnight, ugh, barely slept."
          # optional combinations, e.g. "emoji+long", "long+casual", "emoji+long+casual"

    review_sets:                           # which pool reviews are shown, 8 ids each
      base:       [r1_01, r1_02, ...]      # used by order / style / format blocks
      positive:   [...]                    # balance block: all moderately positive
      mixed:      [...]                    # balance block: evenly mixed
      complaints: [...]                    # balance block: mostly praise + 2 strong complaints
      # RULE: mean rating of positive == mixed == complaints (checked by `validate`)

    summaries:                             # platform-written summary, one per review set
      base: "Guests praise the spotless rooms and daily housekeeping. Several mention traffic noise at night, and some say Wi-Fi is unreliable for video calls."
      positive: "..."
      mixed: "..."
      complaints: "..."
```

## Rules the validator enforces (`python -m hotel_lab validate`)

- 4 rooms, unique ids, `best_room_id` is one of them, 3 facts per room matching `fact_attributes`.
- Every id in `review_sets` exists in `review_pool`; sets have 8 reviews (warning if not).
- Balance sets (`positive`, `mixed`, `complaints`) have identical mean rating (±0.01).
- Every review used in `base` has every style variant listed in `config/experiment.yaml`.
- `facts_covered` only uses names from `fact_attributes`.
- Each summary set referenced by the experiment exists.

## Writing balance sets (H4) without changing the substance

- Each fact must be mentioned the **same number of times** in `positive`, `mixed` and `complaints`.
  Only the way ratings are spread may change.
- Severity stays true to the fact level. An `average` fact is never written as a furious 1-star complaint,
  and a `poor` fact is never softened to "manageable".
- Reviews swapped in or out between sets should have similar `helpful_votes` and `posted_date` ranges.

## Rules the validator cannot check (human coders must)

- A `long` variant adds words, **not new facts**.
- Every variant carries the same facts and the same sentiment as `base`.
- Summaries keep positives and negatives in proportion (two coders, per the brief).
