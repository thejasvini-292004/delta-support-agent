# Delta — Proposed Intent Taxonomy (v1, discovered from data)

**How this was derived (not invented):** clustered 44,294 cleaned Delta customer
messages with TF-IDF + KMeans (`src/discover_intents.py`), read each cluster's top
terms + representative examples, then named a small taxonomy and refined it with
transparent keyword rules (`src/make_golden_candidates.py`).

## The single most important finding
**~59% of inbound @Delta messages are NOT support requests** — they are praise,
thanks, jokes, travel photos, and tagging. Design consequences:
- You need an explicit **`other_non_actionable`** class; it may be the *largest* class.
- Your agent must **not** fabricate a "solution" for these — most should be handled
  with a short acknowledgement or **escalated / dropped**, never auto-"resolved."
- Report **macro-F1 + per-intent**, never plain accuracy (a model that only nails the
  big class looks great and is useless on the intents that matter).

## The taxonomy (7 actionable + 1 non-actionable)
Volumes are rough weak-label estimates, for stratification only — not ground truth.

| Intent | Definition | Est. share | Default decision |
|---|---|---|---|
| `flight_disruption` | Delay, cancellation, missed/at-risk connection, diversion, stranded | ~9% | often **escalate** (time-critical, needs live data) |
| `seating_upgrade` | Seat selection/assignment, upgrades, Comfort+, cabin/class | ~8% | auto if info-only, else escalate |
| `loyalty_account` | SkyMiles, Medallion status, points, SkyClub lounge access | ~6% | auto (mostly FAQ) |
| `baggage` | Lost/delayed/damaged bags, baggage fees, checked/carry-on rules | ~6% | escalate if lost/damaged; auto for policy Qs |
| `airport_gate` | Gate, boarding, terminal, check-in counter, TSA, kiosk | ~5% | auto (info), escalate if time-critical |
| `booking_change` | Change/cancel/rebook, standby, fare rules, refunds, itinerary | ~4% | **escalate** (transactional, account-specific) |
| `tech_support` | App/website/wifi/online check-in not working, login/errors | ~3% | auto (known workarounds) |
| `other_non_actionable` | Praise, thanks, jokes, tagging, venting with no ask | ~59% | **do not auto-answer**; acknowledge or route |

> Note: a *complaint with a real request* is labelled by its underlying intent (e.g. an
> angry "my bag is lost" is `baggage`) and escalated via the risk rule — not dumped in `other`.

## How to label the golden set (Step 2)
Open `data/golden_to_label.csv` (220 stratified candidates). For each row:
1. **`intent`** — correct the `suggested_intent` (it's a keyword hint, often wrong on noise).
2. **`decision`** — `auto` (safe for the agent to answer now) or `escalate` (needs a human).
   Escalate when: transactional/account-specific, time-critical, angry/complaint,
   safety/legal, or you're not confident the docs cover it.
3. **`notes`** — one phrase on *why* (this seeds your failure analysis & decision log).

Aim for 150–250 labelled. Keep the split stratified; over-sample the hard/rare cases.
Save your finished labels as `data/golden.csv` (don't overwrite the candidate file).
