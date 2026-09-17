# An AI Support Agent for Delta

**Delta · Twitter Customer Support**

## 1. Problem framing

### What "good" means for this brand

Delta being an airline company, safety and rules matter a lot. Most of their tweets are complaints from customers about flight cancellations, disrupted transfer schedules, lost bags, and refund issues. So for Delta, a good agent isn't the one that replies to the most tweets. It's the one that knows which tweets it should reply to and which ones it should pass to a human. In practice, that means:

1. **Safety is more important than responding to every single message.** If the agent provides a confident but incorrect answer regarding a refund, a change of booking, or any other safety-related matter, this is far worse than simply passing the message on to a human agent.
2. **Only say what Delta has actually said.** The agent's replies must be based on things that Delta has already stated previously. The agent must never invent any policies, figures, or promises.
3. **Each decision must be easy to verify.** The agent should, for every tweet, explain why it has made its choice — that is, what the customer wanted, how certain it was, on what it based its answer, and the reason for either replying or escalating. This makes the agent trustworthy.

### What I chose not to build

- **No multi-turn dialogue / memory.** Single message in, single decision out.
- **No live account / booking lookup.** Anything needing real customer state is escalated by design rather than faked. This is a feature, not a gap: those are exactly the messages a human should own.
- **No fine-tuned or LLM-based classifier in production.** A free-endpoint LLM few-shot classifier was built as a baseline; it did not beat TF-IDF on the set and was unreliable to run, so the LLM was reserved for reply phrasing only.

## 2. How I approached it — thought process & research grounding

I started from the literature rather than the code. I read six recent papers on production customer-support agents and pulled out the design decisions they agreed on, then built the smallest system that honors those decisions on this dataset and budget.

| Paper | What I took from it into this agent |
|---|---|
| **Building Customer Support AI Agents at 100M-User Scale: An Evaluation-Driven Framework** | The spine of the whole project: define a golden set, compare against baselines, and treat the eval harness as a first-class deliverable. My "proof > system" framing is this paper. |
| **Beyond IVR: Benchmarking Customer Support LLM Agents for Business-Adherence** | Adherence over cleverness — the agent must follow policy and refuse to overstep. Motivated the rule-based escalation gate and the "never invent policy" constraint on drafting. |
| **τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains** | Escalation is a controlled action with a measurable pass/fail, not a vibe. Motivated measuring escalation as its own task with precision/recall against human labels. |
| **Agent-in-the-Loop: A Data Flywheel for Continuous Improvement** | Bootstrap from cheap signal and log everything. Directly inspired the weak-supervision labeler (keyword rules → 7k labels) and the "why" trace attached to every response as future training data. |
| **LLM-Friendly Knowledge Representation for Customer Support** | Represent brand knowledge as retrievable (customer → resolution) pairs. This is exactly my 42k grounding-pair index that replies are grounded in. |
| **Benchmarking and Learning Real-World Customer Service Dialogue** | Real support text is messy, negation-heavy, and long-tailed. Motivated the negation-aware sentiment check and the per-intent (not just aggregate) reporting. |

A seventh consensus point — "judge with a measured judge" — is why reply quality is scored by an LLM-as-judge whose agreement with human labels is itself measured (Cohen's κ) before its scores are trusted.

## 3. System architecture

Four layers. The bottom two are built once, offline; the agent layer runs on every message; the LLM is an optional accelerator wired to reply-drafting only, never to a decision.

![End-to-end architecture](docs/diagram1_architecture.png)

*Figure 1 — End-to-end architecture. Raw tweets are prepared once into grounding pairs and clean messages; those bootstrap a TF-IDF intent classifier (via weak supervision) and a cosine retriever; `SupportAgent.handle()` chains classify → retrieve → route → draft on every request. The dashed amber box is the only place a language model touches the pipeline.*

## 4. What happens when a message arrives

A single request travels left to right. Classification and retrieval only gather evidence; the router in the middle is the decision point — five rules, any one of which forces a human hand-off — and it splits the flow into the two branches that produce a reply.

![Message to response workflow](docs/diagram2_workflow.png)

*Figure 2 — Message → response workflow. The green branch answers directly from grounded history; the red branch writes a hand-off tailored to the message. Either way the payload carries the full reasoning trace.*

## 5. Results vs. baselines

Everything below is measured on the 220-message set, each model against at least two baselines. Intent uses 5-fold out-of-fold prediction (never scored on a row it trained on); escalation is measured with oracle intents to isolate the policy from classifier error.

### Task 1 — Intent classification (11 classes, accuracy & macro-F1)

| Model | Accuracy | Macro-F1 |
|---|--:|--:|
| Trivial baseline — majority class | 0.164 | 0.026 |
| Simple / production — TF-IDF (1–2gram) + LogReg | 0.491 | 0.426 |

A 16× lift in macro-F1 over the trivial baseline. An LLM few-shot classifier was built as the third baseline; free inference endpoints were exhausted mid-run, and it had not beaten TF-IDF, so TF-IDF was kept as production and the LLM reserved for reply drafting.

### Per-intent F1 — where it is strong and where it is not

| Intent | F1 | Support (n) |
|---|--:|--:|
| seating | 0.76 | 29 |
| baggage | 0.59 | 29 |
| technology | 0.57 | 11 |
| flight_disruption | 0.49 | 36 |
| loyalty | 0.47 | 9 |
| airport_gate | 0.46 | 14 |
| other_non_actionable | 0.44 | 28 |
| booking_fare | 0.36 | 24 |
| service_feedback | 0.35 | 29 |
| refund_compensation | 0.20 | 8 |
| safety_legal | 0.00 | 3 |

### Task 3 — Escalation decision (the part that got the most care)

| Model | Acc | Auto P | Auto R | Esc P | Esc R |
|---|--:|--:|--:|--:|--:|
| Trivial — escalate all | 0.691 | — | — | 0.691 | 1.000 |
| Trivial — auto all | 0.309 | 0.309 | 1.000 | — | — |
| Rule-based policy (ours) | 0.786 | 0.657 | 0.647 | 0.843 | 0.849 |

The policy beats the strongest trivial baseline (escalate-everything, 69.1%) by ten points, while catching 84.9% of messages that truly needed a human and being right 84.3% of the time it pulls that trigger. Escalate-everything only looks respectable because the golden set is escalation-heavy — but it auto-handles nothing, so it is not an agent. Tightening the grounding rule (R5, 0.12→0.38) raised escalate-recall from 0.78 to 0.85.

### Task 2 — Reply quality

Replies are scored by an LLM-as-judge on grounding, correctness, and tone, with a judge–human agreement check (Cohen's κ) validating the judge before its scores are trusted. This task needs a live model; with free endpoints exhausted, the honest status is that the harness and rubric are in place and run on demand, but there is **no headline reply-quality number to stand behind yet**. What is guaranteed structurally: no reply is sent from a below-0.38 grounding match, so the failure mode of confidently pasting an unrelated historical reply is closed off by construction.

## 6. Failure analysis — top 5 modes, real examples & hypotheses

### F1 — Model struggles with intents that have very few examples
- **Example.** "Is it safe to fly with my lithium battery pack in carry-on?" The correct intent is `safety_legal`, but the model predicted `baggage` with 73% confidence.
- **Hypothesis.** `safety_legal` has 3 training examples and `refund_compensation` has 8. A linear model cannot carve a reliable boundary from single-digit support, so these classes get absorbed by lexically-similar neighbors ("carry-on" → baggage).
- **Status / fix.** Mitigated, not solved: even though the model chose the wrong intent, the system correctly decided to escalate the message instead of answering automatically. The low grounding score (0.27, below our 0.38 threshold) helped prevent a potentially incorrect answer.

### F2 — A strong keyword can make the model choose the wrong topic
- **Example.** "How many SkyMiles do I need for a free checked bag?" → The correct intent is `loyalty`, because the main question is about SkyMiles. However, the model classified it as `baggage` with 55% confidence.
- **Hypothesis.** Our TF-IDF model mainly looks at which words and phrases are associated with each intent. The phrase "checked bag" is strongly associated with baggage, so it overpowered the word "SkyMiles", which should have indicated loyalty. The model does not really understand that SkyMiles is the main topic and the checked bag is just part of the question.
- **Status / fix.** Fixed with a small high-precision override: unambiguous brand terms (SkyMiles, Medallion, SkyClub) win outright. Now classified `loyalty` (auto-OK) and answered directly. A learned model would need many more examples to encode this.

### F3 — The model can retrieve an unrelated old answer
- **Example.** "Can I get a credit card discount on my next booking? I have many points on my ICICI credit card" → The system found an old customer-support response with a similarity score of only 0.30. The old response was actually about risk-free cancellation, so it was unrelated. It also contained another customer's name.
- **Hypothesis.** The grounding floor was 0.12, so a 0.30 (off-topic) nearest neighbor was treated as "well-grounded" and pasted verbatim. Classic RAG failure: low similarity is still the closest, but the closest result is not always a relevant result.
- **Status / fix.** Fixed by raising the minimum similarity threshold to 0.38 (calibrated: genuine matches score 0.40–0.62, this one sat alone at 0.30). Below the bar the agent now escalates instead of parroting. This single threshold closes the most dangerous reply failure.

### F4 — Some intents are very similar, so the model confuses them
- **Example.** "Can I change my flight to tomorrow morning?" → The model correctly classified this as `booking_fare` in this example, but it frequently confuses `booking_fare ↔ flight_disruption` and `service_feedback ↔ other_non_actionable`.
- **Hypothesis.** These intents use very similar words. For example, a customer saying "My flight was changed to tomorrow." and "Can I change my flight to tomorrow?" can look very similar to a simple text-based model, even though one is a flight disruption and the other is a customer-requested change.
- **Status / fix.** Partly absorbed by routing (both members of each pair are handled similarly), but it caps intent quality. The real fix is more labels or an embedding model — deliberately out of scope here.

### F5 — Some incorrect training labels come from our own rules
- **Example.** "…many points on my ICICI credit card" still reads as `loyalty` because the keyword rule maps "points" → loyalty, though these are a third party's points, not SkyMiles.
- **Hypothesis.** The classifier inherits the biases of the keyword labeler that bootstrapped it. Any gap or over-broad rule in the ~7k weak labels becomes a systematic error the gold set is too small to fully correct.
- **Status / fix.** Contained by the escalation policy (the message escalated anyway via weak grounding), but it shows the ceiling of weak supervision: the model is only as sharp as its labeling functions. Fix path is auditing/adding labeling functions, tracked in the decision log.

## 7. What is misleading about my headline number?

- **"94% confidence" in the UI is not accuracy.** It is the model's own predicted probability. On held-out golden data the real intent accuracy is 49% (macro-F1 0.43). Confidence is a routing signal, not a correctness guarantee.
- **The 78.6% escalation score assumes perfect intents.** Task 3 is measured with oracle intents to isolate policy quality. End-to-end, classifier errors propagate into routing, so live escalation accuracy is lower than the policy's own ceiling.
- **Macro-F1 averages all 11 intents equally.** Two tiny-support classes (n=3, n=8) with near-zero F1 drag the mean down hard. Weighted-F1 is 0.48, and the common intents customers actually send are well above the headline — one number hides that the agent is strong where the volume is and weak in the long tail.
- **Accuracy looks inflated against a 16% baseline** only because the classes are imbalanced. The honest comparison is macro-F1 vs the majority baseline (0.43 vs 0.03), not raw accuracy vs chance.

The one number I would actually stand behind: **escalation recall of 0.85 with 0.84 precision** on the hand-labelled set — because for this brand, catching the messages that need a human is the metric that maps to real-world cost, and it is measured against human labels, not self-reported.

## 8. What it lacks & what I'd do next

- **Modest intent accuracy (macro-F1 0.43):** fine for routing, not for unattended auto-replies at scale.
- **Long tail unlearned:** `safety_legal` and `refund_compensation` need either targeted labels or a small embedding classifier.
- **Single-turn, no memory or account lookup:** account-state messages are escalated rather than resolved.
- **Reply-quality headline pending a stable LLM:** the judge + κ harness is built and runs on demand; it needs a funded endpoint to produce a number.

**Next steps, in priority order:**

1. **Add more training examples:** around 30 more examples for each of the two underrepresented intents.
2. **Use sentence embeddings instead of TF-IDF:** then run the same evaluation to see if performance improves.
3. **Evaluate the AI responses:** use an LLM judge to measure answer accuracy, relevance, and tone.
4. **Log the agent's reasoning and decisions:** use these real-world cases to continuously improve the system over time.
