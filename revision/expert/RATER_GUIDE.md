---
title: "Expert Review of Answers to Medical Imaging Questions: Rater Guide"
---

# Rater guide

Thank you for helping. This is a short reviewing task for a research paper on an AI system that answers students' questions about recorded lectures in an undergraduate medical imaging course (CT, MRI, ultrasound, nuclear and optical imaging, and the related signal processing).

You will review **100 questions**. For each question, two AI systems have each written a short answer. You will judge the question, the reference answer, and the two AI answers. Expect it to take **3 to 4 hours**. You can stop and continue later; the workbook saves your progress.

## What you receive

- **Your workbook:** `ALIVE_expert_review_raterN.xlsx`. It has two sheets, *Instructions* and *Items*. You work in *Items*.
- **The course slides:** `All_Lectures_Slides.pdf`, all 23 lectures in one file (shared as a link). Use the bookmarks (Lecture, then Slide) to jump to the lecture named in the workbook's *Lecture* column; each page also shows a small "Lecture N - Slide k" label. You may consult the slides at any time.

## Ground rules

1. **Blinding.** The two answers are labelled **A** and **B**. Which system wrote A and which wrote B was decided at random for each item. Please don't try to work out which system is which. Judge each answer on its own merits.
2. **Work independently.** A second expert is rating the same items. Please don't discuss the items or your ratings with them, or with anyone on the research team, until you have both returned your workbooks.
3. **Allowed sources.** You may use your own expertise, the lecture context shown in the workbook, and the course slides. Please don't use AI chatbots or web searches. We want your independent professional judgement.
4. **Edit only the rating columns.** These are the columns from *Q answerable…* through *Comment*. Please leave the other columns unchanged.

## What each row shows

| Column | Content |
|---|---|
| Item | Item number, 1 to 100 |
| Difficulty | Basic, intermediate or advanced, as labelled when the question was generated |
| Lecture | The lecture the question comes from (match it to the slide PDF) |
| Lecture context | The lecture text the AI systems were given when answering |
| Question | A student-style question |
| Reference answer | An answer written by an AI model when the question was created. **It may be wrong.** Part of your job is to check it. |
| Answer A, Answer B | The two AI answers being compared |

## How to rate one item

Work across the row in this order:

1. Read the **question**, then the **lecture context** and the **reference answer**.
2. Fill in the three item-level fields (Y/N).
3. Read **Answer A** and fill in its three fields.
4. Read **Answer B** and fill in its three fields.
5. Give your **preference**.
6. Add a **comment** if anything was unclear or borderline. A few words is enough.

## The fields

### Item-level fields (Y or N)

**Q answerable from lecture?**
Y if the question can be answered from the material in this course's lectures. N if it relies on knowledge outside the course, is ambiguous, or does not make sense.

**Reference correct?**
Y if the reference answer is factually and conceptually correct according to the lecture material and your expertise. Small wording issues do not make it incorrect. A wrong formula, a wrong relationship or a false statement does.

**Context supports reference?**
Y if the *lecture context* shown in the row contains the information needed to arrive at the reference answer. N if the context is about something else or is missing the key information. Answer this even if the reference itself is wrong.

### Answer-level fields (fill in for A and for B)

**Correctness (C / P / I)**
Judge the answer **against the lecture material and your own expertise, not against the reference answer**. A correct answer that differs from the reference is still correct.

- **C (correct):** accurate and answers the question.
- **P (partial):** accurate as far as it goes but incomplete, or mostly right with a minor inaccuracy.
- **I (incorrect):** contains a substantive error, contradicts the lecture, or does not answer the question.

**Error type.** This is the most serious error in the answer. Use exactly one of these:

| Error type | Use when the answer… | Example |
|---|---|---|
| none | has no error (use with C) | n/a |
| conceptual | states a wrong concept, definition or relationship | says T2 relaxation is spin–lattice relaxation |
| mathematical | has a wrong formula, derivation or quantitative relationship | gives the Larmor frequency as proportional to 1/B0 |
| causal | gives a wrong mechanism, cause or effect | says ring artifacts come from patient motion instead of detector miscalibration |
| numerical | has a wrong number or unit | gives CT number of water as 1000 HU |
| other | is off-topic, unanswered, truncated or self-contradictory | an empty answer, or one about a different modality |

A *P* answer may have *none* as its error type if it is only incomplete. If it contains a minor inaccuracy, record the type of that inaccuracy.

**Clarity (1 to 5).** How clear the explanation is for an undergraduate taking this course, regardless of correctness.

| Score | Meaning |
|---|---|
| 5 | Clear and well organised; a student would understand it on first reading |
| 4 | Clear, with minor awkwardness or a little unnecessary detail |
| 3 | Understandable, but vague, wordy or poorly ordered |
| 2 | Hard to follow; key terms are unexplained or the logic jumps |
| 1 | Confusing or misleading |

### Preference (A, B or Tie)

Which answer would you rather a student received? Weigh correctness first, then clarity. Use **Tie** when you have no real preference.

## Special cases

- **The reference is wrong.** Mark *Reference correct?* N, then rate both answers on their own correctness as usual.
- **An answer is empty or cut off mid-sentence.** Rate it I with error type *other*.
- **The lecture context is irrelevant.** Mark *Context supports reference?* N. Rate correctness from the slides and your own knowledge.
- **Both answers are identical or almost identical.** Rate each one, and give Tie as the preference.
- **You are unsure.** Give your best judgement and explain briefly in *Comment*. Please don't leave fields blank.

## When you finish

1. Check that every rating cell in rows 1 to 100 is filled in.
2. Save the file as `ALIVE_expert_review_raterN_<your initials>.xlsx`. Please keep it as an Excel (.xlsx) file; do not save or export it as CSV, which breaks the long text cells.
3. Email it back to the contact below.

## Acknowledgement and data use

Your ratings will be analysed together with the other expert's ratings. Results are reported only in aggregate: agreement between raters and comparisons between the two systems. With your permission, you will be named in the paper's acknowledgements as an expert reviewer. Please tell us if you prefer not to be named.

**Questions:** Md Zabirul Islam, Rensselaer Polytechnic Institute (islamm11@rpi.edu)
