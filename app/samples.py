"""Curated example articles for the *Analyze* page.

These are **written for demonstration** -- they are not scraped from any
publisher. Each one is labelled with the behaviour it is meant to trigger so a
reader can check whether the model reacts the way the page claims it should.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SampleArticle:
    """One demo document."""

    key: str
    title: str
    body: str
    expected: str
    note: str


SAMPLES: list[SampleArticle] = [
    SampleArticle(
        key="Policy briefing",
        title="Finance ministry outlines three-year plan for public school digital upgrades",
        body=(
            "The finance ministry published a three-year implementation plan on Tuesday that "
            "allocates funding for broadband equipment, teacher training and device maintenance "
            "across public schools. The document states that districts will receive funds in two "
            "tranches, and that progress will be reviewed by a parliamentary committee each year. "
            "Ministry officials said the schedule depends on annual budget approval and that the "
            "first tranche is expected to reach districts before the next academic year begins. "
            "Teachers' associations have asked for more detail on the training budget."
        ),
        expected="Real",
        note="Neutral register, named institutions, attributed statements, no emotional language.",
    ),
    SampleArticle(
        key="Research readout",
        title="Astronomers report a nearby galaxy with an unusually high rate of star formation",
        body=(
            "A team of astronomers using the same survey data for three consecutive seasons reports "
            "that a galaxy about 40 million light years away forms stars at roughly four times the "
            "rate of a typical spiral galaxy. The finding is based on ultraviolet observations that "
            "were peer reviewed before publication. The lead author cautioned that the sample covers "
            "a single region and that the estimate will be revised as additional seasons are "
            "analysed. The dataset and analysis code have been made publicly available."
        ),
        expected="Real",
        note="Hedged claims, explicit limitations, peer review and data availability mentioned.",
    ),
    SampleArticle(
        key="Court ruling",
        title="Appeals court narrows the scope of the data-transfer ruling issued last year",
        body=(
            "The appeals court on Monday set out a narrower reading of last year's data-transfer "
            "decision, clarifying that the ruling applies to transfers between companies in the same "
            "corporate group. Two of the three judges concurred with the narrower scope. The court "
            "remanded the case to the trial judge for a fresh assessment of remedies. Counsel for the "
            "regulator said the decision would be studied before any appeal is filed."
        ),
        expected="Real",
        note="Procedural language, specific procedural facts, no superlatives.",
    ),
    SampleArticle(
        key="Viral health claim",
        title="SHOCKING: Drinking ten cups of coffee every hour makes you invisible for five minutes",
        body=(
            "Scientists have CONFIRMED that drinking ten cups of coffee every hour makes people "
            "invisible for five minutes! The discovery has SHOCKED the entire world and you won't "
            "believe what happens next. Doctors are FURIOUS and big pharma doesn't want you to know "
            "the truth!!! Share this before it is DELETED!!!"
        ),
        expected="Fake",
        note="Sensational headline, shouting, exclamation spam, no named study or institution.",
    ),
    SampleArticle(
        key="Political outrage bait",
        title="You won't believe what this SECRET government agency has been hiding from you",
        body=(
            "What if the truth about the last election was buried on purpose? Exposed: the "
            "terrifying conspiracy they do NOT want you to know about. Top 10 reasons the officials "
            "are panicking right now, and what happens next will leave you speechless. Viral: this "
            "must-see report has been banned in four countries!!!"
        ),
        expected="Fake",
        note="Clickbait phrases, conspiracy framing, no verifiable specifics.",
    ),
    SampleArticle(
        title="Miracle fruit cures all illness overnight, doctors cannot explain why",
        key="Miracle fruit",
        body=(
            "A miracle fruit discovered in the Amazon is said to cure every illness overnight, and "
            "medical experts are left speechless. Amazing new research reveals the incredible secret "
            "that pharmaceutical companies have hidden for decades. This astonishing truth about your "
            "body will shock you. Act now before this is banned and you miss out!!!"
        ),
        expected="Fake",
        note="Universal cure claim, unverifiable superlatives, urgency and fear framing.",
    ),
]

SAMPLES_BY_KEY = {sample.key: sample for sample in SAMPLES}
