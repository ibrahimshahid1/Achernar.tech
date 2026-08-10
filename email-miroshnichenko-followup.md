# Follow-up to Prof. Miroshnichenko

**Send as a reply in the original May 20 thread** so his message and your first
email stay visible underneath. Don't start a new thread.

---

Dear Professor Miroshnichenko,

I hope your travels went well. I'm the junior at Middle College at UNCG who
wrote to you in May about contributing computationally to observational work,
and you kindly said you'd think about what might be possible. That was right as
you were leaving for Asia, so I wanted to follow up now that the fall semester
and the TCO observing season are both about to start.

I've spent most of the time since then on the applied side of this, and I think
I can be more specific than I was in my first email about where I might be
useful.

The work of yours I keep coming back to is the search for new objects with the
B[e] phenomenon — the 2025 *Galaxies* paper where you and your co-authors built
photometric criteria from optical and near-infrared colour indices, found around
forty candidates, and then used spectroscopy to sort out their classification.
That select-then-confirm loop is exactly the setting where cross-modal embedding
methods tend to earn their keep. I've been working with AstroM3 (Rizhko & Bloom
2024), which learns a shared representation across spectra, photometry, and
metadata, so objects can be ranked by similarity to confirmed examples rather
than by fixed colour cuts.

What makes it appealing to me is that it's testable against work you've already
done: your existing candidate list and the objects you followed up
spectroscopically are a ready-made validation set. If the method is any good it
should recover them, and anything it surfaces *outside* your cuts is a
prediction that a spectrum can check rather than a claim I'd be asking you to
take on faith. I'd want to run it on a small subset first, so you can judge
whether the output is worth anything before committing more data or time.

If the TCO échelle archive from the long-term binary program is more useful to
start with, I'd be glad to work with that instead — I mention the B[e] search
only because it seemed like the cleanest place to test whether I'm actually
adding anything.

I should also be upfront about a separate side project, since it's mine and not
yours: I've been building an imaging control system that does real-time focus
quality estimation, cloud and obstruction detection, and exposure
recommendation. Two things from an operating observatory would help it
enormously, and both cost close to nothing:

- Any autofocus or engineering logs TCO has kept — the focuser position,
  HFR/FWHM, and temperature records that focus routines write out each night.
  These are usually treated as throwaway, but they're precisely the training
  data this kind of model needs, and I have no other source for them.
- If it were ever possible, fifteen minutes of engineering time on a bright star
  to step the focuser through a defocus sequence. That needs neither dark time
  nor good conditions, so it could go on a night that's otherwise unusable.

Neither is a priority over the archive work. I mention them only because you're
in a position to make them trivial and nobody else I have access to is.

I'm also happy to just take direction. If there's data reduction, pipeline work,
or analysis that's been sitting on the shelf because nobody has had time for it,
I'd rather be useful on something you already need than propose something new.

Would you have twenty minutes in the next few weeks to talk? I can work around
my class schedule and come to campus.

Thank you again for your time.

Sincerely,
Ibrahim Shahid

---

## What I verified before writing this

Everything the email claims about his work is confirmed from published sources,
not inferred:

| Claim | Confirmed by |
|---|---|
| Hot stars, circumstellar matter, B[e] phenomenon is his field | Co-author, *Galaxies* 2025, 13, 67, affiliated "Dept. of Physics and Astronomy, UNC Greensboro" |
| The photometric-criteria paper exists and says what you say it says | Abstract: "We developed photometric criteria to search for candidate objects with this phenomenon based on a combination of optical and near-infrared color indices and found nearly 40 objects that satisfy these criteria. Spectroscopy of the candidates allows us to make more confident conclusions on their classification." |
| TCO runs an échelle spectrograph | Fiber-fed eShel, R ≈ 12,000, continuous 3800–7900 Å, on the 0.81 m |
| He leads a long-term binary program there | "a long-term program of binary systems observing since 2013 at the 0.81 m telescope of the Three College Observatory… reviewed by Miroshnichenko and colleagues" |
| The archive is substantial | One 2025 paper alone uses 71 TCO échelle spectra spanning 2014–2025; TCO spectroscopy of Polaris Aa covers 2015–2024 |
| He directs TCO | His own email signature |

Two things I corrected from my first draft: the binary program is
**spectroscopic**, and the **photometry** belongs to the B[e] search work — so
the email no longer blurs them. And the B[e] paper is a better hook than the
binary archive, because their hand-built colour cuts and confirmed candidates
give you an honest benchmark to be measured against.

## Why it's written this way

- **No apology for the gap.** He said "a few days" and it's been eleven weeks.
  Drawing attention makes replying harder, not easier. The semester and
  observing season restarting is a clean, true reason to write now.
- **One-line re-identification.** Eleven weeks and an international trip is long
  enough that "Ibrahim" alone may not land.
- **Leads with his research.** Your first email listed what you can do and
  closed with an open offer — hard for a busy person to act on, and the likeliest
  reason it stalled.
- **Offers falsifiability.** Proposing a method that can be checked against work
  he's already published is far more credible than promising it will work.
- **The side project is labelled as yours.** Asking for data that serves your own
  project is fine. Disguising it as collaboration isn't, and he'd see through it.
- **The "take direction" paragraph is the real closer.** Faculty have a backlog
  of unglamorous work; volunteering for it beats any proposal.

## Optional additions

- If Dr. Aarnio has taught you or you've spoken with her, one clause — "Dr.
  Aarnio suggested I follow up" — is worth a paragraph. **Only if true.**
- If Achernar is on a public GitHub repo, one link after the side-project
  paragraph. Don't attach anything.
- If you want to cite the paper precisely: Turekhanova et al., "Searching for New
  Objects with the B[e] Phenomenon," *Galaxies* **13**, 67 (2025),
  doi:10.3390/galaxies13030067. Read at least the abstract before you send this,
  in case he asks.
