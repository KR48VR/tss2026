# The two Work and AI Map films

Two short films about the map, made in code from real recordings of the map being used. Neither has a voice-over: everything is on screen, so they work with the sound off, and the music is original (generated in code, nothing to license).

| Film | Length | For | Where it fits |
|---|---|---|---|
| 1. The Work and AI Map | 1 min 40 s | anyone who might use, share or fund the map | a website, a social post, the opening of a talk, a funding conversation |
| 2. How we built it together | 5 min 12 s | students, teachers, teams starting to work with AI | a class or workshop on working with AI, shown whole and then chapter by chapter |

Both start and end on the same title card, so a paused film works as a slide.

## Why these lengths

**Film 1** follows the usual length for a short promotional film: long enough to show one job, the whole economy, who it serves, why it can be trusted, and the ask; short enough to be watched to the end on a phone.

**Film 2** is kept under six minutes, in six chapters of under a minute each, because of two pieces of research on teaching videos:

- Guo, Kim and Rubin (2014) studied 6.9 million viewing sessions of edX course videos. However long the video, the typical viewer watched at most about six minutes; shorter videos held people better. [Paper](https://www.cs.rochester.edu/hci/pubs/pdfs/edX-MOOC-video-production-and-engagement_LAS-2014.pdf)
- Brame (2016) sums up what helps people learn from video: keep it short; mark what matters on screen (signalling); break it into parts (segmenting); cut whatever does not serve the point (weeding); and give viewers something to do, such as questions to think about. [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC5132380/)

So the film has numbered chapter cards (segmenting), one idea on screen at a time with the key words picked out (signalling), quiet music and no decoration (weeding), and a question at the end. The pause questions below are the "something to do".

## Film 2: chapters and pause questions

Show the film once straight through. Then replay it a chapter at a time, pausing on each chapter's last frame for the question.

| Starts at | Chapter | Pause question |
|---|---|---|
| 0:09 | 1. One question | What did King bring that Claude could not? Could the project have started without it? |
| 0:51 | 2. Three ways of working | Think of a task you have given an AI. Was it a tool request, a teammate conversation, or something you left it to do alone? |
| 1:48 | 3. Use it, then ask | Why did the best questions come after King used the map himself? |
| 2:39 | 4. Question the evidence | What is the difference between asking for a feature and challenging the evidence? Which is harder to do? |
| 3:24 | 5. Check everything | Which of the problems the checks found would you have noticed by eye? |
| 4:03 | 6. Grow, and keep the record | Why keep a record of who did what? Who is it for? |
| 4:33 | Five things to take away | Which of the five would change how you work with AI tomorrow? |
| 4:54 | To discuss | Which of King's questions changed the work most, and why? What would you have asked? |

## Film 2: talk-along script (optional)

For a teacher, or King, to say while the film plays or between chapters. It adds to the screen rather than repeating it.

1. **One question.** The project began with one message from King. Everything else, from the first download to this film, came from that question and from King's judgment about what came back. Claude did the building; King decided what mattered.
2. **Three ways of working.** Every step of the project is on record, and each one is marked by how it happened. Sometimes King simply asked for something. Sometimes the two reasoned it through and the plan changed. Sometimes one of them worked alone for a while. None of the three is better; they do different jobs.
3. **Use it, then ask.** King did not ask how the search worked until he had tried it. That question exposed a real weakness, and the fix took search through two more versions, ending with search by meaning.
4. **Question the evidence.** The most important question of the whole project asked nothing new of the map. It asked whether the evidence underneath it was still good enough, given how fast AI is moving. The answer changed three things at once. Then King handed Claude the order of the work: trust earned by the previous answer.
5. **Check everything.** Checking was done by scripts rather than by eye: numbers against their sources, every view in a real browser, accessibility rules on every view. The first round alone found nine groups of problems. There have been twenty-one rounds so far.
6. **Grow, and keep the record.** The map grew from one country to sixteen, plus thirty-two in Europe. Where a website refused a download, the record says so. And the record of who did what, step by step, is what this film was made from.

## Where the facts come from

Every fact in film 2 comes from the project's companion page (`site/companion.html`): the brief, the step-by-step record of 96 steps, the counts of exchanges (31 as a tool, 35 as teammates), the three versions of search, the first round of checks, and the twenty-one rounds since. King's quoted messages have their spelling tidied (for example "mving" to "moving"); the wording is otherwise his.

Every number in film 1 comes from the map's own build (Singapore, central case unless marked): AI could be doing about 42% of the paid work by 2035 and be involved in 87% of it; a quarter of it by mid-2030 (fast case late 2027, slow case early 2035); 18,796 tasks; 16 countries plus 32 in Europe.

The funding ask in film 1 (fresh AI evidence as each release lands, more countries, testing with workers, employers and teachers) is a suggestion for King to confirm or change.

## How the films are made

The sources are in `tools/film/`:

- `record.py` records the real map being used, frame by frame, with a drawn cursor (`python3 tools/film/record.py OUTDIR [names]`).
- `prep_field.py` and `prep_log.py` export the data the films draw (the 18,796 tasks laid out by job family; the 96 steps of the record).
- `engine.js`, `film1.js`, `film2.js` and `film.html` draw any moment of either film on a canvas, so a headless browser can render every frame the same way each time.
- `score.py` writes each film's music on the same bar grid as its script, so every scene change lands on a bar line.
- `render_all.sh` renders the frames in parallel, encodes the picture and sets the loudness (film 1 at -14 LUFS, film 2 at -16 LUFS).
- `qa.py` measures picture jumps, sound on the beat grid, loudness, silence at the end, and whether the first and last frames match.

To rebuild: serve a folder holding `film.html`, `engine.js`, the film scripts, `fonts.css` with its font files, `data/` and the recordings as `rec/` on port 8790, then run `tools/film/render_all.sh film1 100 OUT 3 film1.wav -14` (and `film2 312 ... -16`).
