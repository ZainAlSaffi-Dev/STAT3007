# Zain's script

This is the script for slides 16, 17, 19 and 20 of the seminar: Tier 1, its limits, and Tier 3 with its models. Each section is one paragraph to read aloud. Together they run to just under two minutes at the deck's pace of 150 words a minute. The numbers on slides 16 to 19 come from the report's `numbers.tex`, through `_variables.yml`. The ladder of models on slide 20 comes from `model_fitting/4_nonlinear_exploration.ipynb`, Part 3, on the same 355 runs, and its kernel ridge matches the report.

## The script

### Slide 16. Tier 1: is there a threshold? (about 0:34)

Our first test asks whether grokking happens when a statistic hits a fixed value. If it did, that statistic would read about the same at every run's grokking step. A statistic that barely moves would pass that check too, so we read each run twice: at its own grokking step, and at another run's step as a placebo. A real threshold spreads out at the wrong step. Shape spreads the most, 1.71 times, scale 1.29 times, and alignment only 1.09, which is close to one.

### Slide 17. No statistic is a fixed threshold (about 0:18)

That tie only holds at a fixed width. Across widths, no statistic takes the same value at grokking. Wider networks grok with much less change of shape, and yet they grok later. So shape matters more than scale, but shape alone does not set the time.

### Slide 19. Tier 3: which statistic predicts best? (about 0:28)

Next, can the kernel, read early, tell us when a run will grok? We read scale, shape and alignment at step 2,430, before any run groks, and predict the grokking time on cells the model has never seen. Alignment explains almost nothing. Shape alone explains the most, at 0.35, and adding scale lifts it to 0.51. So shape is the best single reading, but it needs scale beside it.

### Slide 20. Tier 3: which model fits best? (about 0:36)

So which model fits best? We climbed a ladder, from straight lines up to kernel ridge. A straight line in the three statistics explains 0.31, and taking the log of shape lifts it to 0.36. Adding a square of scale jumps to 0.50, and kernel ridge on all three reaches 0.55, the best we found. So this equation gets most of the way, with two bends: the log of shape and a square of scale. The settings still reach 0.93, while the kernel can be measured in any network.

## Questions

These are not part of the timed talk.

- **Why the log of R?** We fitted R as it is and log R in straight lines and in kernel ridge, and log R predicted better on held-out cells. Almost half the runs have R below 0.01, and the log spreads them out.
- **Did you try the log of S?** No. S entered on its own scale, and a square term and splines let its curve bend.
- **Do the statistics interact?** Only a little among runs that grok. A curve for each statistic reaches 0.52, and adding their products reaches 0.54. Each gain above 0.50 is about 0.02, which is close to our noise level.
- **Which statistic matters most?** Shape. Without it, kernel ridge falls from 0.55 to 0.23.
- **Why hold out whole cells?** The five seeds of a cell are close copies. Seed noise sets a ceiling of 0.96 on R squared, and the statistics say nothing about which seed of a cell groks first.
- **Does the kernel beat weight decay?** No. On an unseen alpha, trees on width and weight decay reach a concordance of 0.79, and weight decay alone reaches 0.72. Trees on the three statistics reach 0.70.
- **What is kernel ridge regression?** It predicts a run's time from the runs that look most like it, with a Gaussian kernel that sets what "look like" means. That lets it follow curves and interactions without us choosing their form.
- **What are the p values in Tier 1?** Below 0.0005 for shape and for scale, and 0.029 for alignment, over 2,000 shuffles.
