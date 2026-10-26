# Zain's script

This is the script for slides 16, 17 and 19 of the seminar: Tier 1, its limits, and Tier 3. Each section is one paragraph to read aloud. Together they run to about two minutes at the deck's pace of 150 words a minute. The numbers on these slides come from the report's `numbers.tex`, through `_variables.yml`. The linear regression numbers on slide 19 come from `presentation/fit_tier3_linear.py`, which fits least squares on the same 355 runs and the same folds as the report's kernel ridge. The two slides on the models are now backup slides.

## The script

### Slide 16. Tier 1: is there a threshold? (about 0:40)

Okay, so first we test whether grokking happens when a statistic hits a fixed value. Think of water, which boils at 100 degrees in any pot. If grokking had a boiling point, the statistic would read the same in every run at the moment that run groks. But a thermometer stuck on one number also reads the same every time. So we also read each run at the step where some other run grokked. If the statistic is a real boiling point, those readings should be much more spread out. Shape spreads out the most, 1.71 times. Scale spreads 1.29 times, and alignment hardly at all.

### Slide 17. No statistic is a fixed threshold (about 0:20)

But water only boils at 100 degrees at sea level. Up a mountain it boils at a lower temperature. Width acts like the mountain. Once we change the width, no statistic has the same value at grokking. Wide networks grok with much less change of shape, and yet they grok later. So shape alone does not decide when a network groks.

### Slide 19. Tier 3: which statistic predicts best? (about 1:00)

Can the statistics still forecast when a run will grok? It is like guessing a runner's marathon time from their first few kilometres. We read them at step 2,430, before any run groks, and predict settings the model has never seen. The light bars are plain linear regression, which gives each statistic one fixed slope. The dark bars are kernel ridge, which predicts a run from the runs most like it, a bit like pricing a house from similar houses. In both models alignment explains almost nothing, and shape alone explains about a third. In a straight line, adding scale to shape adds nothing. In kernel ridge, adding scale takes it from a third to a half. So scale helps once the model can bend it or combine it with shape. A straight line with scale squared, or with the products of the statistics, closes most of the gap.

## Questions

These are not part of the timed talk.

- **Didn't boosted trees do well?** It depends on what they read. On the three statistics at step 2,430 they reach only 0.39, below kernel ridge at 0.55. On the 25 early readings of the statistics they reach 0.74, and on the settings 0.93. The history mostly records the settings. On an unseen alpha, trees on the three statistics rank runs at a concordance of 0.70, better than any statistic alone.
- **What is the spline?** Each statistic gets 7 cubic B-spline basis functions on 5 knots, and ridge regression fits their weights. The square is the simplest curve, since its slope, beta plus 2 beta prime z, changes in a straight line.
- **What are the betas?** Each beta is fitted by least squares on the training cells, and each z is a statistic rescaled to mean 0 and spread 1. The notebook did not save the fitted betas, so we quote only the form and its score.
- **Why the log of R?** We fitted R as it is and log R in straight lines and in kernel ridge, and log R predicted better on held-out cells. Almost half the runs have R below 0.01, and the log spreads them out.
- **Did you try the log of S?** No. S entered on its own scale, and a square term and splines let its curve bend.
- **Do the statistics interact?** We cannot tell. A straight line on all three reaches 0.36. Adding scale squared lifts it to 0.50, adding the products of the statistics lifts it to 0.54, and adding both also gives 0.54. A bend in scale and an interaction with shape explain the same runs, so the data cannot separate them.
- **Which statistic matters most?** Shape. Without it, kernel ridge falls from 0.55 to 0.23.
- **Why hold out whole cells?** The five seeds of a cell are close copies. Seed noise sets a ceiling of 0.96 on R squared, and the statistics say nothing about which seed of a cell groks first.
- **Does the kernel beat weight decay?** No. On an unseen alpha, trees on width and weight decay reach a concordance of 0.79, and weight decay alone reaches 0.72.
- **What is kernel ridge regression?** It predicts a run's time from the runs that look most like it, with a Gaussian kernel that sets what "look like" means. That lets it follow curves and interactions without us choosing their form.
- **What are the p values in Tier 1?** Below 0.0005 for shape and for scale, and 0.029 for alignment, over 2,000 shuffles.
